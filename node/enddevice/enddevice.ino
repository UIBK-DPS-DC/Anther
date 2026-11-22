
//  Heltec V3 – Class-Switchable LoRaWAN End Device
//
//  FPort 1: TELEMETRY – uplink sensor data
//  FPort 2: CONTROL   – class-switch command (0x10=A, 0x11=B, 0x12=C)
//
//  Class switches persist the target class + session to NVS, then reboot.
//  This avoids RadioLib's live class-transition edge cases (setClass() does
//  not fully re-arm RX timing/IRQ state) — a clean boot + session restore
//  reliably lands in the requested class every time.

#include <Preferences.h>
#include <RadioLib.h>
#include <esp_sleep.h>
#define DEBUG

#define LORA_SCK 9
#define LORA_MISO 11
#define LORA_MOSI 10
#define LORA_CS 8
#define LORA_RST 12
#define LORA_DIO1 14
#define LORA_BUSY 13

// NOTE: DR0/SF12 and DR1/SF11 do not work for class A, because of LDRO, which causes problems in this library.
#define LORA_SF 9
#define LORA_BW 125.0
#define LORA_CR 5
#define LORA_TX_POWER 14

#define FPORT_TELEMETRY 1
#define FPORT_CONTROL 2

#define CLASS_B_PING_PERIOD_MS 8000UL  // how often Class B "wakes" to check for downlinks
#define CLASS_B_PING_WINDOW_MS 300UL   // how long each listen burst lasts


SPIClass *spi = new SPIClass(HSPI);
SPISettings spiSettings(2'000'000, MSBFIRST, SPI_MODE0);
SX1262 radio = new Module(LORA_CS, LORA_DIO1, LORA_RST, LORA_BUSY, *spi, spiSettings);
LoRaWANNode node(&radio, &EU868);
Preferences prefs;

uint64_t joinEUI = 0x0000000000000000;
uint64_t devEUI = 0xe21ca312cb69528a;
uint8_t appKey[] = { 0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56,
                     0x78, 0x90, 0x12, 0x34, 0x56, 0x78, 0x90, 0x12 };
uint8_t nwkKey[] = { 0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56,
                     0x78, 0x90, 0x12, 0x34, 0x56, 0x78, 0x90, 0x12 };

enum DeviceClass : uint8_t { CLASS_A_MODE,
                             CLASS_B_MODE, // NOT IMPLEMENTED!
                             CLASS_C_MODE };
DeviceClass currentClass = CLASS_A_MODE;
unsigned long uplinkInterval_ms = 10000UL;

uint16_t counter = 0;
uint32_t lastUplink = 0;

struct PayloadBuilder {
  uint8_t buf[32];
  uint8_t len = 0;
  void clear() {
    len = 0;
  }
  void addUint8(uint8_t v) {
    buf[len++] = v;
  }
  void addUint16(uint16_t v) {
    buf[len++] = v >> 8;
    buf[len++] = v & 0xFF;
  }
};

struct PayloadParser {
  const uint8_t *buf;
  uint8_t len, pos = 0;
  PayloadParser(const uint8_t *b, uint8_t l)
    : buf(b), len(l) {}
  uint8_t readUint8() {
    return buf[pos++];
  }
};

void printHex(const uint8_t *buf, size_t len) {
  for (size_t i = 0; i < len; i++) {
    if (buf[i] < 16) Serial.print("0");
    Serial.print(buf[i], HEX);
    Serial.print(" ");
  }
  Serial.println();
}

void haltOnError(const char *msg, int code) {
#ifdef DEBUG
  Serial.printf("%s, code: %d\n", msg, code);
#endif
  while (true) delay(1000);
}

// Session persistence (NVS)
void saveSession() {
  prefs.begin("lorawan", false);
  prefs.putBytes("session", node.getBufferSession(), RADIOLIB_LORAWAN_SESSION_BUF_SIZE);
  prefs.putBytes("nonces", node.getBufferNonces(), RADIOLIB_LORAWAN_NONCES_BUF_SIZE);
  prefs.putUChar("class", (uint8_t)currentClass);
  prefs.putULong64("ul_interval_ms", (uint64_t)uplinkInterval_ms);
  prefs.end();
#ifdef DEBUG
  Serial.printf("[SAVE SESSION] Current Class %d\n", currentClass);
  Serial.printf("[SAVE SESSION] Uplink Interval %lu\n", uplinkInterval_ms);
#endif
}

bool loadSession() {
  prefs.begin("lorawan", true);
  if (!prefs.isKey("session") || !prefs.isKey("nonces")) {
    prefs.end();
    return false;
  }

  uint8_t nonces[RADIOLIB_LORAWAN_NONCES_BUF_SIZE];
  uint8_t session[RADIOLIB_LORAWAN_SESSION_BUF_SIZE];
  bool ok = prefs.getBytes("nonces", nonces, sizeof(nonces)) == sizeof(nonces) && prefs.getBytes("session", session, sizeof(session)) == sizeof(session);
  if (ok) currentClass = (DeviceClass)prefs.getUChar("class", CLASS_C_MODE);
  if (ok) uplinkInterval_ms = (unsigned long)prefs.getULong64("ul_interval_ms", 10000UL);
  prefs.end();
  if (!ok) return false;

  // Nonces must be restored before session.
  return node.setBufferNonces(nonces) == RADIOLIB_ERR_NONE && node.setBufferSession(session) == RADIOLIB_ERR_NONE;
}

void requestClassSwitch(DeviceClass cls) {
  currentClass = cls;
  saveSession();
#ifdef DEBUG
  Serial.printf("[CLASS] Switching to %d, rebooting...\n", cls);
#endif
  delay(50);
  ESP.restart();
  ESP.
}

// Downlink handling
void processControlDownlink(const uint8_t *data, size_t len) {
  if (len == 0) return;
  PayloadParser pp(data, (uint8_t)len);
  switch (pp.readUint8()) {
    case 0x10: requestClassSwitch(CLASS_A_MODE); break;
    case 0x11: requestClassSwitch(CLASS_B_MODE); break;
    case 0x12: requestClassSwitch(CLASS_C_MODE); break;
    case 0x20: {
      uint8_t interval_s = pp.readUint8();
      uplinkInterval_ms = (uint64_t)(interval_s * 1000UL);
      saveSession();
      break;
    }
    default:
#ifdef DEBUG
    Serial.println("[RX|CTRL] Unknown command"); 
#endif 
      break;
  }
}

void dispatchDownlink(int state, uint8_t fport, const uint8_t *data, size_t len) {
  if (state < 0) {
#ifdef DEBUG
    Serial.printf("[TX] Failed, code: %d\n", state);
#endif
    return;
  }
  if (state == 0) return;
  if (len == 0) {
#ifdef DEBUG
  Serial.println("len = 0");
#endif
    return;
  }
#ifdef DEBUG
  Serial.printf("[RX] FPort %u | %u byte(s): ", fport, (unsigned)len);
  printHex(data, len);
#endif

  if (fport == FPORT_CONTROL) processControlDownlink(data, len);
}

void sendPeriodicUplink() {
  PayloadBuilder pb;
  pb.clear();
  pb.addUint16(counter++);

  uint8_t dlBuf[256];
  size_t dlLen = sizeof(dlBuf);
  LoRaWANEvent_t eventDown{};
  int state = node.sendReceive(pb.buf, pb.len, FPORT_TELEMETRY, dlBuf, &dlLen, false, nullptr, &eventDown);
  dispatchDownlink(state, eventDown.fPort, dlBuf, dlLen);

  saveSession();
}

// Per-class execution
void classACommunication() {
  sendPeriodicUplink();
  uint64_t sleep_time = (uint64_t) node.timeUntilUplink();
  if ((uint64_t)uplinkInterval_ms > sleep_time) sleep_time = (uint64_t)uplinkInterval_ms;
#ifdef DEBUG
  Serial.printf("[SLEEP] Deep sleeping for %lu ms\n", sleep_time);
#endif
  esp_sleep_disable_wakeup_source(ESP_SLEEP_WAKEUP_ALL);
  esp_sleep_enable_timer_wakeup(sleep_time * 1000ULL);
  esp_deep_sleep_start();
}

void classCCommunication() {
  node.setClass(RADIOLIB_LORAWAN_CLASS_C);
  uint8_t dlBuf[256];
  size_t dlLen = sizeof(dlBuf);
  LoRaWANEvent_t eventDown{};
  int state = node.getDownlinkClassC(dlBuf, &dlLen, &eventDown);
  dispatchDownlink(state, eventDown.fPort, dlBuf, dlLen);
#ifdef DEBUG
  if (state > 0 && dlLen > 0)
    Serial.println("Recieved by getDownlinkClassC!");
#endif

  if (node.timeUntilUplink() > 0 || millis() - lastUplink < uplinkInterval_ms) return;
  lastUplink = millis();
  sendPeriodicUplink();
}

// Setup / Loop
void setup() {
  Serial.begin(115200);
  spi->begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_CS);

  int state = radio.begin(868.1, LORA_BW, LORA_SF, LORA_CR,
                          RADIOLIB_SX126X_SYNC_WORD_PRIVATE, LORA_TX_POWER, 8, 1.8, false);
  if (state != RADIOLIB_ERR_NONE) haltOnError("[Radio] begin failed", state);

  node.beginOTAA(joinEUI, devEUI, nwkKey, appKey);
  bool hadSession = loadSession();
  state = node.activateOTAA();

  if (hadSession && state == RADIOLIB_LORAWAN_SESSION_RESTORED) {
#ifdef DEBUG
    Serial.println("[LoRaWAN] Session restored");
#endif
  } else if (!hadSession && state == RADIOLIB_LORAWAN_NEW_SESSION) {
#ifdef DEBUG
    Serial.println("[LoRaWAN] Joined");
#endif
    saveSession();
  } else {
    haltOnError("[LoRaWAN] Activation failed", state);
  }

  // Repeatability: fixed datarate, no ADR-driven changes, no randomized CSMA
  // Duty cycle is left on.
  node.setADR(false); // ADR, can still be used. This just disables Backoff behaviour.
  node.setCSMA(0, 0, false);
  node.setDutyCycle(true, 0);

  uint8_t radioClass = currentClass == CLASS_A_MODE   ? RADIOLIB_LORAWAN_CLASS_A
                       : currentClass == CLASS_B_MODE ? RADIOLIB_LORAWAN_CLASS_B
                                                      : RADIOLIB_LORAWAN_CLASS_C;
  node.setClass(radioClass);
#ifdef DEBUG
  Serial.printf("[CLASS] Running as %d\n", currentClass);
#endif

  if (currentClass == CLASS_C_MODE) {
    // Class C's continuous RX only reliably arms after an uplink's RX1/RX2
    // cycle completes — without this, getDownlinkClassC() sees nothing.
    PayloadBuilder pb;
    pb.clear();
    pb.addUint16(counter++);
    uint8_t dlBuf[256];
    size_t dlLen = sizeof(dlBuf);
    LoRaWANEvent_t eventDown{};
    int state = node.sendReceive(pb.buf, pb.len, FPORT_TELEMETRY, dlBuf, &dlLen, false, nullptr, &eventDown);
    dispatchDownlink(state, eventDown.fPort, dlBuf, dlLen);
#ifdef DEBUG
    if (state > 0 && dlLen > 0)
      Serial.println("Recieved by normal sendReceive on start of the device!");
#endif
    saveSession();
    lastUplink = millis();
  }
}

void loop() {
  switch (currentClass) {
    case CLASS_A_MODE: classACommunication(); break;
    case CLASS_C_MODE: classCCommunication(); break;
  }
}
