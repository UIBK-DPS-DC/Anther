// Heltec V3 Class-C FUOTA multicast receiver for ChirpStack v4.
// Receives authenticated TS004 fragment data blocks and discards them after
// counting. This sketch does not reconstruct, store, verify, or install an image.

// This was generated using Copilot on the 28.8.2026.

#include <Preferences.h>
#include <RadioLib.h>
#include <esp_sleep.h>
#include <mbedtls/aes.h>
#include <string.h>

#define DEBUG

#define LORA_SCK 9
#define LORA_MISO 11
#define LORA_MOSI 10
#define LORA_CS 8
#define LORA_RST 12
#define LORA_DIO1 14
#define LORA_BUSY 13

#define ENERGY_MARKER_PIN 2
#define LORA_SF 9
#define LORA_BW 125.0
#define LORA_CR 5
#define LORA_TX_POWER 14

#define FPORT_TELEMETRY 1
#define FPORT_MCAST_SETUP 200
#define FPORT_FRAG 201
#define FRAG_DATA_BLOCK_CID 0x08

#define UPLINK_INTERVAL_MS 10000UL

#ifndef RADIOLIB_LORAWAN_MAC_DEVICE_TIME
#define RADIOLIB_LORAWAN_MAC_DEVICE_TIME 0x0D
#endif

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

struct MulticastSession {
  bool configured = false;
  bool scheduled = false;
  uint8_t groupId = 0;
  uint32_t address = 0;
  uint8_t appSKey[16] = {};
  uint8_t nwkSKey[16] = {};
  uint32_t minFrameCounter = 0;
  uint32_t maxFrameCounter = 0xFFFFFFFF;
  uint32_t startGps = 0;
  uint32_t timeoutSeconds = 0;
  uint32_t frequencyHz = 0;
  uint8_t dataRate = RADIOLIB_LORAWAN_DATA_RATE_UNUSED;
  uint32_t fragmentFrames = 0;
  uint32_t fragmentBytes = 0;
};

struct FragmentSession {
  bool active = false;
  uint8_t index = 0;
  uint16_t fragmentCount = 0;
  uint8_t fragmentSize = 0;
  uint8_t padding = 0;
};

struct AckOutbox {
  bool pending = false;
  uint8_t fport = 0;
  uint8_t length = 0;
  uint8_t data[8] = {};
};

MulticastSession multicast;
FragmentSession fragmentation;
AckOutbox ackOutbox;

uint16_t uplinkCounter = 0;
uint32_t lastUplinkMs = 0;
bool timeSynced = false;
uint32_t gpsAtSync = 0;
uint32_t millisAtSync = 0;

void haltOnError(const char *message, int code) {
#ifdef DEBUG
  Serial.printf("%s, code: %d\n", message, code);
#endif
  while (true) delay(1000);
}

uint32_t currentGpsTime() {
  if (!timeSynced) return 0;
  return gpsAtSync + (millis() - millisAtSync) / 1000UL;
}

void updateDeviceTime() {
  uint32_t seconds;
  uint16_t milliseconds;
  if (node.getMacDeviceTimeAns(&seconds, &milliseconds, false) == RADIOLIB_ERR_NONE) {
    gpsAtSync = seconds;
    millisAtSync = millis();
    timeSynced = true;
  }
}

bool encryptAesBlock(const uint8_t key[16], const uint8_t input[16], uint8_t output[16]) {
  mbedtls_aes_context context;
  mbedtls_aes_init(&context);
  int result = mbedtls_aes_setkey_enc(&context, key, 128);
  if (result == 0) result = mbedtls_aes_crypt_ecb(&context, MBEDTLS_AES_ENCRYPT, input, output);
  mbedtls_aes_free(&context);
  return result == 0;
}

bool multicastKeySelfTest() {
  const uint8_t testKey[16] = { 0x02, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10 };
  const uint8_t derivationBlock[16] = { 0x20 };
  const uint8_t expected[16] = { 0x26, 0x4F, 0xD8, 0x59, 0x58, 0x3F, 0xCC, 0x67,
                                 0x02, 0x41, 0xAC, 0x07, 0x1C, 0xC9, 0xF5, 0xBB };
  uint8_t derived[16];
  return encryptAesBlock(testKey, derivationBlock, derived) &&
         memcmp(derived, expected, sizeof(expected)) == 0;
}

bool deriveMulticastKeys(const uint8_t encryptedMcKey[16], uint32_t address,
                         uint8_t appSKey[16], uint8_t nwkSKey[16]) {
  // TS005 derives the LoRaWAN 1.1 McRootKey from AppKey using 0x20.
  uint8_t derivationBlock[16] = {};
  derivationBlock[0] = 0x20;
  uint8_t mcRootKey[16];
  uint8_t mcKeKey[16];
  uint8_t mcKey[16];
  uint8_t zeroBlock[16] = {};
  if (!encryptAesBlock(appKey, derivationBlock, mcRootKey) ||
      !encryptAesBlock(mcRootKey, zeroBlock, mcKeKey) ||
      !encryptAesBlock(mcKeKey, encryptedMcKey, mcKey)) {
    return false;
  }

  memset(derivationBlock, 0, sizeof(derivationBlock));
  derivationBlock[0] = 0x01;
  memcpy(derivationBlock + 1, &address, sizeof(address));
  if (!encryptAesBlock(mcKey, derivationBlock, appSKey)) return false;

  derivationBlock[0] = 0x02;
  return encryptAesBlock(mcKey, derivationBlock, nwkSKey);
}

void saveSession() {
  prefs.begin("fuota", false);
  prefs.putBytes("session", node.getBufferSession(), RADIOLIB_LORAWAN_SESSION_BUF_SIZE);
  prefs.putBytes("nonces", node.getBufferNonces(), RADIOLIB_LORAWAN_NONCES_BUF_SIZE);
  prefs.putBytes("multicast", &multicast, sizeof(multicast));
  prefs.putBytes("fragment", &fragmentation, sizeof(fragmentation));
  prefs.putUShort("ul_counter", uplinkCounter);
  prefs.end();
}

bool loadSession() {
  prefs.begin("fuota", true);
  if (!prefs.isKey("session") || !prefs.isKey("nonces")) {
    prefs.end();
    return false;
  }

  uint8_t session[RADIOLIB_LORAWAN_SESSION_BUF_SIZE];
  uint8_t nonces[RADIOLIB_LORAWAN_NONCES_BUF_SIZE];
  bool valid = prefs.getBytes("session", session, sizeof(session)) == sizeof(session) &&
               prefs.getBytes("nonces", nonces, sizeof(nonces)) == sizeof(nonces);
  if (valid && prefs.getBytes("multicast", &multicast, sizeof(multicast)) != sizeof(multicast)) {
    multicast = MulticastSession();
  }
  if (valid && prefs.getBytes("fragment", &fragmentation, sizeof(fragmentation)) != sizeof(fragmentation)) {
    fragmentation = FragmentSession();
  }
  if (valid) uplinkCounter = prefs.getUShort("ul_counter", 0);
  prefs.end();
  if (!valid) return false;

  return node.setBufferNonces(nonces) == RADIOLIB_ERR_NONE &&
         node.setBufferSession(session) == RADIOLIB_ERR_NONE;
}

void queueAck(uint8_t fport, const uint8_t *data, uint8_t length) {
  if (length > sizeof(ackOutbox.data)) return;
  ackOutbox.fport = fport;
  ackOutbox.length = length;
  memcpy(ackOutbox.data, data, length);
  ackOutbox.pending = true;
}

bool configureRadioLibMulticast() {
  if (!multicast.configured || !multicast.scheduled) return true;

  MulticastGroup_t group = {};
  group.cls = RADIOLIB_LORAWAN_CLASS_C;
  group.mcAddr = multicast.address;
  memcpy(group.mcAppSKey, multicast.appSKey, sizeof(group.mcAppSKey));
  memcpy(group.mcNwkSKey, multicast.nwkSKey, sizeof(group.mcNwkSKey));
  group.mcFCnt = multicast.minFrameCounter;
  group.mcFCntMax = multicast.maxFrameCounter;
  group.rxFCnt = 0;
  group.mcFreq = multicast.frequencyHz;
  group.mcDr = multicast.dataRate;

  int state = node.startMulticastSession(multicast.groupId, &group);
#ifdef DEBUG
  if (state != RADIOLIB_ERR_NONE) Serial.printf("[MCAST] RadioLib setup failed: %d\n", state);
#endif
  return state == RADIOLIB_ERR_NONE;
}

void processMcGroupSetup(const uint8_t *data, size_t length) {
  if (length != 29) return;

  uint8_t groupId = data[0] & 0x03;
  uint32_t address = (uint32_t)data[1] | ((uint32_t)data[2] << 8) |
                     ((uint32_t)data[3] << 16) | ((uint32_t)data[4] << 24);
  uint32_t minFrameCounter = (uint32_t)data[21] | ((uint32_t)data[22] << 8) |
                             ((uint32_t)data[23] << 16) | ((uint32_t)data[24] << 24);
  uint32_t maxFrameCounter = (uint32_t)data[25] | ((uint32_t)data[26] << 8) |
                             ((uint32_t)data[27] << 16) | ((uint32_t)data[28] << 24);

  uint8_t appSKey[16];
  uint8_t nwkSKey[16];
  bool supported = groupId < RADIOLIB_LORAWAN_MAX_NUM_MC_GROUPS &&
                   minFrameCounter < maxFrameCounter &&
                   deriveMulticastKeys(data + 5, address, appSKey, nwkSKey);
  uint8_t answer[2] = { 0x02, (uint8_t)(groupId & 0x03) };
  if (!supported) {
    answer[1] |= 0x04;
    queueAck(FPORT_MCAST_SETUP, answer, sizeof(answer));
    return;
  }

  multicast.configured = true;
  multicast.groupId = groupId;
  multicast.address = address;
  multicast.minFrameCounter = minFrameCounter;
  multicast.maxFrameCounter = maxFrameCounter;
  memcpy(multicast.appSKey, appSKey, sizeof(appSKey));
  memcpy(multicast.nwkSKey, nwkSKey, sizeof(nwkSKey));
  queueAck(FPORT_MCAST_SETUP, answer, sizeof(answer));
  saveSession();
}

bool eu868DataRateSupported(uint8_t dataRate) {
  return dataRate <= 6;
}

void processMcClassCSession(const uint8_t *data, size_t length) {
  if (length != 10) return;

  uint8_t groupId = data[0] & 0x03;
  uint32_t startGps = (uint32_t)data[1] | ((uint32_t)data[2] << 8) |
                      ((uint32_t)data[3] << 16) | ((uint32_t)data[4] << 24);
  uint8_t timeoutExponent = data[5] & 0x0F;
  uint32_t frequencyHz = ((uint32_t)data[6] | ((uint32_t)data[7] << 8) |
                          ((uint32_t)data[8] << 16)) * 100UL;
  uint8_t dataRate = data[9];
  bool validGroup = multicast.configured && groupId == multicast.groupId;
  bool validRadio = frequencyHz >= 863000000UL && frequencyHz <= 870000000UL &&
                    eu868DataRateSupported(dataRate);

  uint8_t answer[5] = { 0x04, (uint8_t)(groupId & 0x03), 0, 0, 0 };
  if (!validGroup || !validRadio) {
    answer[1] |= validGroup ? 0x0C : 0x10;
    queueAck(FPORT_MCAST_SETUP, answer, 2);
    return;
  }

  multicast.startGps = startGps;
  multicast.timeoutSeconds = 1UL << timeoutExponent;
  multicast.frequencyHz = frequencyHz;
  multicast.dataRate = dataRate;
  multicast.scheduled = true;

  if (!timeSynced) node.sendMacCommandReq(RADIOLIB_LORAWAN_MAC_DEVICE_TIME);
  uint32_t nowGps = currentGpsTime();
  uint32_t secondsToStart = (timeSynced && (int32_t)(startGps - nowGps) > 0)
                              ? startGps - nowGps
                              : 0;
  answer[1] = (uint8_t)(groupId & 0x03);
  answer[2] = (uint8_t)secondsToStart;
  answer[3] = (uint8_t)(secondsToStart >> 8);
  answer[4] = (uint8_t)(secondsToStart >> 16);
  queueAck(FPORT_MCAST_SETUP, answer, sizeof(answer));
  saveSession();
}

void processFragSessionSetup(const uint8_t *data, size_t length) {
  if (length != 10) return;

  uint8_t sessionIndex = (data[0] >> 6) & 0x03;
  uint16_t fragmentCount = (uint16_t)data[1] | ((uint16_t)data[2] << 8);
  uint8_t fragmentSize = data[3];
  uint8_t control = data[4];
  uint8_t padding = data[5];
  uint8_t algorithm = (control >> 3) & 0x07;

  uint8_t status = (uint8_t)(sessionIndex << 6);
  if (algorithm != 0) status |= 0x01;
  if (fragmentCount == 0 || fragmentSize == 0 || padding >= fragmentSize) status |= 0x08;

  if ((status & 0x0F) == 0) {
    fragmentation.active = true;
    fragmentation.index = sessionIndex;
    fragmentation.fragmentCount = fragmentCount;
    fragmentation.fragmentSize = fragmentSize;
    fragmentation.padding = padding;
    saveSession();
  }

  uint8_t answer[2] = { 0x02, status };
  queueAck(FPORT_FRAG, answer, sizeof(answer));
}

void processUnicastDownlink(uint8_t fport, const uint8_t *data, size_t length) {
  if (length == 0) return;

  if (fport == FPORT_MCAST_SETUP) {
    if (data[0] == 0x02) processMcGroupSetup(data + 1, length - 1);
    else if (data[0] == 0x04) processMcClassCSession(data + 1, length - 1);
  } else if (fport == FPORT_FRAG && data[0] == 0x02) {
    processFragSessionSetup(data + 1, length - 1);
  }
}

void countFragmentDownlink(int state, const LoRaWANEvent_t &event,
                           const uint8_t *data, size_t length) {
  if (state <= 0 || !event.multicast || event.mcGroupId != multicast.groupId ||
      event.fPort != FPORT_FRAG || !fragmentation.active || length < 4 ||
      data[0] != FRAG_DATA_BLOCK_CID) {
    return;
  }

  // Only counters survive this call; the decrypted fragment buffer is discarded.
  multicast.fragmentFrames++;
  multicast.fragmentBytes += length - 3;
}

void handleDownlink(int state, const LoRaWANEvent_t &event,
                    const uint8_t *data, size_t length) {
  if (state <= 0) return;
  if (event.multicast) {
    countFragmentDownlink(state, event, data, length);
  } else {
    processUnicastDownlink(event.fPort, data, length);
  }
}

void sendUplink() {
  uint8_t downlink[255];
  size_t downlinkLength = sizeof(downlink);
  LoRaWANEvent_t event = {};
  uint8_t telemetry[2] = { (uint8_t)(uplinkCounter >> 8), (uint8_t)uplinkCounter };
  uint8_t *payload = ackOutbox.pending ? ackOutbox.data : telemetry;
  uint8_t payloadLength = ackOutbox.pending ? ackOutbox.length : sizeof(telemetry);
  uint8_t fport = ackOutbox.pending ? ackOutbox.fport : FPORT_TELEMETRY;
  bool wasAck = ackOutbox.pending;
  bool sentClassCAnswer = wasAck && fport == FPORT_MCAST_SETUP && payload[0] == 0x04;

  int state = node.sendReceive(payload, payloadLength, fport, downlink, &downlinkLength,
                               false, nullptr, &event);
  if (!wasAck) uplinkCounter++;
  if (wasAck && state >= 0) ackOutbox.pending = false;
  handleDownlink(state, event, downlink, downlinkLength);
  updateDeviceTime();
  lastUplinkMs = millis();

  if (state >= 0 && sentClassCAnswer && !ackOutbox.pending && multicast.scheduled) {
    if (!configureRadioLibMulticast()) {
#ifdef DEBUG
      Serial.println("[MCAST] Failed to configure RadioLib multicast receive");
#endif
      multicast.scheduled = false;
    }
  }
  saveSession();
}

void pollClassC() {
  uint8_t downlink[255];
  size_t downlinkLength = sizeof(downlink);
  LoRaWANEvent_t event = {};
  int state = node.getDownlinkClassC(downlink, &downlinkLength, &event);
  handleDownlink(state, event, downlink, downlinkLength);
}

void runScheduledMulticast() {
  while (ackOutbox.pending && multicast.scheduled) {
    sendUplink();
    if (ackOutbox.pending) delay(50);
  }
  if (!multicast.scheduled) return;

  while (!timeSynced || (int32_t)(multicast.startGps - currentGpsTime()) > 0) {
    pollClassC();
    if (ackOutbox.pending || millis() - lastUplinkMs >= UPLINK_INTERVAL_MS) {
      if (!timeSynced) node.sendMacCommandReq(RADIOLIB_LORAWAN_MAC_DEVICE_TIME);
      sendUplink();
    }
    delay(10);
  }

  uint32_t startMs = millis();
  uint32_t durationMs = multicast.timeoutSeconds * 1000UL;
  uint32_t framesBefore = multicast.fragmentFrames;
  uint32_t bytesBefore = multicast.fragmentBytes;
  digitalWrite(ENERGY_MARKER_PIN, HIGH);

  while ((uint32_t)(millis() - startMs) < durationMs) {
    pollClassC();
    delay(2);
  }

  digitalWrite(ENERGY_MARKER_PIN, LOW);
  multicast.scheduled = false;
  saveSession();
#ifdef DEBUG
  Serial.printf("[FUOTA] Received %lu fragment frame(s), %lu fragment byte(s)\n",
                (unsigned long)(multicast.fragmentFrames - framesBefore),
                (unsigned long)(multicast.fragmentBytes - bytesBefore));
  Serial.println("[FUOTA] RX window complete; rebooting to restore ordinary Class-C operation");
#endif
  delay(50);
  ESP.restart();
}

void setup() {
  Serial.begin(115200);
  if (!multicastKeySelfTest()) haltOnError("[AES] Multicast key self-test failed", -1);
  pinMode(ENERGY_MARKER_PIN, OUTPUT);
  digitalWrite(ENERGY_MARKER_PIN, LOW);
  spi->begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_CS);

  int state = radio.begin(868.1, LORA_BW, LORA_SF, LORA_CR,
                          RADIOLIB_SX126X_SYNC_WORD_PRIVATE, LORA_TX_POWER, 8, 1.8, false);
  if (state != RADIOLIB_ERR_NONE) haltOnError("[Radio] begin failed", state);

  state = node.beginOTAA(joinEUI, devEUI, nwkKey, appKey);
  if (state != RADIOLIB_ERR_NONE) haltOnError("[LoRaWAN] beginOTAA failed", state);
  bool hadSession = loadSession();
  state = node.activateOTAA();
  if (hadSession && state != RADIOLIB_LORAWAN_SESSION_RESTORED) {
    haltOnError("[LoRaWAN] Session restore failed", state);
  }
  if (!hadSession && state != RADIOLIB_LORAWAN_NEW_SESSION) {
    haltOnError("[LoRaWAN] Join failed", state);
  }

  node.setADR(false);
  node.setCSMA(0, 0, false);
  node.setDutyCycle(true, 0);
  state = node.setClass(RADIOLIB_LORAWAN_CLASS_C);
  if (state < 0) haltOnError("[LoRaWAN] Class-C setup failed", state);
  if (multicast.scheduled && !configureRadioLibMulticast()) {
    haltOnError("[MCAST] Restoring multicast session failed", RADIOLIB_ERR_INVALID_MULTICAST_GROUP);
  }

  node.sendMacCommandReq(RADIOLIB_LORAWAN_MAC_DEVICE_TIME);
  sendUplink();
  if (multicast.scheduled) runScheduledMulticast();
}

void loop() {
  pollClassC();

  if (multicast.scheduled) {
    runScheduledMulticast();
    return;
  }

  if (ackOutbox.pending || millis() - lastUplinkMs >= UPLINK_INTERVAL_MS) {
    sendUplink();
  }
  delay(10);
}