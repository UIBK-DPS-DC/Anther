"""
ChirpStack downlink server
==========================================================
Matches the end-device protocol exactly:

  Downlink commands
  ── FPort 2 = CONTROL ──────────────────────────────────────────────────────────────
  0x10                       Switch to Class A
  0x11                       Switch to Class B
  0x12                       Switch to Class C
  0x20                       Set Interval

  ── FPort 1 = TELEMETRY / DATA ─────────────────────────────────────────────────────
  0x30 <chunk_id:u16> <len:u8> <data>     File chunk

  Everything else will be consumed, but nothing happens.

Usage
-----
  # Switch the device to Class A, B or C
  python server.py switch-class --class A

  # Send a file — DYNAMIC mode (default): re-decides A vs C after every
  # `--chunks-per-decision` chunks, using the actual SF/BW/power reported
  # by each chunk's TXACK event, and only sends a switch-class command
  # when the decision actually changes.
  python server.py send-file --file firmware.bin

  # Batch the decision: send 10 chunks per class re-evaluation instead of
  # re-deciding (and potentially switching) after every single chunk.
  python server.py send-file --file firmware.bin --chunks-per-decision 10

  # Send with confirmed downlinks (device ACKs each chunk)
  python server.py send-file --file firmware.bin --confirmed

  # Adjust chunk payload size (default 218 bytes of data; + 4-byte header = 222 bytes,
  # which matches the DR4/DR5 EU868 payload cap)
  python server.py send-file --file firmware.bin --chunk-data-size 218

  # Set the uplink interval (seconds) — useful to speed up Class A testing
  python server.py set-interval --seconds 5

  # Flush the downlink queue before a new test run
  python server.py flush-queue

  # Show pending queue depth
  python server.py queue-status

  # Calculate + switch to the energy-optimal class for the given file/params
  python server.py decide-class

  # --- MQTT / live radio-parameter sync -----------------------------------
  # One-shot: block until the device's NEXT uplink arrives, cache its SF/BW
  python server.py sync-radio --timeout 30

  # Continuous: keep listening and refresh the cache on every uplink
  # (run this in its own terminal while you test)
  python server.py mqtt-listen

  # send-file / decide-class automatically use the cached SF/BW if fresh;
  # force a fresh read right before sending with --wait-uplink
  python server.py send-file --file firmware.bin --wait-uplink

  # --- Power parameters (P_TX/P_RX/P_sleep, mW) ----------------------------
  # These can't be auto-detected (see README/thread) — set them once from a
  # datasheet or a real current-draw measurement:
  python server.py set-power --p-tx 3200 --p-rx 45 --p-sleep 0.001
  python server.py show-power

  # Or override just for a single run without touching the saved config:
  python server.py send-file --file firmware.bin --p-tx 2800 --p-rx 40
"""

# This code was written with Claude assistance. 

import argparse
import json
import math
import os
import struct
import sys
import threading
import time
import zlib
from dataclasses import asdict
import requests

from grpc import insecure_channel

from chirpstack_api import api
from estimation_model import DATA_RATES, EnergyModel, EnergyParameters
from evaluation_custom_adr import AdrOverrideSchedule, trigger_adr_override as evaluation_trigger_adr_override

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None

# ── Config ────────────────────────────────────────────────────────────────────
SERVER    = "localhost:8080"
API_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
    ".eyJhdWQiOiJjaGlycHN0YWNrIiwiaXNzIjoiY2hpcnBzdGFjayIsInN1YiI6ImFiYmU0"
    "OWQ4LTkwYmMtNDQwYi04YWIwLTcxOTQxNzg2MjEzOCIsInR5cCI6ImtleSJ9"
    ".myWd6FV4I1lymPc6gxnJ1YkwsKynW5k0ujY1ySM2G0g"
)
DEV_EUI = "e21ca312cb69528a"

# ChirpStack ships with an internal Mosquitto broker by default — plain
# localhost:1883, no auth, unless you've changed chirpstack.toml. Override
# via --mqtt-host / --mqtt-port / --mqtt-username / --mqtt-password if your
# deployment uses an external broker with credentials.
MQTT_HOST     = "localhost"
MQTT_PORT     = 1883
MQTT_USERNAME = None
MQTT_PASSWORD = None

# Local cache so SF/BW/power detected via MQTT survives across separate CLI
# runs (each `python server.py ...` invocation is a fresh process).
CACHE_FILE_PREFIX = "configuration_cache"
CACHE_FILE = CACHE_FILE_PREFIX + "_" + DEV_EUI + ".json"

# Separate small state file tracking which class we last told the device to
# use. Kept apart from CACHE_FILE/EnergyParameters on purpose — it's session
# state about the device, not a modeling parameter — so the dynamic FUOTA
# loop (and the plain `switch-class` command) can skip a redundant
# switch-class downlink when the device is already believed to be in the
# target class, even across separate CLI invocations.
STATE_FILE_PREFIX = "device_state"
STATE_FILE = STATE_FILE_PREFIX + "_" + DEV_EUI + ".json"

# ChirpStack's REST gateway (separate from the gRPC API on SERVER/8080).
# Used only for the ADR-override evaluation hook below, since device
# `variables` aren't exposed on the gRPC DeviceService used elsewhere in
# this file — the REST gateway is the straightforward way to patch them.
REST_SERVER = "http://localhost:8090"

# Power-draw parameters (P_TX/P_RX/P_sleep, all in mW) cannot be auto-detected
# — they depend on your specific board/radio and are best obtained from a
# datasheet or, better, a real current-draw measurement. Rather than editing
# them in source, they live here and can be set once with `set-power` (which
# persists them to CACHE_FILE) or overridden ad-hoc per run with
# --p-tx/--p-rx/--p-sleep on send-file / decide-class.
POWER_FIELDS = ("p_tx", "p_rx", "p_sleep")

# ── FPorts – must match end-device constants ───────────────────────────────────
FPORT_TELEMETRY = 1   # data-plane downlinks (file transfer, interval, counter)
FPORT_CONTROL   = 2   # control-plane downlinks (class switch)

# ── Commands – must match end-device processDataDownlink / processControlDownlink
CMD_SWITCH_CLASS_A   = 0x10
CMD_SWITCH_CLASS_B   = 0x11
CMD_SWITCH_CLASS_C   = 0x12
CMD_SET_INTERVAL     = 0x20
CMD_DOWNLOAD_CHUNK   = 0x30

# ── Payload sizes ─────────────────────────────────────────────────────────────
# Chunk command overhead:  1 (cmd) + 2 (chunk_id) + 1 (chunk_len) =  4 B
# Default chunk data size matches the DR4/DR5 EU868 payload cap (222 B) minus
# that 4 B header.
CHUNK_DATA_SIZE = 46

parameters = EnergyParameters(
    remaining_bytes=1_024,
    uplink_payload_bytes=5,
    report_interval=60.0,
    ul_dr=5,
    class_c_dr=0, # Fixed RX2 Downlink DR for EU868
    duty_cycle=0.01,
    dl_duty_cycle=0.10,
    lambda_=0.65,
)


# ChirpStack client wrapper
class ChirpStackClient:
    def __init__(self, server: str, token: str, dev_eui: str):
        self._channel  = insecure_channel(server)
        self._stub     = api.DeviceServiceStub(self._channel)
        self._meta     = [("authorization", f"Bearer {token}")]
        self._dev_eui  = dev_eui

    # ── Low-level enqueue ─────────────────────────────────────────────────────
    def enqueue(self, payload: bytes, fport: int, confirmed: bool = False) -> str:
        req = api.EnqueueDeviceQueueItemRequest()
        req.queue_item.dev_eui   = self._dev_eui
        req.queue_item.f_port    = fport
        req.queue_item.data      = payload
        req.queue_item.confirmed = confirmed
        resp = self._stub.Enqueue(req, metadata=self._meta)
        return resp.id

    # ── Queue management ──────────────────────────────────────────────────────
    def queue_depth(self) -> int:
        req = api.GetDeviceQueueItemsRequest()
        req.dev_eui = self._dev_eui
        resp = self._stub.GetQueue(req, metadata=self._meta)
        return len(resp.result)

    def wait_for_empty_queue(
        self,
        timeout: float = 300.0,
        poll_interval: float = 0.25,
    ) -> None:
        """Wait until ChirpStack reports no pending downlinks for the device."""
        deadline = time.monotonic() + timeout
        while True:
            depth = self.queue_depth()
            if depth == 0:
                return

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for the device queue to empty; "
                    f"{depth} item(s) remain"
                )

            print(f"[QUEUE] Waiting for queue to empty ({depth} item(s) remain)")
            time.sleep(min(poll_interval, remaining))

    def flush_queue(self):
        req = api.FlushDeviceQueueRequest()
        req.dev_eui = self._dev_eui
        self._stub.FlushQueue(req, metadata=self._meta)


# MQTT plumbing shared by uplink sync and TXACK monitoring
def _require_mqtt():
    if mqtt is None:
        print("[MQTT] paho-mqtt is not installed. Run: pip install paho-mqtt")
        sys.exit(1)


def _make_mqtt_client() -> "mqtt.Client":
    """Construct an mqtt.Client compatible with both paho-mqtt 1.x and 2.x."""
    _require_mqtt()
    if hasattr(mqtt, "CallbackAPIVersion"):
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    return mqtt.Client()


def _parse_uplink_payload(raw: bytes) -> dict | None:
    """Extract SF/BW/CR/DR/RSSI/SNR from a ChirpStack `event/up` MQTT message."""
    try:
        evt = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    tx_info = evt.get("txInfo") or {}
    lora = (tx_info.get("modulation") or {}).get("lora")
    if not lora:
        return None

    rx_info = evt.get("rxInfo") or []
    rssi = rx_info[0].get("rssi") if rx_info else None
    snr  = rx_info[0].get("snr")  if rx_info else None

    device_info = evt.get("deviceInfo") or {}
    device_class = device_info.get("deviceClassEnabled")
    if isinstance(device_class, str):
        device_class = device_class.upper().replace("CLASS_", "")

    return {
        "sf":        lora.get("spreadingFactor"),
        "bw_hz":     lora.get("bandwidth"),
        "code_rate": lora.get("codeRate", "CR_4_5"),
        "dr":        evt.get("dr"),
        "fcnt":      evt.get("fCnt"),
        "rssi":      rssi,
        "snr":       snr,
        "device_time": evt.get("time"),
        "device_class": device_class if device_class in {"A", "B", "C"} else None,
    }


def _parse_txack_payload(raw: bytes) -> dict | None:
    """
    Extract the fields we care about from a ChirpStack v4 `event/txack`
    MQTT message.

    Schema (from ChirpStack's integration.proto / event-types docs):
        {
          "downlinkId": ...,
          "time": "...",
          "deviceInfo": {...},
          "queueItemId": "<uuid — matches the id Enqueue() returned>",
          "fCntDown": ...,
          "gatewayId": "...",
          "txInfo": {
            "frequency": ...,
            "power": ...,
            "modulation": {"lora": {"bandwidth": ..., "spreadingFactor": ..., "codeRate": "..."}},
            "timing": {...}
          }
        }

    Note this is the *network-server-sent* confirmation ("a queued
    downlink was transmitted to a gateway"), not a device-level ACK —
    there's no guarantee of receipt. For the SF/BW actually used over
    the air, though, it's the right (and only) event to look at.
    """
    try:
        evt = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    tx_info = evt.get("txInfo") or {}
    lora = (tx_info.get("modulation") or {}).get("lora")
    if not lora:
        return None

    return {
        "queue_item_id": evt.get("queueItemId"),
        "downlink_id":   evt.get("downlinkId"),
        "f_cnt_down":    evt.get("fCntDown"),
        "gateway_id":    evt.get("gatewayId"),
        "sf":            lora.get("spreadingFactor"),
        "bw_hz":         lora.get("bandwidth"),
        "code_rate":     lora.get("codeRate", "CR_4_5"),
        "power":         tx_info.get("power"),
        "frequency":     tx_info.get("frequency"),
        "time":          evt.get("time"),
    }


class TxAckMonitor:
    """
    Background MQTT listener for `event/txack` messages, subscribed once
    and kept alive across many enqueue/wait cycles (reconnecting per chunk
    would add MQTT round-trip latency to every single fragment).

    Usage:
        mon = TxAckMonitor(dev_eui)
        mon.start()
        ...
        item_id = client.enqueue(payload, fport)
        txack = mon.wait_for(item_id, timeout=30)
        ...
        mon.stop()

    Events are matched to enqueue calls via `queueItemId`, which
    ChirpStack's txack event carries and which equals the `id` returned
    by the gRPC Enqueue() call.
    """

    def __init__(self, dev_eui, host=MQTT_HOST, port=MQTT_PORT, username=None, password=None):
        _require_mqtt()
        self._dev_eui  = dev_eui
        self._host     = host
        self._port     = port
        self._username = username
        self._password = password
        self._client    = None
        self._lock      = threading.Lock()
        self._results    = {}   # queue_item_id -> parsed event dict (arrived, unclaimed)
        self._waiters    = {}   # queue_item_id -> threading.Event (someone is waiting)
        self._connected  = threading.Event()

    def start(self):
        self._client = _make_mqtt_client()
        if self._username:
            self._client.username_pw_set(self._username, self._password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        try:
            self._client.connect(self._host, self._port, keepalive=30)
        except (OSError, ConnectionRefusedError) as e:
            raise RuntimeError(f"[TXACK] Could not connect to MQTT broker at {self._host}:{self._port} — {e}")
        self._client.loop_start()
        if not self._connected.wait(10):
            raise RuntimeError("[TXACK] Timed out connecting/subscribing to MQTT broker")

    def stop(self):
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc not in (0, getattr(mqtt, "MQTT_ERR_SUCCESS", 0)):
            print(f"[TXACK] Connect failed, rc={rc}")
            return
        topic = f"application/+/device/{self._dev_eui}/event/txack"
        client.subscribe(topic)
        print(f"[TXACK] Connected, subscribed to {topic}")
        self._connected.set()

    def _on_message(self, client, userdata, msg):
        parsed = _parse_txack_payload(msg.payload)
        if parsed is None:
            return
        qid = parsed.get("queue_item_id")
        if not qid:
            return
        waiter = None
        with self._lock:
            self._results[qid] = parsed
            waiter = self._waiters.get(qid)
        if waiter is not None:
            waiter.set()

    def wait_for(self, queue_item_id: str, timeout: float = 30.0) -> dict | None:
        """Block until a TXACK for `queue_item_id` arrives, or timeout. Returns
        the parsed event dict, or None on timeout."""
        with self._lock:
            if queue_item_id in self._results:
                return self._results.pop(queue_item_id)
            ev = threading.Event()
            self._waiters[queue_item_id] = ev

        arrived = ev.wait(timeout)

        with self._lock:
            self._waiters.pop(queue_item_id, None)
            if arrived:
                return self._results.pop(queue_item_id, None)
        return None


class UplinkMonitor:
    """Keep the latest uplink radio settings and device class available."""

    def __init__(self, dev_eui: str):
        _require_mqtt()
        self._dev_eui = dev_eui
        self._client = None
        self._connected = threading.Event()
        self._uplink = threading.Condition()
        self._latest: dict | None = None
        self._generation = 0
        self._consumed_generation = 0

    def start(self) -> None:
        self._client = _make_mqtt_client()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
        self._client.loop_start()
        if not self._connected.wait(10):
            raise RuntimeError("[MQTT] Timed out connecting to uplink topic")

    def stop(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc != 0:
            print(f"[MQTT] Uplink connect failed, rc={rc}")
            return
        client.subscribe(f"application/+/device/{self._dev_eui}/event/up")
        self._connected.set()

    def _on_message(self, client, userdata, msg):
        parsed = _parse_uplink_payload(msg.payload)
        if parsed is None:
            return
        # Cache disabled: do not persist or reapply stale radio parameters.
        with self._uplink:
            self._latest = parsed
            self._generation += 1
            self._uplink.notify_all()

    def wait_for_next(self, timeout: float) -> dict | None:
        with self._uplink:
            if not self._uplink.wait_for(
                lambda: self._generation > self._consumed_generation,
                timeout,
            ):
                return None
            self._consumed_generation = self._generation
            return dict(self._latest)

    def discard_pending(self) -> None:
        """Discard uplinks received before the next class-A receive window."""
        with self._uplink:
            self._consumed_generation = self._generation
            self._latest = None


def read_parameters() -> dict | None:
    """Read cached EnergyParameters from CACHE_FILE, if present."""
    if not os.path.exists(CACHE_FILE):
        return None

    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    return data


def save_parameters(params: EnergyParameters) -> None:
    """Persist the complete EnergyParameters object."""
    data = asdict(params)

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_radio_cache(params: dict) -> None:
    data = read_parameters() or asdict(parameters)
    data.update(params)

    for k, v in data.items():
        if hasattr(parameters, k):
            setattr(parameters, k, v)

    # Cache disabled: power settings are not persisted locally.


def apply_uplink_radio_params(uplink: dict) -> int | None:
    """Make the queued downlink match the device's latest uplink state."""
    uplink_dr = uplink.get("dr")
    device_class = uplink.get("device_class")
    print(f"[Uplink] [{time.localtime().tm_hour}:{time.localtime().tm_min}:{time.localtime().tm_sec}] Applying uplink DR{uplink_dr} and device class {device_class}")
    if device_class == "C":
        downlink_dr = parameters.class_c_dr
    elif isinstance(uplink_dr, int) and uplink_dr in DATA_RATES:
        downlink_dr = uplink_dr
    else:
        return None

    if device_class != "C" and isinstance(uplink_dr, int) and uplink_dr in DATA_RATES:
        parameters.ul_dr = uplink_dr
    if device_class == "C":
        parameters.class_c_dr = downlink_dr
    # Cache disabled: device class is not persisted locally.
    # Cache disabled: keep the fresh uplink parameters in memory only.
    return downlink_dr


def load_radio_cache() -> dict | None:
    return read_parameters()


def apply_cached_radio_params() -> None:
    cached = read_parameters()
    if not cached:
        print("[RADIO] No cached configuration file yet — using in-memory defaults")
        return

    for k, v in cached.items():
        if hasattr(parameters, k):
            setattr(parameters, k, v)

    print(
        f"[RADIO] Using cached Uplink DR{parameters.ul_dr}"
    )


def load_power_config() -> dict:
    cached = read_parameters()
    if not cached:
        return {}
    return {k: cached[k] for k in POWER_FIELDS if k in cached}


def save_power_config(
    p_tx: float | None = None,
    p_rx: float | None = None,
    p_sleep: float | None = None,
) -> dict:
    if p_tx is not None:
        parameters.p_tx = p_tx
    if p_rx is not None:
        parameters.p_rx = p_rx
    if p_sleep is not None:
        parameters.p_sleep = p_sleep

    # Cache disabled: power settings are applied in memory only.

    return {
        "p_tx": parameters.p_tx,
        "p_rx": parameters.p_rx,
        "p_sleep": parameters.p_sleep,
    }


# Device-state tracking (which class we last told the device to use)
def load_device_state() -> str | None:
    """Return the last class we believe the device is in ('A'/'B'/'C'), or
    None if unknown (no state file yet, or it couldn't be read)."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("current_class")


def save_device_state(current_class: str) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump({"current_class": current_class, "updated": time.time()}, f, indent=2)


def mqtt_wait_for_uplink(
    dev_eui: str,
    timeout: float = 30.0,
    host: str = MQTT_HOST,
    port: int = MQTT_PORT,
    username: str | None = MQTT_USERNAME,
    password: str | None = MQTT_PASSWORD,
) -> dict | None:
    """
    Connect to the ChirpStack MQTT broker and block until the device's next
    uplink event arrives (or timeout). Returns the parsed radio-param dict
    (and caches it to disk), or None on timeout/failure.
    """
    _require_mqtt()
    result: dict = {}
    got_message = threading.Event()
    topic = f"application/+/device/{dev_eui}/event/up"

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc not in (0, getattr(mqtt, "MQTT_ERR_SUCCESS", 0)):
            print(f"[MQTT] Connect failed, rc={rc}")
            return
        client.subscribe(topic)
        print(f"[MQTT] Connected, subscribed to {topic}")

    def on_message(client, userdata, msg):
        parsed = _parse_uplink_payload(msg.payload)
        if parsed:
            result.update(parsed)
            got_message.set()

    client = _make_mqtt_client()
    if username:
        client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(host, port, keepalive=30)
    except (OSError, ConnectionRefusedError) as e:
        print(f"[MQTT] Could not connect to {host}:{port} — {e}")
        return None

    client.loop_start()
    print(f"[MQTT] Waiting up to {timeout:.0f}s for next uplink from {dev_eui} ...")
    arrived = got_message.wait(timeout)
    client.loop_stop()
    client.disconnect()

    if not arrived:
        print("[MQTT] Timed out waiting for an uplink — is the device transmitting?")
        return None

    print(f"[MQTT] SF{result['sf']} / BW{result['bw_hz']} Hz  "
          f"DR{result['dr']}  RSSI={result['rssi']}  SNR={result['snr']}  "
          f"(fCnt={result['fcnt']})")
    # Cache disabled: radio parameters are not persisted locally.
    return result


def cmd_mqtt_listen(
    dev_eui: str,
    host: str = MQTT_HOST,
    port: int = MQTT_PORT,
    username: str | None = MQTT_USERNAME,
    password: str | None = MQTT_PASSWORD,
) -> None:
    """Continuously listen for uplinks and refresh the on-disk cache on each
    one. Meant to run in its own terminal alongside your test commands."""
    _require_mqtt()
    topic = f"application/+/device/{dev_eui}/event/up"

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc != 0:
            print(f"[MQTT] Connect failed, rc={rc}")
            return
        client.subscribe(topic)
        print(f"[MQTT] Listening on {topic}  (Ctrl+C to stop)")

    def on_message(client, userdata, msg):
        parsed = _parse_uplink_payload(msg.payload)
        if not parsed:
            return
        # Cache disabled: radio parameters are not persisted locally.
        print(f"[MQTT] fCnt={parsed['fcnt']:<6} SF{parsed['sf']} / BW{parsed['bw_hz']} Hz  "
              f"DR{parsed['dr']}  RSSI={parsed['rssi']}  SNR={parsed['snr']}  -> cached")

    client = _make_mqtt_client()
    if username:
        client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(host, port, keepalive=30)
    except (OSError, ConnectionRefusedError) as e:
        print(f"[MQTT] Could not connect to {host}:{port} — {e}")
        return

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[MQTT] Stopped")
        client.disconnect()


# Payload builders  (big-endian, matching PayloadBuilder on the device)
def build_switch_class(target: str) -> bytes:
    target = target.upper()
    if target == "A":
        cmd = CMD_SWITCH_CLASS_A
    elif target == "B":
        cmd = CMD_SWITCH_CLASS_B
    elif target == "C":
        cmd = CMD_SWITCH_CLASS_C
    else:
        raise ValueError(f"Unknown class target: {target!r}")
    return bytes([cmd])


def build_set_interval(seconds: int) -> bytes:
    # 0x20 <uint8 seconds>
    return struct.pack(">BB", CMD_SET_INTERVAL, seconds)


def build_download_chunk(chunk_id: int, data: bytes) -> bytes:
    # 0x30 <uint16 BE: chunk_id> <uint8: length> <data>
    return data


# High-level commands — control plane
def cmd_switch_class(client: ChirpStackClient, target: str, persist_state: bool = True):
    """Send a class-switch command on FPORT_CONTROL (fire-and-forget; use
    switch_class_and_wait_txack() instead if you need TXACK confirmation)."""
    target = target.upper()
    payload = build_switch_class(target)
    item_id = client.enqueue(payload, FPORT_CONTROL, confirmed=False)
    print(f"[CTRL] Switch to Class {target} enqueued  (id={item_id})")
    print(f"       Class A: device switches after its next uplink opens RX1/RX2")
    print(f"       Class B: device switches after its next uplink opens RX1/RX2")
    print(f"       Class C: device switches immediately once the downlink arrives")
    # Cache disabled: device class is not persisted locally.


def cmd_set_interval(client: ChirpStackClient, seconds: int):
    """Tell the device to change its uplink interval."""
    parameters.report_interval = seconds
    # Cache disabled: interval changes are not persisted locally.
    payload = build_set_interval(seconds)
    item_id = client.enqueue(payload, FPORT_CONTROL, confirmed=False)
    print(f"[DATA] Set interval → {seconds} s  (id={item_id})")


def cmd_flush_queue(client: ChirpStackClient):
    client.flush_queue()
    print("[QUEUE] Flushed")


def cmd_queue_status(client: ChirpStackClient):
    depth = client.queue_depth()
    print(f"[QUEUE] {depth} item(s) pending")


# TXACK-confirmed helpers (used by the dynamic FUOTA loop)
def send_one_chunk_and_wait_txack(
    client: ChirpStackClient,
    monitor: TxAckMonitor,
    chunk_id: int,
    chunk: bytes,
    confirmed: bool,
    timeout: float = 3000.0,
) -> dict | None:
    payload = build_download_chunk(chunk_id, chunk)
    item_id = client.enqueue(payload, FPORT_TELEMETRY, confirmed=confirmed)
    print(f"[FILE] Chunk {chunk_id + 1:>4}  {len(chunk):>3} B  enqueued (id={item_id}) — waiting for TXACK ...")
    txack = monitor.wait_for(item_id, timeout=timeout)
    if txack is None:
        print(f"[FILE] No TXACK for chunk {chunk_id + 1} within {timeout:.0f}s")
        return None
    print(f"[FILE] TXACK chunk {chunk_id + 1}: SF{txack['sf']} / BW{txack['bw_hz']} Hz "
          f"/ {txack['power']} dBm / gw={txack['gateway_id']}")
    return txack


def switch_class_and_wait_txack(
    client: ChirpStackClient,
    monitor: TxAckMonitor,
    target_class: str,
    timeout: float = 30.0,
) -> dict | None:
    target_class = target_class.upper()
    payload = build_switch_class(target_class)
    item_id = client.enqueue(payload, FPORT_CONTROL, confirmed=False)
    print(f"[CTRL] Switch to Class {target_class} enqueued (id={item_id}) — waiting for TXACK ...")
    txack = monitor.wait_for(item_id, timeout=timeout)
    if txack is None:
        print(f"[CTRL] No TXACK for class-switch within {timeout:.0f}s")
        return None
    print(f"[CTRL] Switch-to-{target_class} TXACK received "
          f"(SF{txack['sf']} / BW{txack['bw_hz']} Hz / gw={txack['gateway_id']})")
    return txack

# ── REST helpers — evaluation-only hook for the custom ADR algorithm ───────────
# The gRPC DeviceService used above doesn't expose device `variables`, so this
# talks to ChirpStack's REST gateway instead (same one your
# `curl -X PUT http://localhost:8090/api/devices/...` example hits). We GET
# the device first and only patch the one variable, rather than blindly
# PUTting a hand-built body, so we never accidentally clobber applicationId /
# deviceProfileId / other variables you have set.
def get_device_via_rest(dev_eui: str, rest_server: str, token: str, timeout: float = 10.0) -> dict:
    url = f"{rest_server.rstrip('/')}/api/devices/{dev_eui}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
 
 
def set_device_variable_via_rest(
    dev_eui: str,
    key: str,
    value: str,
    rest_server: str = REST_SERVER,
    token: str = API_TOKEN,
    timeout: float = 10.0,
) -> dict:
    """
    GET the device, set `variables[key] = value`, PUT it back. Returns the
    device object as sent (not re-fetched — ChirpStack's PUT doesn't echo
    the stored object back in a form worth re-parsing here).
    """
    current = get_device_via_rest(dev_eui, rest_server, token, timeout=timeout)
    device = current.get("device")
    if device is None:
        raise RuntimeError(
            f"[ADR-OVERRIDE] Unexpected response shape from GET {rest_server}/api/devices/{dev_eui}: "
            f"no 'device' key. Got keys: {list(current.keys())}"
        )
 
    variables = dict(device.get("variables") or {})
    variables[key] = str(value)
    device["variables"] = variables
 
    url = f"{rest_server.rstrip('/')}/api/devices/{dev_eui}"
    resp = requests.put(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=json.dumps({"device": device}),
        timeout=timeout,
    )
    resp.raise_for_status()
    return device

def trigger_adr_override(dev_eui: str, value: str, rest_server: str = REST_SERVER, token: str = API_TOKEN) -> bool:
    """Best-effort: set the ADR_OVERRIDE_VARIABLE device variable. Returns
    True on success, False on failure (never raises — a failed override
    trigger shouldn't abort an in-progress file transfer)."""
    return evaluation_trigger_adr_override(dev_eui, int(value), rest_server, token)

def cmd_set_datarate(dev_eui: str = DEV_EUI,
                     datarate: int = 5,
                     rest_server: str = REST_SERVER,
                     rest_token: str = API_TOKEN):
    trigger_adr_override(dev_eui, datarate, rest_server, rest_token)


def switch_class(client, txack_monitor, dev_eui, target_class, timeout=30.0,
                 uplink_monitor=None) -> bool:
    # client.flush_device_queue(dev_eui)  # avoid collision with anything already queued
    target_class = target_class.upper()
    item_id = client.enqueue(build_switch_class(target_class), FPORT_CONTROL, confirmed=False)
    txack = txack_monitor.wait_for(item_id, timeout=timeout)
    if txack is None:
        print(f"[CTRL] No TXACK for switch to {target_class}; not trusting the change")
        return False
    print(f"[CTRL] Switched to Class {target_class}")
    if target_class == "A" and uplink_monitor is not None:
        uplink_monitor.discard_pending()
    return True


def apply_uplink_radio_params(uplink: dict) -> int | None:
    dr = uplink.get("dr")
    if isinstance(dr, int) and dr in DATA_RATES:
        parameters.ul_dr = dr
        return dr
    return None

def send_class_a_chunk(
    client,
    uplink_monitor,
    dev_eui,
    chunk_id,
    file_bytes,
    last_fcnt,
    pending_adr_target_dr,
    confirmed,
    timeout,
    max_retries=5,
):
    """Wait for a fresh uplink and enqueue the Class-A downlink."""
    print("Sending Class A downlink")

    t0 = time.perf_counter()

    prepared_payloads = {}
    for dr in DATA_RATES:
        chunk_data_size = DATA_RATES[dr][2] - 2
        chunk = file_bytes[:chunk_data_size]

        prepared_payloads[dr] = build_download_chunk(
            chunk_id,
            chunk,
        )

    t0 = time.perf_counter()
    retries = 0
    while True:
        uplink = uplink_monitor.wait_for_next(timeout)
        t1 = time.perf_counter()
        print("Uplink:", dict(uplink) if uplink else None)
        if uplink is None:
            return None
        fcnt = uplink.get("fcnt")
        if fcnt == last_fcnt:
            retries += 1
            if retries >= max_retries:
                return None
            continue
        dr = uplink.get("dr")
        if not isinstance(dr, int) or dr not in DATA_RATES:
            return None
        break

    t2 = time.perf_counter()

    effective_dr = dr if pending_adr_target_dr is None else min(
        dr, pending_adr_target_dr
    )

    item_id = client.enqueue(
        prepared_payloads[effective_dr],
        FPORT_TELEMETRY,
        confirmed=confirmed,
    )

    t3 = time.perf_counter()

    current_time_in_sec = time.time()
    print(f"Hour: {int(current_time_in_sec / 3600) % 24}, minute: {int(current_time_in_sec % 3600 / 60)}, second: {int(current_time_in_sec % 60)}")
    print(f"Sent Class A downlink — waiting for TXACK: {item_id}")

    print("Timing summary:")
    print(f"waiting for uplink: {(t1-t0)*1000:.2f} ms")
    print(f"from uplink to enqueue: {(t2-t1)*1000:.2f} ms")
    print(f"enqueue: {(t3-t2)*1000:.2f} ms")
    print(f"From before uplink to after enqueue: {(t3-t0)*1000:.2f} ms")

    return {
        "fcnt": fcnt,
        "dr": dr,
        "effective_dr": effective_dr,
        "chunk": file_bytes[:DATA_RATES[dr][2] - 2],
        "item_id": item_id,
    }


def send_class_c_chunk(client, chunk_id, file_bytes, class_c_dr, confirmed):
    print("Sending Class C downlink")
    chunk_data_size = DATA_RATES[class_c_dr][2] - 2
    chunk = file_bytes[:chunk_data_size]
    item_id = client.enqueue(build_download_chunk(chunk_id, chunk), FPORT_TELEMETRY, confirmed=confirmed)
    current_time_in_sec = time.time()
    print(f"Hour: {int(current_time_in_sec / 3600) % 24}, minute: {int(current_time_in_sec % 3600 / 60)}, second: {int(current_time_in_sec % 60)}")
    print(f"Sent Class C downlink — waiting for TXACK: {item_id}")
    return {"fcnt": None, "dr": class_c_dr, "effective_dr": class_c_dr, "chunk": chunk, "item_id": item_id}

def cmd_send_file(
    client: ChirpStackClient,
    file_bytes: bytes,
    l: float = 0.65,
    confirmed: bool = False,
    enqueue_delay: float = 0.0,
    dev_eui: str = DEV_EUI,
    mqtt_timeout: float = 300.0,
    p_tx: float | None = None,
    p_rx: float | None = None,
    p_sleep: float | None = None,
    optimize_interval: bool = False,
    adr_override: bool = False,
    rest_server: str = REST_SERVER,
    rest_token: str = API_TOKEN,
    class_mode: str = "dynamic",
    class_c_dr: int = 0,
):
    """Dynamic FUOTA sender. Class A sizes chunks off the DR of a fresh
    uplink; Class C uses the fixed class_c_dr, trusted immediately once a
    switch's TXACK confirms the device received it."""
    global parameters
    parameters.lambda_ = max(0.0, min(l, 1.0))
    parameters.class_c_dr = class_c_dr
    parameters.ul_dr = 5
    parameters.dl_dr_a = 5
    if p_tx is not None:
        parameters.p_tx = p_tx
    if p_rx is not None:
        parameters.p_rx = p_rx
    if p_sleep is not None:
        parameters.p_sleep = p_sleep

    uplink_monitor = UplinkMonitor(dev_eui)
    uplink_monitor.start()
    txack_monitor = TxAckMonitor(
        dev_eui=dev_eui, host=MQTT_HOST, port=MQTT_PORT,
        username=MQTT_USERNAME, password=MQTT_PASSWORD,
    )
    txack_monitor.start()

    pending_adr_target_dr: int | None = None

    def _trigger_adr(dr: int) -> None:
        nonlocal pending_adr_target_dr
        pending_adr_target_dr = dr
        trigger_adr_override(dev_eui, dr, rest_server, rest_token)

    adr_schedule = AdrOverrideSchedule(_trigger_adr) if adr_override else None

    file_size = len(file_bytes)
    crc32 = zlib.crc32(file_bytes) & 0xFFFFFFFF
    parameters.remaining_bytes = file_size
    print(f"[FILE] {file_size}B  CRC32=0x{crc32:08X}  lambda={parameters.lambda_:.3f}  confirmed={confirmed}")
    start = time.time()

    try:
        uplink = uplink_monitor.wait_for_next(mqtt_timeout)
        if uplink is None or apply_uplink_radio_params(uplink) is None:
            print("[FILE] No usable initial uplink; aborting")
            return
        print(dict(uplink))
        last_used_fcnt = uplink.get("fcnt")

        if class_mode == "dynamic":
            best_class = cmd_decide_class(
                client, remaining_bytes=file_size,
                p_tx=p_tx, p_rx=p_rx, p_sleep=p_sleep,
                l=l, optimize_interval=optimize_interval,
            )
            if best_class is None:
                print("[FILE] Could not determine initial class; aborting")
                return
        else:
            best_class = class_mode

        if not switch_class(
            client, txack_monitor, dev_eui, best_class, mqtt_timeout,
            uplink_monitor=uplink_monitor,
        ):
            print("[FILE] Initial class switch not confirmed; aborting")
            return
        current_class = best_class

        switch_from_class_C_to_A = 0 # This is only needed, because there is an additional delay when switching from class C to class A.

        i = 0
        while len(file_bytes) > 0:
            client.wait_for_empty_queue()
            if current_class == "C":
                result = send_class_c_chunk(client, i, file_bytes, class_c_dr, confirmed)
            elif switch_from_class_C_to_A > 0:
                result = send_class_a_chunk(
                    client, uplink_monitor, dev_eui, i, file_bytes,
                    last_used_fcnt, 2, confirmed, mqtt_timeout,
                )
                last_used_fcnt = result["fcnt"]
            else:
                result = send_class_a_chunk(
                    client, uplink_monitor, dev_eui, i, file_bytes,
                    last_used_fcnt, pending_adr_target_dr, confirmed, mqtt_timeout,
                )
                last_used_fcnt = result["fcnt"]
            switch_from_class_C_to_A = max(0, switch_from_class_C_to_A - 1)
            if result is None:
                print("[FILE] Could not send next chunk; stopping")
                return

            # if current_class == "A":
            #     if pending_adr_target_dr is not None:
            #         if result["dr"] <= pending_adr_target_dr:
            #             pending_adr_target_dr = None
            # parameters.ul_dr = result["dr"]
            # parameters.dl_dr = result["dr"]

            txack = txack_monitor.wait_for(result["item_id"], timeout=mqtt_timeout)
            if txack is None:
                print(f"[FILE] No TXACK for chunk {i+1}; stopping")
                return

            file_bytes = file_bytes[len(result["chunk"]):]
            print(f"[FILE] Chunk {i+1}  {len(result['chunk'])}B  class={current_class}  "
                  f"DR{result['effective_dr']}  {len(file_bytes)}B remaining")

            if adr_schedule is not None:
                target_dr = adr_schedule.update()
                if target_dr is not None:
                    pending_adr_target_dr = target_dr
                    # parameters.ul_dr = last_dr
                    # parameters.dl_dr_a = last_dr


            if enqueue_delay:
                time.sleep(enqueue_delay)

            remaining = len(file_bytes)
            if remaining <= 0:
                break

            if class_mode == "dynamic":
                next_class = cmd_decide_class(
                    client, remaining_bytes=remaining,
                    p_tx=p_tx, p_rx=p_rx, p_sleep=p_sleep,
                    l=l, optimize_interval=optimize_interval,
                ) or current_class
                if next_class != current_class:
                    if switch_class(
                        client, txack_monitor, dev_eui, next_class, mqtt_timeout,
                        uplink_monitor=uplink_monitor,
                    ):
                        if current_class == "C" and next_class == "A":
                            switch_from_class_C_to_A = 1
                        current_class = next_class
            i += 1

            if adr_schedule is not None and target_dr is not None:
                parameters.ul_dr = target_dr
                parameters.dl_dr_a = target_dr


        print(f"[FILE] Done: {i} chunks, {time.time()-start:.2f}s, class={current_class}, CRC32=0x{crc32:08X}")
    finally:
        txack_monitor.stop()
        uplink_monitor.stop()

def cmd_decide_class(
    client: ChirpStackClient,
    sync_radio: bool = False,
    wait_uplink: bool = False,
    dev_eui: str = DEV_EUI,
    remaining_bytes: int | None = None,
    mqtt_timeout: float = 30.0,
    p_tx: float | None = None, p_rx: float | None = None, p_sleep: float | None = None,
    l: float = 0.7,
    optimize_interval: bool = False,
) -> str | None:
    """Print the energy-model report and return the recommended class
    ('A' or 'C'). Optionally pushes a tighter report_interval to the
    device if Class A wins and the current interval is meaningfully
    above the duty-cycle floor."""
    if wait_uplink:
        if mqtt_wait_for_uplink(dev_eui, timeout=mqtt_timeout) is None:
            print("[CTRL] Could not refresh radio parameters via MQTT; aborting class decision")
            return None
        # Cache disabled: use the fresh in-memory uplink parameters.
    elif sync_radio:
        # Cache disabled: there are no cached values to apply.
        pass

    if remaining_bytes is not None:
        parameters.remaining_bytes = remaining_bytes
    if p_tx is not None:
        parameters.p_tx = p_tx
    if p_rx is not None:
        parameters.p_rx = p_rx
    if p_sleep is not None:
        parameters.p_sleep = p_sleep
    parameters.lambda_ = l

    try:
        model = EnergyModel(parameters)
        best = model.print_report()
    except ValueError as e:
        # Parameter validation failed (e.g. a payload/DR mismatch that
        # slipped through). Don't let a bad decision crash an in-progress
        # transfer — report it and let the caller keep the current class.
        print(f"[OPT] Could not evaluate energy model: {e}")
        return None

    if best == "A" and optimize_interval:
        t_min = model.ul_duty_cycle_wait
        current_interval = parameters.report_interval
        # Only push a change if it's a meaningful improvement — otherwise
        # every re-decision in the dynamic loop would needlessly re-send
        # the same set-interval command.
        if t_min < current_interval * 0.9:
            new_interval = max(1, math.ceil(t_min))
            print(f"[OPT] Class A selected — duty-cycle floor t_min={t_min:.3f}s is below "
                  f"current interval {current_interval:.3f}s; optimizing interval "
                  f"{current_interval:.3f}s -> {new_interval}s")
            cmd_set_interval(client, new_interval)
        else:
            print(f"[OPT] Current interval {current_interval:.3f}s is already close to the "
                  f"duty-cycle floor ({t_min:.3f}s) — leaving unchanged")

    return best


# CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ChirpStack downlink test tool for LoRaWAN Class A/C dynamic FUOTA"
    )
    p.add_argument("--server",    default=SERVER,    help="ChirpStack gRPC address")
    p.add_argument("--token",     default=API_TOKEN, help="API token")
    p.add_argument("--dev-eui",   default=DEV_EUI,   help="Device EUI")

    p.add_argument("--mqtt-host",     default=MQTT_HOST,     help="ChirpStack MQTT broker host")
    p.add_argument("--mqtt-port",     default=MQTT_PORT,     type=int, help="ChirpStack MQTT broker port")
    p.add_argument("--mqtt-username", default=MQTT_USERNAME, help="MQTT username, if the broker requires auth")
    p.add_argument("--mqtt-password", default=MQTT_PASSWORD, help="MQTT password, if the broker requires auth")

    sub = p.add_subparsers(dest="command", required=True)

    # switch-class
    sc = sub.add_parser("switch-class", help="Switch the device to Class A, B or C")
    sc.add_argument("--class", dest="cls", required=True, choices=["A", "B", "C"])

    rd = sub.add_parser("set-datarate", help="Set Datarate of the device to 0 through 5")
    rd.add_argument("--datarate", type=int, dest="datarate", required=True, default=5, choices=range(6))

    # send-file
    sf = sub.add_parser("send-file", help="Dynamically fragment, send, and class-switch a binary file")
    sf.add_argument("--file",            required=True, help="Binary file to send")
    sf.add_argument("--l",          type=float, default=0.65, help="Number between 0 and 1 to prioritize Energy vs. Latency (default 0.7)")
    sf.add_argument("--confirmed",       action="store_true",
                    help="Request an ACK for each downlink frame")
    sf.add_argument("--chunk-data-size", type=int, default=CHUNK_DATA_SIZE,
                    help=f"Bytes of file data per chunk (default {CHUNK_DATA_SIZE})")
    sf.add_argument("--enqueue-delay",   type=float, default=0.0,
                    help="Seconds to sleep between enqueue calls (default 0)")
    sf.add_argument("--wait-uplink",     action="store_true",
                    help="Block until the device's next uplink before sending, "
                         "to refresh SF/BW via MQTT rather than using the cache")
    sf.add_argument("--mqtt-timeout",    type=float, default=30.0,
                    help="Seconds to wait for a TXACK / uplink when applicable")
    sf.add_argument("--p-tx",    type=float, default=None, help="Override TX power draw (mW) for this run only")
    sf.add_argument("--p-rx",    type=float, default=None, help="Override RX power draw (mW) for this run only")
    sf.add_argument("--p-sleep", type=float, default=None, help="Override sleep power draw (mW) for this run only")
    sf.add_argument("--optimize-interval", action="store_true",
                    help="If Class A wins, also check whether shrinking report_interval "
                            "to t_min (the duty-cycle floor) raises utility, and push a "
                            "set-interval command to the device if so")
    sf.add_argument("--adr-override", action="store_true",
                    help="Evaluation hook: This triggers the evaluation setup, where a signal weakening is simulated in the middle the download process.")
    sf.add_argument("--rest-server", default=REST_SERVER,
                    help=f"ChirpStack REST gateway base URL, used only for "
                         f"--adr-override-at-chunk (default {REST_SERVER})")
    sf.add_argument("--class-mode", type=str, default="dynamic",
                    help="This is to switch between the proposed method and the baselines: 'static class A' and 'static class C'")

    # set-interval
    si = sub.add_parser("set-interval", help="Set the device uplink interval")
    si.add_argument("--seconds", type=int, required=True)

    # flush-queue
    sub.add_parser("flush-queue", help="Flush the ChirpStack downlink queue")

    # queue-status
    sub.add_parser("queue-status", help="Show pending queue depth")

    # decide-class
    dc = sub.add_parser("decide-class", help="Calculate best class to use (does not send/switch anything else)")
    dc.add_argument("--wait-uplink", action="store_true",
                     help="Block until the next uplink before deciding, to refresh SF/BW via MQTT")
    dc.add_argument("--mqtt-timeout", type=float, default=30.0)
    dc.add_argument("--n", type=int, default=200)
    dc.add_argument("--l", type=float, default=0.65,
                     help="Number between 0 and 1 to prioritize Energy vs. Latency (default 0.65)")
    dc.add_argument("--p-tx",    type=float, default=None, help="Override TX power draw (mW) for this run only")
    dc.add_argument("--p-rx",    type=float, default=None, help="Override RX power draw (mW) for this run only")
    dc.add_argument("--p-sleep", type=float, default=None, help="Override sleep power draw (mW) for this run only")
    dc.add_argument("--optimize-interval", action="store_true",
                     help="If Class A wins, also check whether shrinking report_interval "
                          "to t_min (the duty-cycle floor) raises utility, and push a "
                          "set-interval command to the device if so")

    # set-power / show-power (persistent power-parameter config)
    sp = sub.add_parser("set-power", help="Persist P_TX/P_RX/P_sleep (mW) to the cache file")
    sp.add_argument("--p-tx",    type=float, default=None, help="TX power draw (mW)")
    sp.add_argument("--p-rx",    type=float, default=None, help="RX power draw (mW)")
    sp.add_argument("--p-sleep", type=float, default=None, help="Sleep power draw (mW)")

    sub.add_parser("show-power", help="Show hardcoded / cached / effective P_TX, P_RX, P_sleep")

    # sync-radio (one-shot MQTT wait)
    sr = sub.add_parser("sync-radio", help="Wait for the next uplink and cache its SF/BW via MQTT")
    sr.add_argument("--timeout", type=float, default=30.0)

    # mqtt-listen (continuous)
    sub.add_parser("mqtt-listen", help="Continuously listen for uplinks and refresh the SF/BW cache")

    sub.add_parser("mqtt-wait-empty-queue", help="Wait for empty queue")

    return p


def main():
    start = time.time()
    global CACHE_FILE, STATE_FILE
    args = build_parser().parse_args()
    # CACHE_FILE/STATE_FILE default to the DEV_EUI constant at import time;
    # rebind them to whatever device this invocation actually targets so
    # --dev-eui gets its own radio/power/state cache instead of always
    # hitting the default one.
    CACHE_FILE = f"{CACHE_FILE_PREFIX}_{args.dev_eui}.json"
    STATE_FILE = f"{STATE_FILE_PREFIX}_{args.dev_eui}.json"

    # Cache disabled: use in-memory defaults for every invocation.

    global MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
    MQTT_HOST     = args.mqtt_host
    MQTT_PORT     = args.mqtt_port
    MQTT_USERNAME = args.mqtt_username
    MQTT_PASSWORD = args.mqtt_password

    client = ChirpStackClient(args.server, args.token, args.dev_eui)

    if args.command == "switch-class":
        cmd_switch_class(client, args.cls)

    if args.command == "set-datarate":
        cmd_set_datarate(datarate=args.datarate)

    elif args.command == "send-file":
        with open(args.file, "rb") as f:
            data = f.read()
            cmd_send_file(
                client,
                data,
                l=args.l,
                # chunk_data_size=args.chunk_data_size,
                confirmed=args.confirmed,
                enqueue_delay=args.enqueue_delay,
                # wait_uplink=args.wait_uplink,
                dev_eui=args.dev_eui,
                mqtt_timeout=args.mqtt_timeout,
                p_tx=args.p_tx,
                p_rx=args.p_rx,
                p_sleep=args.p_sleep,
                optimize_interval=args.optimize_interval,
                adr_override=args.adr_override,
                rest_server=args.rest_server,
                rest_token=args.token,
                class_mode=args.class_mode,
            )

    elif args.command == "set-interval":
        cmd_set_interval(client, args.seconds)

    elif args.command == "flush-queue":
        cmd_flush_queue(client)

    elif args.command == "queue-status":
        cmd_queue_status(client)

    elif args.command == "decide-class":
        cmd_decide_class(
            client,
            wait_uplink=args.wait_uplink,
            dev_eui=args.dev_eui,
            n=args.n,
            mqtt_timeout=args.mqtt_timeout,
            p_tx=args.p_tx,
            p_rx=args.p_rx,
            p_sleep=args.p_sleep,
            l=args.l,
            optimize_interval=args.optimize_interval,
        )

    elif args.command == "set-power":
        saved = save_power_config(args.p_tx, args.p_rx, args.p_sleep)
        print(f"[POWER] Applied for this process: {saved}")

    elif args.command == "show-power":
        hardcoded = {"p_tx": parameters.p_tx, "p_rx": parameters.p_rx, "p_sleep": parameters.p_sleep}
        print(f"Hardcoded default : {hardcoded}")
        print(f"Effective (used)  : P_TX={parameters.p_tx} mW  "
              f"P_RX={parameters.p_rx} mW  P_sleep={parameters.p_sleep} mW")

    elif args.command == "sync-radio":
        mqtt_wait_for_uplink(
            args.dev_eui,
            timeout=args.timeout,
            host=args.mqtt_host,
            port=args.mqtt_port,
            username=args.mqtt_username,
            password=args.mqtt_password,
        )

    elif args.command == "mqtt-listen":
        cmd_mqtt_listen(
            args.dev_eui,
            host=args.mqtt_host,
            port=args.mqtt_port,
            username=args.mqtt_username,
            password=args.mqtt_password,
        )

    elif args.command == "mqtt-wait-empty-queue":
        client.wait_for_empty_queue()

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()