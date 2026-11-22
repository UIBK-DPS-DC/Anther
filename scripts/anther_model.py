import numpy as np

DR_PARAMS = {
    2: {"SF": 10, "BW": 125e3, "MTU": 51},
    3: {"SF": 9,  "BW": 125e3, "MTU": 115},
    4: {"SF": 8,  "BW": 125e3, "MTU": 222},
    5: {"SF": 7,  "BW": 125e3, "MTU": 222},
}

DEFAULT_POWER = {
    "P_tx": 0.700,
    "P_rx": 0.400,
    "P_idle": 0.220,
    "P_sleep": 0.004,
}

# Timing & protocol constants (aligned with Section 3.4 & Table 4)
T_INTERVAL = 60.0
T_RX1 = 1.0
T_TX_OH = 0.05
T_RX_OH = 0.05
T_DL_INT = 5.0
N_UL_BYTES = 15
D_UL = 0.01

# Class B LoRaWAN TS001 constants
T_BEACON = 128.0          # Gateway beacon periodicity (s)
T_BEACON_RX = 0.032       # Beacon acquisition window (s)
PING_PERIODICITY_K = 3    # k in [0..7] -> T_ping = 2^3 * 0.96 = 7.68 s
T_PING_RX = 0.030         # Preamble detection window per ping slot (s)
MTU_RX2 = 51              # Default RX2 / ping-slot payload cap (bytes)


def lora_toa(n_bytes, sf, bw=125e3, cr=1, preamble=8):
    t_sym = (2 ** sf) / bw
    de = 1 if (sf >= 11 and bw == 125e3) else 0
    num = 8 * n_bytes - 4 * sf + 28
    den = 4 * (sf - 2 * de)
    payload_syms = 8 + max(np.ceil(num / den) * (cr + 4), 0)
    return (preamble + 4.25 + payload_syms) * t_sym


def evaluate_classes(
    n_bytes,
    dr,
    power=DEFAULT_POWER,
    t_interval=T_INTERVAL,
    ping_periodicity_k=PING_PERIODICITY_K,
    candidate_classes=("A", "B", "C"),
):
    """
    Closed-form completion latency T_m(n_m) and energy E_m(n_m) for m in candidate_classes.
    Returns a dict: {m: (T_m, E_m)}.
    """
    if n_bytes <= 0:
        return {m: (0.0, 0.0) for m in candidate_classes}

    sf, bw, mtu_a = DR_PARAMS[dr]["SF"], DR_PARAMS[dr]["BW"], DR_PARAMS[dr]["MTU"]
    t_air_ul = lora_toa(N_UL_BYTES, sf, bw)
    t_duty = t_air_ul / D_UL
    t_int = max(t_interval, t_duty)

    results = {}

    # --- Class A (Uplink-Driven RX1) ---
    if "A" in candidate_classes:
        n_a = int(np.ceil(n_bytes / mtu_a))
        t_air_dl_a = lora_toa(mtu_a, sf, bw)
        t_a = (n_a - 1) * t_int + n_a * (T_TX_OH + t_air_ul + T_RX1 + t_air_dl_a + T_RX_OH)
        e_a = n_a * (
            power["P_idle"] * T_TX_OH + power["P_tx"] * t_air_ul + power["P_idle"] * T_RX1 +
            power["P_rx"] * t_air_dl_a + power["P_idle"] * T_RX_OH + power["P_sleep"] * t_int
        )
        results["A"] = (t_a, e_a)

    # --- Class B (Beacon-Synchronized Ping Slots) ---
    if "B" in candidate_classes:
        n_b = int(np.ceil(n_bytes / MTU_RX2))
        t_air_dl_b = lora_toa(MTU_RX2, 12, 125e3)
        t_ping = (2 ** ping_periodicity_k) * 0.96
        t_slot = np.ceil(max(t_ping, T_DL_INT) / t_ping) * t_ping

        t_b = (n_b - 1) * t_slot + t_air_dl_b + T_RX_OH
        n_beacons = np.floor(t_b / T_BEACON)
        n_pings = np.floor(t_b / t_ping)
        t_rx_b = n_b * (t_air_dl_b + T_RX_OH) + n_pings * T_PING_RX + n_beacons * T_BEACON_RX
        t_sleep_b = max(0.0, t_b - t_rx_b)
        e_b = power["P_rx"] * t_rx_b + power["P_sleep"] * t_sleep_b
        results["B"] = (t_b, e_b)

    # --- Class C (Continuous RX2 Reception) ---
    if "C" in candidate_classes:
        n_c = int(np.ceil(n_bytes / MTU_RX2))
        t_air_dl_c = lora_toa(MTU_RX2, 12, 125e3)
        n_dl_per_ul = max(1, int(np.floor(t_int / T_DL_INT)))
        u_total = int(np.ceil(n_c / n_dl_per_ul))

        t_c, e_c = 0.0, 0.0
        rem_frags = n_c
        for u in range(1, u_total + 1):
            n_u = min(n_dl_per_ul, rem_frags)
            rem_frags -= n_u
            if u < u_total:
                t_rx_u = t_int - t_air_ul - T_TX_OH
            else:
                t_rx_u = t_air_dl_c + (n_u - 1) * T_DL_INT + n_u * T_RX_OH
            t_c += t_air_ul + T_TX_OH + t_rx_u
            e_c += power["P_tx"] * (t_air_ul + T_TX_OH) + power["P_rx"] * t_rx_u
        results["C"] = (t_c, e_c)

    return results


def geometric_utility(t_val, e_val, lam):
    return (e_val ** -(1.0 - lam)) * (t_val ** -lam)


def select_optimal_class(n_bytes, dr, lam, candidate_classes=("A", "B", "C"), power=DEFAULT_POWER):
    evals = evaluate_classes(n_bytes, dr, power=power, candidate_classes=candidate_classes)
    best_m = max(candidate_classes, key=lambda m: geometric_utility(*evals[m], lam))
    return best_m, evals


def adr_schedule_dr(t_sec):
    t_min = (t_sec / 60.0) % 60.0
    if t_min < 5.0:
        return 5
    elif t_min < 13.3:
        return 4
    elif t_min < 21.7:
        return 3
    elif t_min < 43.3:
        return 2
    elif t_min < 51.7:
        return 3
    else:
        return 4
