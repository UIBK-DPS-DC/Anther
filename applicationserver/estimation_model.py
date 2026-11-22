from dataclasses import dataclass, field
import math

# DR -> (SF, BW_Hz, max_payload_bytes) — EU868, repeater-compatible caps
DATA_RATES = {
    0: (12, 125_000, 51),
    1: (11, 125_000, 51),
    2: (10, 125_000, 51),
    3: (9,  125_000, 115),
    4: (8,  125_000, 222),
    5: (7,  125_000, 222),
}

# Per-fragment header/framing overhead subtracted from a DR's raw payload
# cap to get the usable firmware bytes per chunk. Must match whatever
# cmd_send_file actually subtracts (currently 2 bytes there).
FRAGMENT_OVERHEAD_BYTES = 2


def dr_for_sf_bw(sf: int, bw_hz: int) -> int | None:
    """
    Reverse-lookup an EU868 DR index for an observed (SF, BW) pair.
    """
    for dr, (dr_sf, dr_bw, _max_payload) in DATA_RATES.items():
        if dr_sf == sf and dr_bw == bw_hz:
            return dr
    return None


@dataclass
class EnergyParameters:
    # Experiment — bytes of firmware still to be transferred. Each class
    # derives its OWN fragment count from this, since Class A and Class C
    # can have different usable payload sizes per fragment (see dl_dr_a /
    # class_c_dr below).
    remaining_bytes: int

    # Application payloads (small periodic status/report uplink, unrelated
    # to the firmware fragments themselves)
    uplink_payload_bytes: int = 5

    # Time between transactions / reports
    report_interval: float = 60.0

    # Class C fragment pacing — NOT derivable from PHY/firmware; this is
    # governed by the network server's scheduler config (e.g. ChirpStack's
    # network.scheduler.interval / class_c_lock_duration). Set manually
    # per measured/configured value for each test condition.
    class_c_fragment_interval: float = 5.0

    # LoRa PHY — uplink DR is a single shared value (ADR affects the
    # device's uplink regardless of class). Downlink DR is NOT shared:
    #   - dl_dr_a: Class A's RX1 datarate. RX1 is derived from the uplink
    #     that unlocked it; this deployment uses RX1DROffset=0, so it
    #     defaults to mirroring ul_dr. Override explicitly if your
    #     RX1DROffset isn't 0.
    #   - class_c_dr: Class C's RX2 datarate. RX2 is fixed by region
    #     (DR0 for EU868 by default) and does NOT move with ul_dr/ADR —
    #     it must be supplied independently, never derived from the
    #     currently observed uplink/RX1 state.
    ul_dr: int = 5
    dl_dr_a: int | None = None  # None => mirrors ul_dr (RX1DROffset=0)
    class_c_dr: int = 0
    cr: int = 1
    preamble: int = 7
    crc: int = 0.
    header: int = -1

    # Radio power -- mW
    p_tx: float = 700.0        # 200mA * 5V  (based on the datasheet for 14dBm)
    p_rx: float = 400.0         # 90mA * 5V (based on datasheet)
    p_idle: float = 220.0       # mW, idle state (based on measurements)
    p_sleep: float = 4.0        # deep sleep (based on very consistent 0.8mA measurements)

    # Class A receive timing
    rx1_delay: float = 1.0
    rx2_delay: float = 2.0
    t_tx_overhead: float = 0.35  # s, fixed MCU/SPI overhead per TX (measured)
    t_rx_overhead: float = 0.26   # s, fixed ESP32 overhead for every RX (measured)
    rx_window: float = 0.05

    # Duty cycle (enforced as a lower bound for interval timing to ensure compliance)
    duty_cycle: float = 0.01       # uplink, e.g. 1%
    dl_duty_cycle: float = 0.10    # downlink sub-band, e.g. 10% (869.525 MHz EU868)

    # Class B
    t_beacon_rx: float = 0.12
    t_ping_rx: float = 0.128
    t_beacon: float = 128.0

    # Utility
    lambda_: float = 0.5

    def __post_init__(self):
        if self.dl_dr_a is None:
            self.dl_dr_a = self.ul_dr

    def __str__(self):
        return (
            "EnergyParameters(\n"
            f"\tremaining_bytes={self.remaining_bytes},\n"
            f"\tuplink_payload_bytes={self.uplink_payload_bytes},\n"
            f"\treport_interval={self.report_interval},\n"
            f"\tclass_c_fragment_interval={self.class_c_fragment_interval},\n"
            f"\tul_dr={self.ul_dr},\n"
            f"\tdl_dr_a={self.dl_dr_a},\n"
            f"\tclass_c_dr={self.class_c_dr},\n"
            f"\tduty_cycle={self.duty_cycle},\n"
            f"\tdl_duty_cycle={self.dl_duty_cycle},\n"
            f"\tlambda_={self.lambda_},\n"
            f"\tp_tx={self.p_tx},\n"
            f"\tp_rx={self.p_rx},\n"
            f"\tp_idle={self.p_idle},\n"
            f"\tp_sleep={self.p_sleep}\n"
            ")"
        )


class EnergyModel:

    def __init__(self, params: EnergyParameters):
        self.p = params

        if self.p.remaining_bytes <= 0:
            raise ValueError("remaining_bytes must be positive")
        if self.p.duty_cycle <= 0:
            raise ValueError("duty_cycle must be positive")
        if self.p.dl_duty_cycle <= 0:
            raise ValueError("dl_duty_cycle must be positive")
        if self.p.ul_dr not in DATA_RATES:
            raise ValueError(f"Unsupported uplink DR: {self.p.ul_dr}. Supported: {sorted(DATA_RATES)}")
        if self.p.dl_dr_a not in DATA_RATES:
            raise ValueError(f"Unsupported Class A downlink (RX1) DR: {self.p.dl_dr_a}. Supported: {sorted(DATA_RATES)}")
        if self.p.class_c_dr not in DATA_RATES:
            raise ValueError(f"Unsupported Class C downlink (RX2) DR: {self.p.class_c_dr}. Supported: {sorted(DATA_RATES)}")
        if self.p.p_tx < 0:
            raise ValueError("p_tx must be non-negative")
        if self.p.p_rx < 0:
            raise ValueError("p_rx must be non-negative")
        if self.p.p_sleep < 0:
            raise ValueError("p_sleep must be non-negative")
        if self.p.class_c_fragment_interval < 0:
            raise ValueError("class_c_fragment_interval must be positive")

        # Usable firmware bytes per fragment, per class — these can differ
        # substantially (Class A follows the live link's DR; Class C is
        # capped by whatever the fixed RX2 DR allows, often much smaller).
        self.dl_a_max_payload = DATA_RATES[self.p.dl_dr_a][2] - FRAGMENT_OVERHEAD_BYTES
        self.dl_c_max_payload = DATA_RATES[self.p.class_c_dr][2] - FRAGMENT_OVERHEAD_BYTES
        if self.dl_a_max_payload <= 0:
            raise ValueError(
                f"DR{self.p.dl_dr_a} payload cap ({DATA_RATES[self.p.dl_dr_a][2]} B) is too small "
                f"for the {FRAGMENT_OVERHEAD_BYTES} B fragment overhead"
            )
        if self.dl_c_max_payload <= 0:
            raise ValueError(
                f"DR{self.p.class_c_dr} payload cap ({DATA_RATES[self.p.class_c_dr][2]} B) is too small "
                f"for the {FRAGMENT_OVERHEAD_BYTES} B fragment overhead"
            )

        # Fragment counts implied by the same remaining_bytes, one per class.
        self.n_a = math.ceil(self.p.remaining_bytes / self.dl_a_max_payload) + 2
        self.n_c = math.ceil(self.p.remaining_bytes / self.dl_c_max_payload)

        # Payload-cap validation (uplink only — downlink fragments are
        # always sized to fit by construction above)
        if self.p.uplink_payload_bytes > self.ul_max_payload:
            raise ValueError(
                f"uplink_payload_bytes ({self.p.uplink_payload_bytes}) exceeds "
                f"DR{self.p.ul_dr} max ({self.ul_max_payload})"
            )

    @staticmethod
    def _dr_params(dr: int):
        return DATA_RATES[dr]

    @property
    def ul_sf(self): return self._dr_params(self.p.ul_dr)[0]

    @property
    def ul_bw_hz(self): return self._dr_params(self.p.ul_dr)[1]

    @property
    def ul_max_payload(self): return self._dr_params(self.p.ul_dr)[2]

    @property
    def dl_a_sf(self): return self._dr_params(self.p.dl_dr_a)[0]

    @property
    def dl_a_bw_hz(self): return self._dr_params(self.p.dl_dr_a)[1]

    @property
    def dl_c_sf(self): return self._dr_params(self.p.class_c_dr)[0]

    @property
    def dl_c_bw_hz(self): return self._dr_params(self.p.class_c_dr)[1]

    def airtime(self, payload_bytes: int, sf: int, bw: int) -> float:
        t_sym = (2 ** sf) / bw
        t_preamble = (self.p.preamble + 4.25) * t_sym
        ldro = 1 if sf >= 11 and bw == 125_000 else 0

        numerator = (
            8 * payload_bytes - 4 * sf + 28
            + 16 * self.p.crc - 20 * self.p.header
        )
        denominator = 4 * (sf - 2 * ldro)

        payload_symbols = 8 + max(
            math.ceil(numerator / denominator) * (self.p.cr + 4), 0
        )

        return t_preamble + payload_symbols * t_sym

    @property
    def t_air_ul(self):
        return self.airtime(self.p.uplink_payload_bytes, sf=self.ul_sf, bw=self.ul_bw_hz)

    @property
    def t_air_dl_a(self):
        """Airtime of one Class A (RX1) firmware fragment, sized to fill dl_dr_a's cap."""
        return self.airtime(self.dl_a_max_payload, sf=self.dl_a_sf, bw=self.dl_a_bw_hz)

    @property
    def t_air_dl_c(self):
        """Airtime of one Class C (RX2) firmware fragment, sized to fill class_c_dr's cap."""
        return self.airtime(self.dl_c_max_payload, sf=self.dl_c_sf, bw=self.dl_c_bw_hz)

    @staticmethod
    def energy(power_mw: float, time_s: float) -> float:
        return power_mw * time_s / 1000.0

    @property
    def ul_duty_cycle_wait(self):
        """Minimum legal interval between uplink starts."""
        return self.t_air_ul / self.p.duty_cycle

    @property
    def dl_c_duty_cycle_wait(self):
        """Minimum legal interval between Class C (RX2) downlink starts."""
        return self.t_air_dl_c / self.p.dl_duty_cycle

    def class_a(self):
        """
        Class A: UL, then idle until RX1 (rx1_delay), then RX window.
        Assumes downlink is received in RX1, at dl_dr_a (mirrors ul_dr by
        default, per RX1DROffset=0). `report_interval` is clamped to the
        duty-cycle minimum to enforce legal transmission rates.
        """
        n = self.n_a
        t_ul = self.t_air_ul
        # print("UL TOA: ", t_ul)
        t_dl = self.t_air_dl_a
        # print("DL TOA: ", t_dl)
        interval = max(self.p.report_interval, self.ul_duty_cycle_wait) # Enforce uplink duty cycle as a lower limit

        # print("SendRecieve time: ", self.p.t_tx_overhead + t_ul + self.p.rx1_delay + t_dl + self.p.t_rx_overhead)

        total_time = (n-1) * interval + n * (self.p.t_tx_overhead + t_ul + self.p.rx1_delay + t_dl + self.p.t_rx_overhead)

        e_tx_overhead = n * self.energy(self.p.p_idle, self.p.t_tx_overhead)
        e_tx = n * self.energy(self.p.p_tx, t_ul)
        e_idle = n * self.energy(self.p.p_idle, self.p.rx1_delay)
        e_rx = n * self.energy(self.p.p_rx, t_dl)
        e_rx_overhead = n * self.energy(self.p.p_idle, self.p.t_rx_overhead)
        e_sleep = n * self.energy(self.p.p_sleep, interval)

        total_energy = e_tx_overhead + e_tx + e_idle + e_rx + e_rx_overhead + e_sleep

        return {
            "fragments": n,
            "latency": total_time,
            "energy": total_energy,
            "tx_energy": e_tx,
            "idle_energy": e_idle,
            "rx_energy": e_rx,
            "sleep_energy": e_sleep,
            "uplink_airtime": t_ul,
            "downlink_airtime": t_dl,
            "interval": interval,
            "required_ul_interval": self.ul_duty_cycle_wait,
            "is_legal_with_duty_cycle": self.p.report_interval >= self.ul_duty_cycle_wait,
        }

    def class_c(self):
        """
        Class C: same Uplink behaviour ar Class A, but downlinks are recieved with fixed DR0 in RX2, because of region EU868.
        Downlinks are continuesly recieved in max speed applying to duty cycle restrictions, so the Datarate should not matter much.
        """
        n = self.n_c
        t_ul = self.t_air_ul
        t_dl = self.t_air_dl_c
        uplink_interval = max(self.p.report_interval, self.ul_duty_cycle_wait)
        dl_interval = self.p.class_c_fragment_interval
        dl_per_interval = int(uplink_interval // dl_interval)
        uplinks = 0
        downlinks = 0
        remaining = n
        total_time = 0.0
        e_tx = e_rx = 0.0

        while remaining > 0:
            uplinks += 1
            total_time += t_ul + self.p.t_tx_overhead
            e_tx += self.energy(self.p.p_tx, t_ul + self.p.t_tx_overhead)
            dl_count = min(dl_per_interval, remaining)
            remaining -= dl_count
            downlinks += dl_count
            if remaining > 0:
                rx_time = uplink_interval - t_ul - self.p.t_tx_overhead
            else:
                rx_time = t_dl + (dl_count - 1) * dl_interval + dl_count * self.p.t_rx_overhead
            total_time += rx_time
            e_rx += self.energy(self.p.p_rx, rx_time)

        return {
            "fragments": n,
            "uplinks": uplinks,
            "downlinks": downlinks,
            "downlinks_per_uplink_interval": dl_per_interval,
            "latency": total_time,
            "energy": e_tx + e_rx,
            "tx_energy": e_tx,
            "idle_energy": 0.0,
            "rx_energy": e_rx,
            "sleep_energy": 0.0,
            "uplink_airtime": t_ul,
            "downlink_airtime": t_dl,
            "uplink_interval": uplink_interval,
            "downlink_interval": dl_interval,
            "required_dl_interval": dl_interval,
            "is_legal_with_duty_cycle": True,
        }

    def utility(self, energy, latency):
        if energy <= 0 or latency <= 0:
            return float("-inf")
        quality = 1
        lam = self.p.lambda_
        return quality * energy ** -(1 - lam) * latency ** -lam

    def select_class(self):
        a = self.class_a()
        c = self.class_c()

        min_energy = min(a["energy"], c["energy"])
        min_latency = min(a["latency"], c["latency"])

        if min_energy <= 0 or min_latency <= 0:
            return "A", a, c, {}

        energy_ratio_a = a["energy"] / min_energy
        energy_ratio_c = c["energy"] / min_energy
        latency_ratio_a = a["latency"] / min_latency
        latency_ratio_c = c["latency"] / min_latency

        utility_a = self.utility(energy_ratio_a, latency_ratio_a)
        utility_c = self.utility(energy_ratio_c, latency_ratio_c)

        best = "A" if utility_a >= utility_c else "C"

        details = {
            "energy_ratio_a": energy_ratio_a,
            "energy_ratio_c": energy_ratio_c,
            "latency_ratio_a": latency_ratio_a,
            "latency_ratio_c": latency_ratio_c,
            "utility_a": utility_a,
            "utility_c": utility_c,
        }
        return best, a, c, details

    def print_report(self):
        a = self.class_a()
        c = self.class_c()
        best, _, _, details = self.select_class()

        print()
        print("===================================================")
        print("Energy Model")
        print("===================================================")
        print(f"Remaining bytes     : {self.p.remaining_bytes} B")
        print(f"Uplink Payload      : {self.p.uplink_payload_bytes} B (DR{self.p.ul_dr} cap: {self.ul_max_payload} B)")
        print(f"UL DR / SF          : {self.p.ul_dr} / {self.ul_sf}")
        print(f"UL Time On Air      : {self.t_air_ul:.6f} s")
        print(f"P-TX/P-RX/P-Idle/P-Sleep : {self.p.p_tx}/{self.p.p_rx}/{self.p.p_idle}/{self.p.p_sleep} mW")
        print(f"UL Duty-Cycle       : {self.p.duty_cycle}  (min legal interval: {self.ul_duty_cycle_wait:.3f} s)")
        print(f"DL Duty-Cycle       : {self.p.dl_duty_cycle}")

        print()
        print("Class A  (RX1 — DR tracks the uplink)")
        print("-------------------------------------------")
        print(f"DL DR / SF          : {self.p.dl_dr_a} / {self.dl_a_sf}")
        print(f"Fragment size       : {self.dl_a_max_payload} B (DR{self.p.dl_dr_a} cap: {self.ul_max_payload if False else DATA_RATES[self.p.dl_dr_a][2]} B)")
        print(f"Fragments           : {a['fragments']}")
        print(f"UL Time On Air      : {a['uplink_airtime']:.6f} s")
        print(f"DL Time On Air      : {a['downlink_airtime']:.6f} s")
        print(f"Latency             : {a['latency']:.3f} s")
        print(f"Energy              : {a['energy']:.6f} J")
        print(f"  TX                : {a['tx_energy']:.6f} J")
        print(f"  Idle              : {a['idle_energy']:.6f} J")
        print(f"  RX                : {a['rx_energy']:.6f} J")
        print(f"  Sleep             : {a['sleep_energy']:.6f} J")
        print(f"  UL interval       : {a['interval']:.3f} s")
        print(f"  Legal Config      : {a['is_legal_with_duty_cycle']}")

        print()
        print("Class C  (RX2 — fixed regional DR, independent of the uplink)")
        print("-------------------------------------------")
        print(f"DL DR / SF          : {self.p.class_c_dr} / {self.dl_c_sf}")
        print(f"Fragment size       : {self.dl_c_max_payload} B (DR{self.p.class_c_dr} cap: {DATA_RATES[self.p.class_c_dr][2]} B)")
        print(f"Fragments           : {c['fragments']}")
        print(f"UL Time On Air      : {c['uplink_airtime']:.6f} s")
        print(f"DL Time On Air      : {c['downlink_airtime']:.6f} s")
        print(f"Latency             : {c['latency']:.3f} s")
        print(f"Energy              : {c['energy']:.6f} J")
        print(f"  TX                : {c['tx_energy']:.6f} J")
        print(f"  Idle              : {c['idle_energy']:.6f} J")
        print(f"  RX                : {c['rx_energy']:.6f} J")
        print(f"  Fragment interval : {c['downlink_interval']:.3f} s")
        print(f"  Legal Config      : {c['is_legal_with_duty_cycle']}")

        print()
        print(f"Recommendation (relative utility, lambda={self.p.lambda_:.2f})")
        print("-------------------------------------------")
        if details:
            print(f"Energy ratio  A / best : {details['energy_ratio_a']:.3f}")
            print(f"Energy ratio  C / best : {details['energy_ratio_c']:.3f}")
            print(f"Latency ratio A / best : {details['latency_ratio_a']:.3f}")
            print(f"Latency ratio C / best : {details['latency_ratio_c']:.3f}")
            print(f"Utility A              : {details['utility_a']:.6f}")
            print(f"Utility C              : {details['utility_c']:.6f}")
        print(f"Best class             : {best}")
        print()

        return best


    def estimate_time_and_energy(self):
        a = self.class_a()
        c = self.class_c()

        return {
            "class_a_latency": a['latency'],
            "class_a_energy": a['energy'],
            "class_c_latency": c['latency'],
            "class_c_energy": c['energy'],
        }


import numpy as np
import pandas as pd

# if __name__ == "__main__":

#     results = []

#     # Lambda: 0.0, 0.1, ..., 0.9
#     lambdas = np.arange(0.0, 1.0, 0.001)

#     # Remaining bytes: 64, 80, 96, ..., 51184
#     remaining_bytes_values = range(64, 51200, 16)

#     total = len(lambdas) * len(remaining_bytes_values)
#     count = 0

#     for lambda_ in lambdas:
#         for remaining_bytes in remaining_bytes_values:

#             sit1 = EnergyParameters(
#                 remaining_bytes=remaining_bytes,
#                 report_interval=60.0,
#                 class_c_fragment_interval=5,
#                 ul_dr=5,
#                 class_c_dr=0,
#                 duty_cycle=0.01,
#                 dl_duty_cycle=0.10,
#                 lambda_=lambda_,
#             )

#             model = EnergyModel(sit1)
#             _, _, _, details = model.select_class()

#             results.append({
#                 "lambda": lambda_,
#                 "remaining_bytes": remaining_bytes,
#                 "utility_a": details["utility_a"],
#                 "utility_c": details["utility_c"],
#             })

#             count += 1

#             if count % 1000 == 0:
#                 print(f"Progress: {count}/{total} ({100 * count / total:.1f}%)")

#     # Convert to DataFrame
#     df = pd.DataFrame(results)

#     # Save in a convenient format
#     df.to_csv("decision_boundary_analysis/utility_grid.csv", index=False)

#     print("\nDone!")
#     print(f"Evaluations: {len(df)}")
#     print("Saved:")
#     print("  utility_grid.csv")
#     print("  utility_grid.pkl")

#     print("\nFirst rows:")
#     print(df.head())


if __name__ == "__main__":

    results = []

    # Lambda: 0.0, 0.1, ..., 0.9
    lambdas = np.arange(0.0, 1.0, 0.001)

    # Remaining bytes: 64, 80, 96, ..., 51184
    remaining_bytes_values = range(64, 51200, 16)

    total = len(lambdas) * len(remaining_bytes_values)
    count = 0

    # for lambda_ in lambdas:
    
    for dr in [2,3]:
        sit1 = EnergyParameters(
            remaining_bytes=51200,
            report_interval=60.0,
            class_c_fragment_interval=5,
            ul_dr=dr,
            dl_dr_a=5,
            class_c_dr=0,
            duty_cycle=0.01,
            dl_duty_cycle=0.10,
            lambda_=0.7,
        )

        model = EnergyModel(sit1)
        _, _, _, details = model.select_class()
        model.print_report()

        # results.append({
        #     "lambda": 0.7,
        #     "remaining_bytes": remaining_bytes,
        #     "utility_a": details["utility_a"],
        #     "utility_c": details["utility_c"],
        # })

        # count += 1

        # if count % 1000 == 0:
        #     print(f"Progress: {count}/{total} ({100 * count / total:.1f}%)")

    # Convert to DataFrame
    # df = pd.DataFrame(results)

    # Save in a convenient format
    # df.to_csv("decision_boundary_analysis/utility_grid.csv", index=False)

    # print("\nDone!")
    # print(f"Evaluations: {len(df)}")
    # print("Saved:")
    # print("  utility_grid.csv")
    # print("  utility_grid.pkl")

    # print("\nFirst rows:")
    # print(df.head())