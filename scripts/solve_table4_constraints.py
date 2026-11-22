import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from anther_model import (
    DR_PARAMS,
    DEFAULT_POWER,
    T_INTERVAL,
    T_RX1,
    T_TX_OH,
    T_RX_OH,
    T_DL_INT,
    N_UL_BYTES,
    D_UL,
    T_BEACON,
    T_BEACON_RX,
    PING_PERIODICITY_K,
    T_PING_RX,
    MTU_RX2,
    lora_toa,
    select_optimal_class,
    adr_schedule_dr,
)


def simulate_transfer_multiclass(
    total_bytes,
    lam=0.5,
    candidate_classes=("A", "B", "C"),
    power=DEFAULT_POWER,
    ping_periodicity_k=PING_PERIODICITY_K,
):
    t_elapsed = 0.0
    e_total = 0.0
    rem_bytes = float(total_bytes)
    current_class = "A"
    t_ping = (2 ** ping_periodicity_k) * 0.96

    while rem_bytes > 0:
        dr = adr_schedule_dr(t_elapsed)
        chosen, _ = select_optimal_class(
            rem_bytes, dr, lam, candidate_classes=candidate_classes, power=power
        )

        if chosen != current_class:
            sf, bw = DR_PARAMS[dr]["SF"], DR_PARAMS[dr]["BW"]
            t_sw = lora_toa(15, sf, bw) + T_RX1 + lora_toa(8, sf, bw) + T_TX_OH + T_RX_OH
            e_sw = (
                power["P_tx"] * lora_toa(15, sf, bw)
                + power["P_rx"] * lora_toa(8, sf, bw)
                + power["P_idle"] * T_RX1
            )
            t_elapsed += t_sw
            e_total += e_sw
            current_class = chosen

        if chosen == "A":
            sf, bw, mtu = DR_PARAMS[dr]["SF"], DR_PARAMS[dr]["BW"], DR_PARAMS[dr]["MTU"]
            chunk = min(rem_bytes, mtu)
            t_ul = lora_toa(N_UL_BYTES, sf, bw)
            t_dl = lora_toa(chunk, sf, bw)
            t_int = max(T_INTERVAL, t_ul / D_UL)
            dt = t_int + T_TX_OH + t_ul + T_RX1 + t_dl + T_RX_OH
            de = (
                power["P_idle"] * T_TX_OH
                + power["P_tx"] * t_ul
                + power["P_idle"] * T_RX1
                + power["P_rx"] * t_dl
                + power["P_idle"] * T_RX_OH
                + power["P_sleep"] * t_int
            )
        elif chosen == "B":
            chunk = min(rem_bytes, MTU_RX2)
            t_dl = lora_toa(chunk, 12, 125e3)
            dt = np.ceil(max(t_ping, T_DL_INT) / t_ping) * t_ping
            t_rx_step = (
                t_dl
                + T_RX_OH
                + (dt / t_ping) * T_PING_RX
                + (dt / T_BEACON) * T_BEACON_RX
            )
            de = power["P_rx"] * t_rx_step + power["P_sleep"] * max(0.0, dt - t_rx_step)
        else:  # Class C
            chunk = min(rem_bytes, MTU_RX2)
            dt = T_DL_INT
            de = power["P_rx"] * dt

        t_elapsed += dt
        e_total += de
        rem_bytes -= chunk

    return t_elapsed, e_total


def solve_lambda_star(
    n_bytes=51200,
    deadline_s=np.inf,
    energy_budget_j=np.inf,
    candidate_classes=("A", "B", "C"),
    eps=1e-4,
):
    t0, e0 = simulate_transfer_multiclass(n_bytes, lam=0.0, candidate_classes=candidate_classes)
    if t0 <= deadline_s and e0 <= energy_budget_j:
        return 0.0, t0, e0

    t1, e1 = simulate_transfer_multiclass(n_bytes, lam=1.0, candidate_classes=candidate_classes)
    if t1 > deadline_s or e1 > energy_budget_j:
        return None, t1, e1

    low, high = 0.0, 1.0
    while (high - low) > eps:
        mid = 0.5 * (low + high)
        t_sim, e_sim = simulate_transfer_multiclass(
            n_bytes, lam=mid, candidate_classes=candidate_classes
        )
        if t_sim <= deadline_s and e_sim <= energy_budget_j:
            high = mid
        else:
            low = mid

    t_opt, e_opt = simulate_transfer_multiclass(
        n_bytes, lam=high, candidate_classes=candidate_classes
    )
    return high, t_opt, e_opt


if __name__ == "__main__":
    scenarios = [
        ("Deadline-constrained only", 8000.0, np.inf),
        ("Energy-constrained only", np.inf, 180.0),
        ("Jointly constrained", 14000.0, 400.0),
        ("Conflicting (Infeasible)", 8000.0, 180.0),
    ]

    for label, c_set in [("C_hw = {A, C}", ("A", "C")), ("C_full = {A, B, C}", ("A", "B", "C"))]:
        print(f"\n=== Constraint Solver Results ({label}) ===")
        for name, d_lim, e_lim in scenarios:
            lam_star, t_res, e_res = solve_lambda_star(
                51200, deadline_s=d_lim, energy_budget_j=e_lim, candidate_classes=c_set
            )
            if lam_star is None:
                print(f"{name:28s} | INFEASIBLE (min T={t_res:.1f}s, min E={e_res:.2f}J)")
            else:
                print(
                    f"{name:28s} | lambda*={lam_star:.4f} | T={t_res:8.1f}s | E={e_res:8.2f}J"
                )
