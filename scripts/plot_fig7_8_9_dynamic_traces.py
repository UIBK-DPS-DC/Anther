import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FIG_DIR = ROOT_DIR / "figures" / "plots"
DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 1.5
import numpy as np
import matplotlib.pyplot as plt
from plot_style import apply_clean_style, save_figure, tab10
from anther_model import (
    DR_PARAMS, DEFAULT_POWER, T_INTERVAL, T_RX1, T_TX_OH, T_RX_OH,
    T_DL_INT, N_UL_BYTES, D_UL, MTU_RX2,
    lora_toa, evaluate_classes, geometric_utility, adr_schedule_dr,
)


def simulate_transfer(total_bytes, mode="dynamic", lam=0.5, power=DEFAULT_POWER):
    t_elapsed = 0.0
    e_total = 0.0
    rem_bytes = float(total_bytes)
    current_class = "A"

    trace_t = [0.0]
    trace_e = [0.0]
    trace_class = []

    while rem_bytes > 0:
        dr = adr_schedule_dr(t_elapsed)
        if mode == "A":
            chosen = "A"
        elif mode == "C":
            chosen = "C"
        else:
            evals = evaluate_classes(rem_bytes, dr, power=power, candidate_classes=("A", "C"))
            u_a = geometric_utility(*evals["A"], lam)
            u_c = geometric_utility(*evals["C"], lam)
            chosen = "C" if u_c > u_a else "A"

        if chosen != current_class:
            sf, bw = DR_PARAMS[dr]["SF"], DR_PARAMS[dr]["BW"]
            t_sw = lora_toa(15, sf, bw) + T_RX1 + lora_toa(8, sf, bw) + T_TX_OH + T_RX_OH
            e_sw = power["P_tx"] * lora_toa(15, sf, bw) + power["P_rx"] * lora_toa(8, sf, bw) + power["P_idle"] * T_RX1
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
                power["P_idle"] * T_TX_OH + power["P_tx"] * t_ul + power["P_idle"] * T_RX1 +
                power["P_rx"] * t_dl + power["P_idle"] * T_RX_OH + power["P_sleep"] * t_int
            )
        else:
            chunk = min(rem_bytes, MTU_RX2)
            dt = T_DL_INT
            de = power["P_rx"] * dt

        t_elapsed += dt
        e_total += de
        rem_bytes -= chunk
        trace_t.append(t_elapsed)
        trace_e.append(e_total)
        trace_class.append(chosen)

    return t_elapsed, e_total, np.array(trace_t), np.array(trace_e), trace_class


def plot_dynamic_experiments(out_dir=None):
    if out_dir is None:
        out_dir = DEFAULT_FIG_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes_kib = np.array([1, 2, 5, 10, 50])
    lambdas = [0.1, 0.4, 0.65, 0.7]
    lam_colors = [tab10(2), tab10(9), tab10(1), tab10(4)]
    lam_markers = ["^", "v", "D", "P"]

    res_A = [simulate_transfer(s * 1024, mode="A") for s in sizes_kib]
    res_C = [simulate_transfer(s * 1024, mode="C") for s in sizes_kib]
    res_lam = {l: [simulate_transfer(s * 1024, mode="dynamic", lam=l) for s in sizes_kib] for l in lambdas}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax1.loglog(sizes_kib, [r[0] for r in res_A], "o-", color=tab10(3), label="Class A", zorder=3)
    for l, col, mk in zip(lambdas, lam_colors, lam_markers):
        ax1.loglog(sizes_kib, [r[0] for r in res_lam[l]], marker=mk, linestyle="--", color=col, label=f"Anther ($\lambda={l}$)", zorder=3)
    ax1.loglog(sizes_kib, [r[0] for r in res_C], "s-", color=tab10(0), label="Class C", zorder=3)
    ax1.set_xticks(sizes_kib)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("File Size (KiB)", fontsize=11)
    ax1.set_ylabel("Completion Time (s)", fontsize=11)
    apply_clean_style(ax1, grid_axis="y")

    ax2.loglog(sizes_kib, [r[1] for r in res_A], "o-", color=tab10(3), label="Class A", zorder=3)
    for l, col, mk in zip(lambdas, lam_colors, lam_markers):
        ax2.loglog(sizes_kib, [r[1] for r in res_lam[l]], marker=mk, linestyle="--", color=col, label=f"Anther ($\lambda={l}$)", zorder=3)
    ax2.loglog(sizes_kib, [r[1] for r in res_C], "s-", color=tab10(0), label="Class C", zorder=3)
    ax2.set_xticks(sizes_kib)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.set_xlabel("File Size (KiB)", fontsize=11)
    ax2.set_ylabel("Energy Consumption (J)", fontsize=11)
    apply_clean_style(ax2, grid_axis="y")

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.22))
    out_comp = out_dir / "comparison.pdf"
    save_figure(fig, out_comp)
    plt.close(fig)
    print(f"wrote {out_comp}")

    avg_t_per_kib = [np.mean([r[0] / s for r, s in zip(res_lam[l], sizes_kib)]) for l in lambdas]
    avg_e_per_kib = [np.mean([r[1] / s for r, s in zip(res_lam[l], sizes_kib)]) for l in lambdas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.2))
    x_pos = np.arange(len(lambdas))
    ax1.bar(x_pos, avg_t_per_kib, color=lam_colors, width=0.55, edgecolor="none", zorder=3)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(l) for l in lambdas], fontsize=11)
    ax1.set_xlabel(r"Sensitivity $\lambda$", fontsize=11)
    ax1.set_ylabel("Time / File Size (s/KiB)", fontsize=11)
    apply_clean_style(ax1, grid_axis="y")

    ax2.bar(x_pos, avg_e_per_kib, color=lam_colors, width=0.55, edgecolor="none", zorder=3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(l) for l in lambdas], fontsize=11)
    ax2.set_xlabel(r"Sensitivity $\lambda$", fontsize=11)
    ax2.set_ylabel("Energy / File Size (J/KiB)", fontsize=11)
    apply_clean_style(ax2, grid_axis="y")

    out_lams = out_dir / "compare_lambdas.pdf"
    save_figure(fig, out_lams)
    plt.close(fig)
    print(f"wrote {out_lams}")

    for l_val, fname in [(0.1, "datarate_utility_0_1.pdf"), (0.65, "datarate_utility_0_65.pdf")]:
        _, _, t_arr, e_arr, _ = res_lam[l_val][-1]
        _, _, t_a_arr, e_a_arr, _ = res_A[-1]
        _, _, t_c_arr, e_c_arr, _ = res_C[-1]

        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        ax.plot(t_a_arr / 60.0, e_a_arr, "--", color=tab10(3), linewidth=1.6, label="Static Class A", zorder=3)
        ax.plot(t_c_arr / 60.0, e_c_arr, "--", color=tab10(0), linewidth=1.6, label="Static Class C", zorder=3)
        ax.plot(t_arr / 60.0, e_arr, "-", color=tab10(2), linewidth=2.3, label=f"Anther ($\lambda={l_val}$)", zorder=3)
        ax.set_xlabel("Elapsed Time (minutes)", fontsize=11)
        ax.set_ylabel("Cumulative Energy (J)", fontsize=11)
        ax.set_title(f"50 KiB Transfer Trajectory ($\lambda={l_val}$)", fontsize=11, pad=4)
        apply_clean_style(ax, grid_axis="y")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.18))
        out_u = out_dir / fname
        save_figure(fig, out_u)
        plt.close(fig)
        print(f"wrote {out_u}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    plot_dynamic_experiments(args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
