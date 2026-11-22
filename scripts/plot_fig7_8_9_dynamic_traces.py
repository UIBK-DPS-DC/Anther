"""Figures 8–9 from hardware logs and INA226 recordings, never simulations.

See hardware_figures.md for source selection, cleaning, and window semantics.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from hardware_data import ROOT_DIR, read_power_trace, read_transfer_log, read_windows
from plot_style import apply_clean_style, plt, save_figure, tab10

DEFAULT_FIG_DIR = ROOT_DIR / "figures" / "plots"
DEFAULT_DATA_DIR = ROOT_DIR / "evaluation" / "experiment3"
SIZES_KIB = (1, 2, 5, 10, 50)
LAMBDAS = (0.1, 0.4, 0.65, 0.7)
LAMBDA_COLORS = (tab10(2), tab10(9), tab10(1), tab10(4))
LAMBDA_MARKERS = ("^", "v", "D", "P")


def _relative(path):
    return str(path.resolve().relative_to(ROOT_DIR)) if path.resolve().is_relative_to(ROOT_DIR) else str(path)


def load_hardware_results(data_dir, windows=None, voltage=5.0):
    data_dir = Path(data_dir)
    windows = windows or {}
    results = []
    traces = {}
    for lam in LAMBDAS:
        for size in SIZES_KIB:
            log = data_dir / "logs" / f"lambda_{lam}" / f"dynamic_{size}KiB.log"
            measurement = data_dir / "measurements" / f"lambda_{lam}" / f"dynamic_{size}KiB_measurement.csv"
            key = str(measurement.relative_to(data_dir))
            window = windows.get(key)
            transfer = read_transfer_log(log)
            if transfer["payload_bytes"] != size * 1024 or transfer["lambda"] != lam:
                raise ValueError(f"Log does not match the requested experiment: {log}")
            trace = read_power_trace(measurement, voltage=voltage, window=window)
            capture_duration = trace.end_s - trace.start_s
            if window is None and capture_duration + 2 < transfer["transfer_duration_s"]:
                raise ValueError(f"Power recording is shorter than the logged transfer: {measurement}")
            result = {
                "series": f"Anther (lambda={lam})",
                "lambda": lam,
                "size_kib": size,
                "transfer_duration_s": transfer["transfer_duration_s"],
                "energy_j": float(trace.energy_j[-1]),
                "energy_scope": "explicit_window" if window else "full_recording",
                "sensor_start_s": trace.start_s,
                "sensor_end_s": trace.end_s,
                "energy_window_duration_s": capture_duration,
                "samples": trace.samples,
                "rejected_rows": trace.rejected_rows,
                "timestamp_wraps": trace.timestamp_wraps,
                "voltage_v": voltage,
                "class_sequence": " -> ".join(transfer["class_sequence"]),
                "log": _relative(log),
                "measurement": _relative(measurement),
            }
            results.append(result)
            if size == 50:
                traces[f"lambda_{lam}"] = trace
            print(f"lambda={lam:<4} {size:2} KiB: log={transfer['transfer_duration_s']:.2f} s, "
                  f"recording={capture_duration:.2f} s, energy={trace.energy_j[-1]:.3f} J "
                  f"({result['energy_scope']}; rejected {trace.rejected_rows} rows)", flush=True)

    # These are the existing cropped reference measurements, not the ADR runs in
    # logs/fixed_class_*. Never pair their energy with those other runs' durations.
    summary = data_dir / "measurements" / "fixed_classes" / "summary.csv"
    with summary.open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = Path(row["file"]).name
            parts = name.removesuffix(".csv").split("_")
            cls, size = parts[1], int(parts[2].removesuffix("KiB"))
            results.append({
                "series": f"Class {cls} (reference)", "lambda": "", "size_kib": size,
                "transfer_duration_s": float(row["duration_s"]),
                "energy_j": float(row["energy_J"]), "energy_scope": "cropped_reference",
                "sensor_start_s": float(row["start_time_s"]),
                "sensor_end_s": float(row["end_time_s"]),
                "energy_window_duration_s": float(row["duration_s"]),
                "samples": int(row["samples"]), "rejected_rows": "",
                "timestamp_wraps": "", "voltage_v": 5.0, "class_sequence": cls,
                "log": "", "measurement": _relative(summary),
            })
            if size == 50:
                trace_path = summary.parent / name
                traces[cls] = read_power_trace(trace_path, voltage=5.0)
                # The original summary uses right-rectangle integration; the
                # trajectory uses trapezoids. Keep the distinction auditable.
                results[-1]["trajectory_energy_j"] = float(traces[cls].energy_j[-1])
    for cls in ("A", "C"):
        rows = [r for r in results if r["series"] == f"Class {cls} (reference)"]
        if sorted(r["size_kib"] for r in rows) != list(SIZES_KIB):
            raise ValueError(f"Incomplete or duplicated Class {cls} reference measurements")
    return results, traces


def _legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0))


def plot_dynamic_experiments(out_dir=None, data_dir=None, windows_path=None, voltage=5.0):
    out_dir = Path(out_dir) if out_dir else DEFAULT_FIG_DIR
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    results, traces = load_hardware_results(data_dir, read_windows(windows_path), voltage)
    dynamic = [r for r in results if r["lambda"] != ""]
    cropped = all(r["energy_scope"] == "explicit_window" for r in dynamic)
    any_cropped = any(r["energy_scope"] == "explicit_window" for r in dynamic)
    scope_note = "Anther: mixed energy windows" if any_cropped else "Anther: setup/reset included"

    series = [
        ("Class A (reference)", "Class A (reference)", tab10(3), "o", "-"),
        *[(f"Anther (lambda={lam})", rf"Anther ($\lambda={lam}$)", col, marker, "--")
          for lam, col, marker in zip(LAMBDAS, LAMBDA_COLORS, LAMBDA_MARKERS)],
        ("Class C (reference)", "Class C (reference)", tab10(0), "s", "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8))
    for ax, metric, ylabel in zip(
        axes, ("transfer_duration_s", "energy_j"),
        ("Completion Time (s)", "Energy (J)"),
    ):
        for key, label, color, marker, linestyle in series:
            rows = sorted((r for r in results if r["series"] == key), key=lambda r: r["size_kib"])
            ax.loglog([r["size_kib"] for r in rows], [r[metric] for r in rows],
                      marker=marker, linestyle=linestyle, color=color, label=label, zorder=3)
        ax.set_xticks(SIZES_KIB)
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("File Size (KiB)")
        ax.set_ylabel(ylabel)
        apply_clean_style(ax)
    if not cropped:
        axes[1].text(0.03, 0.97, scope_note, transform=axes[1].transAxes,
                     va="top", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    _legend(fig, axes[0])
    save_figure(fig, out_dir / "comparison.pdf")
    plt.close(fig)

    # Preserve the existing auxiliary output, now based only on hardware data.
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for ax, metric, ylabel in zip(
        axes, ("transfer_duration_s", "energy_j"),
        ("Time / File Size (s/KiB)", "Energy / File Size (J/KiB)"),
    ):
        means = [np.mean([r[metric] / r["size_kib"] for r in dynamic if r["lambda"] == lam])
                 for lam in LAMBDAS]
        ax.bar(np.arange(len(LAMBDAS)), means, color=LAMBDA_COLORS, width=0.55, zorder=3)
        ax.set_xticks(np.arange(len(LAMBDAS)), labels=[str(lam) for lam in LAMBDAS])
        ax.set_xlabel(r"Sensitivity $\lambda$")
        ax.set_ylabel(ylabel)
        apply_clean_style(ax)
    if not cropped:
        axes[1].text(0.03, 0.97, scope_note, transform=axes[1].transAxes,
                     va="top", bbox={"facecolor": "white", "edgecolor": "none"})
    save_figure(fig, out_dir / "compare_lambdas.pdf")
    plt.close(fig)

    for lam, filename in ((0.1, "datarate_utility_0_1.pdf"), (0.65, "datarate_utility_0_65.pdf")):
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        for key, label, color, linestyle in (
            ("A", "A ref.", tab10(3), "--"),
            ("C", "C ref.", tab10(0), "--"),
            (f"lambda_{lam}", "Anther", tab10(2), "-"),
        ):
            trace = traces[key]
            # Integrate every sample, but don't embed tens of thousands of PDF
            # vertices. Keep the measured endpoint exactly; never rescale it.
            indices = np.unique(np.linspace(0, len(trace.time_s) - 1, min(4000, len(trace.time_s))).astype(int))
            # Display kJ to keep tick labels narrow at 0.475 column width.
            ax.plot(trace.time_s[indices] / 60, trace.energy_j[indices] / 1000, color=color,
                    linestyle=linestyle, linewidth=2 if key.startswith("lambda") else 1.6,
                    label=label, zorder=3)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Energy (kJ)")
        ax.set_yticks([0, 1, 2])
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3, min_n_ticks=2))
        row = next(r for r in dynamic if r["lambda"] == lam and r["size_kib"] == 50)
        if row["energy_scope"] == "full_recording":
            ax.text(0.03, 0.97, "Setup/reset included",
                    transform=ax.transAxes, va="top", fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        apply_clean_style(ax)
        # The LaTeX subcaptions already supply lambda and the transfer size.
        # A small in-axes legend avoids repeating a tall footer in each panel.
        ax.legend(loc="upper right", frameon=False, handlelength=0.9,
                  handletextpad=0.35, labelspacing=0.15, borderaxespad=0.15)
        save_figure(fig, out_dir / filename, column_fraction=0.475)
        plt.close(fig)

    fields = list(dict.fromkeys(key for result in results for key in result))
    with (out_dir / "hardware_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    provenance = {
        "data_root": _relative(data_dir), "voltage_v": voltage,
        "windows_file": _relative(Path(windows_path)) if windows_path else None,
        "dynamic_energy_scope": "explicit_windows" if cropped else "full_recordings_or_mixed",
        "integration": "Trapezoidal V * abs(current_mA) / 1000 at accepted sample timestamps",
        "reference_energy": "Existing cropped fixed_classes/summary.csv (right-rectangle integration)",
        "limitations": [
            "Raw sensor timestamps have no wall-clock synchronization with server logs.",
            "Without explicit windows, dynamic recording energy includes setup/reset activity.",
            "Reference measurements are separate runs, not the ADR runs in logs/fixed_class_*.",
            "No simulated values, duration normalization, or energy rescaling are used.",
        ],
        "results": results,
    }
    (out_dir / "hardware_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote hardware figures and provenance to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--windows", type=Path, help="JSON measurement windows, in unwrapped sensor seconds")
    parser.add_argument("--voltage", type=float, default=5.0)
    args = parser.parse_args()
    plot_dynamic_experiments(args.output_dir, args.data_dir, args.windows, args.voltage)


if __name__ == "__main__":
    main()
