import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_fig3_datarate_schedule import plot_datarate_schedule
from plot_fig4_compression_scatter import plot_scatter
from plot_fig5_fuota_baseline import plot_class_c_vs_fuota
from plot_fig6_model_validation import plot_estimation_vs_measurement
from plot_fig7_8_9_dynamic_traces import plot_dynamic_experiments
from plot_fig10_decision_boundaries import plot_decision_boundaries
from fast_lora_adapter import evaluate_fast_lora_network


def main():
    plot_datarate_schedule()
    plot_scatter()
    plot_class_c_vs_fuota()
    plot_estimation_vs_measurement()
    plot_dynamic_experiments()
    plot_decision_boundaries()

    fl_res = evaluate_fast_lora_network()
    print(
        f"FAST-LoRa check (N=50, K=1): mean_PDR={fl_res['mean_pdr']:.4f}, "
        f"mean_EE={fl_res['mean_ee_bits_per_j']:.2f} bits/J, "
        f"mean_RSS={fl_res['mean_rss_dbm']:.1f} dBm"
    )
    print("All figures regenerated cleanly in figures/plots/ as 300-DPI vector PDFs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
