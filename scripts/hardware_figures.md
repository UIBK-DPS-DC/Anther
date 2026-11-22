# Hardware figures 8 and 9

Run from the repository root:

```sh
git lfs fetch origin --include='evaluation/experiment3/measurements/lambda_*/*.csv,evaluation/experiment3/measurements/fixed_classes/class_A_50KiB.csv,evaluation/experiment3/measurements/fixed_classes/class_C_50KiB.csv' --exclude=''
python scripts/plot_fig7_8_9_dynamic_traces.py
```

The reader can use fetched LFS objects without replacing the CSV pointers in the
working tree. Tracking rules are unchanged. Missing recordings cause an error;
there is no simulation fallback.

## Sources

- Anther durations: the completed `[FILE] Done` entry in each
  `evaluation/experiment3/logs/lambda_*/dynamic_*KiB.log`.
- Anther energy and trajectories: the corresponding INA226 recordings in
  `evaluation/experiment3/measurements/lambda_*/`, using a 5 V supply.
- Class A/C references: `measurements/fixed_classes/summary.csv`, with 50 KiB
  trajectories from the corresponding cropped `class_A_50KiB.csv` and
  `class_C_50KiB.csv` files. These are the earlier reference measurements also
  found in experiment 2. They are **not** paired with the different ADR-run
  durations in `logs/fixed_class_a/` and `logs/fixed_class_c/`. The repository
  does not establish them as matched-ADR baselines for the dynamic runs.

The outputs retain their existing names:

- `figures/plots/comparison.pdf` — Figure 8.
- `figures/plots/datarate_utility_0_1.pdf` and `datarate_utility_0_65.pdf` — Figure 9.
- `figures/plots/compare_lambdas.pdf` — auxiliary summary from the same measurements.
- `figures/plots/hardware_results.csv` and `hardware_provenance.json` — source paths,
  windows, sample counts, cleaning statistics, and the values plotted.

The Figure 9 PDFs are each sized for a `0.475\linewidth` subfigure inside a
single-column figure, with 5.5-point text at that final size. Their energy axes
display kJ (the source reports remain in J). `A ref.` and `C ref.` denote the
separate Class A/C reference runs described above. Lambda and payload size belong
in the LaTeX subcaptions/caption rather than repeated plot titles.

## Recording windows matter

The raw dynamic recordings include initial class-switch markers and post-transfer
reset activity. Their Arduino `micros()` timestamps have no recorded wall-clock
alignment with the application-server logs. No saved crop boundaries were found
for these runs. **By default, energy is therefore full-recording energy, not
isolated transfer energy.** The plots say so. Transfer duration still comes from
the server log. The two metrics do not have identical windows.

For publication-quality transfer-energy comparisons, identify and verify each
transfer window in the power recording and supply a JSON file:

```json
{
  "measurements/lambda_0.1/dynamic_1KiB_measurement.csv": {
    "start_s": 12.0,
    "end_s": 316.3
  }
}
```

Those numbers are an **illustration**, not verified boundaries. Windows use
absolute sensor seconds after timestamp unwrapping, not server wall time.

```sh
python scripts/plot_fig7_8_9_dynamic_traces.py --windows /path/to/verified_windows.json
```

Unspecified recordings remain full-recording plots. The script never stretches a
trace to fit a logged duration or rescales measured energy to match the paper.
Reference curves retain their original cropped windows.

## Cleaning and integration

Serial-port startup leaves stale bytes and repeated fragments from the previous
recording. For raw recordings, the reader requires 32 consecutive monotonic
samples within two seconds of the Arduino reset. It rejects malformed rows,
non-finite values, non-monotonic timestamps, implausible timestamp jumps, and
current magnitudes above 250 mA. A negative timestamp jump is treated as a 32-bit
wrap only when the implied interval is at most one second. The supplied cropped
reference files already contain an unwrapped `time (s)` column.

Energy is integrated with trapezoids over every accepted sample as
`5 * abs(current_mA) / 1000` watts. Point reduction happens only after integration;
PDF curves retain the final measured energy. Static summary energies retain the
original right-rectangle integration, while their trajectory energies use
trapezoids; both endpoint values are recorded in the results CSV.
