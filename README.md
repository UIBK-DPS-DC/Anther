# Anther

Code and experiments for delivering model updates over LoRaWAN. Anther balances transfer time and device energy by choosing between LoRaWAN operating classes as link conditions change.

## Repository

- [`applicationserver/`](applicationserver/) — application server and transfer estimation models.
- [`node/`](node/) — end-device firmware and a Heltec V3 FUOTA baseline.
- [`evaluation/`](evaluation/) — compression benchmarks, transfer measurements, and experiment logs.
- [`scripts/`](scripts/) — analytical models and figure-generation scripts.
- [`figures/plots/`](figures/plots/) — generated PDF figures.

## Reproduce the figures

From the repository root, using Python 3.10 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib
python scripts/generate_all_figures.py
```

This writes nine PDFs to `figures/plots/`. No connected hardware is needed. Figures 8–9 require the recorded hardware datasets; see [hardware figure sources and setup](scripts/hardware_figures.md) for the selective Git LFS fetch command and measurement-window limitations.

To regenerate a single figure:

```sh
python scripts/plot_fig5_fuota_baseline.py
python scripts/plot_fig6_model_validation.py
```

Shared fonts, colors, and spacing are defined in `scripts/plot_style.py`. PDFs use 5.5-point text at their intended IEEE column widths; see [LaTeX sizing](scripts/figure_layout.md) for placement and exact template dimensions.

## Hardware experiments

Firmware lives in `node/`; experiment-specific scripts and data are under `evaluation/experiment1/`, `experiment2/`, and `experiment3/`. Check the relevant scripts and device configuration before running a hardware experiment. The plotting dependencies above are not a full server or firmware setup.

Large measurement CSVs (10 MB or more) are stored with Git LFS; code, documentation, figures, and smaller files use regular Git. To download them, install [Git LFS](https://git-lfs.com/) and run:

```sh
git lfs install
git lfs pull
```

## License

[Apache 2.0](LICENSE).
