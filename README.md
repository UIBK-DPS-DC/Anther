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

This writes nine PDFs to `figures/plots/`. The scripts use embedded measurements and model calculations; no connected hardware is needed.

To regenerate a single figure:

```sh
python scripts/plot_fig5_fuota_baseline.py
python scripts/plot_fig6_model_validation.py
```

Shared fonts, colors, and spacing are defined in `scripts/plot_style.py`.

## Hardware experiments

Firmware lives in `node/`; experiment-specific scripts and data are under `evaluation/experiment1/`, `experiment2/`, and `experiment3/`. Check the relevant scripts and device configuration before running a hardware experiment. The plotting dependencies above are not a full server or firmware setup.

Large measurement CSVs (10 MB or more) are stored with Git LFS; code, documentation, figures, and smaller files use regular Git. You do not need the large datasets to regenerate the figures. To download them, install [Git LFS](https://git-lfs.com/) and run:

```sh
git lfs install
git lfs pull
```

## License

[Apache 2.0](LICENSE).
