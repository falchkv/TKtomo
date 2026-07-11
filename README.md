# TKtomo

Pre-processing and inspection toolkit for **ptycho-tomography** reconstruction.

TKtomo assists the tomography workflow by handling pre-processing (chiefly image
alignment) and visual inspection, while delegating heavy reconstruction to
existing libraries (TomoPy, optionally ASTRA). It provides four independent,
plain-`python`-runnable desktop UIs plus the plumbing to pass images and
parameters between them.

## Status

Early scaffold. Package layout and module interfaces are in place; individual
UIs and backends are being filled in per the build order below.

## Architecture

```
tktomo/
  io/         HDF5/NeXus loading + synthetic phantom generation
  colormaps/  curated perceptually-uniform colormap registry (pyqtgraph LUTs)
  recon/      reconstruction/reprojection backends (TomoPy) behind a Protocol
  align/      alignment methods (phase-corr, TomoPy-joint, ...) behind a Protocol
  messaging/  ZeroMQ bus for live array/parameter transfer between apps
  ui/         four standalone PySide6 + pyqtgraph applications
```

Backends and alignment methods are selected through **name → class registries**,
so libraries and methods are interchangeable and the alignment UI's method
dropdown is populated automatically.

## Installation (development)

TomoPy and ASTRA are best obtained from conda-forge:

```bash
conda install -c conda-forge tomopy astra-toolbox   # optional heavy backends
pip install -e ".[ui,dev]"
```

The core library layer (`io`, `align`, `recon` interfaces, `messaging`) works
with just `pip install -e ".[dev]"`.

## Running the UIs

```bash
python -m tktomo.ui.sinogram_app
python -m tktomo.ui.tomogram_app
python -m tktomo.ui.projection_app
python -m tktomo.ui.alignment_app
python -m tktomo.ui.feature_alignment_app   # align a pair of images by manual marks
```

The feature-alignment method (labelled marks → least squares, unlabelled marks →
RANSAC) is described in [`docs/feature_alignment.md`](docs/feature_alignment.md).

## Build order

1. **Scaffold** — package + `pyproject.toml`, module interfaces (this commit).
2. `io.hdf5_loader` + synthetic Shepp–Logan phantom for testing without data.
3. `colormaps.registry` + shared `ui/common/imageview.py`.
4. Three viewer apps (sinogram, projection, tomogram + reproject).
5. `align` methods (phase-correlation, TomoPy-joint) + the alignment app.
6. `messaging.bus` (ZeroMQ) wiring alignment → sinogram live update.
