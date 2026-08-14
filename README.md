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
python -m tktomo.ui.feature_isolation_app   # hand-track one feature, export a moving crop
python -m tktomo.ui.track_model_app         # fit a tomography model from manual labels
ptycho-align [projections.h5]               # interactive reprojection alignment
```

The feature-alignment method (labelled marks → least squares, unlabelled marks →
RANSAC) is described in [`docs/feature_alignment.md`](docs/feature_alignment.md).
The manual feature-tracking pair (`tktomo-feature-isolation`,
`tktomo-track-model`: label features, fit rotation axis + tilt drift +
per-view shifts, export shifts/ASTRA geometry/aligned stack) is documented in
[`docs/feature_isolation.md`](docs/feature_isolation.md) and
[`docs/track_model.md`](docs/track_model.md).

### `ptycho-align` — interactive reprojection alignment

`tktomo.ptycho_align` drives the iterative reprojection alignment workflow for
ptychographic tomography (Gürsoy et al., *Sci. Rep.* **7**, 11818, 2017): load phase
projections → preprocess (phase-ramp removal) → centre-of-mass pre-alignment →
reconstruct / reproject / register, **one inspectable iteration at a time**.

TomoPy's own `align_joint`/`align_seq` run their loop internally with no way to pause,
so the loop is re-implemented on TomoPy's primitives and exposed one iteration at a
time. `tktomo.ptycho_align.core` is headless and scriptable; the PySide6 GUI is a thin
shell over it, with all computation on a `QThread`.

```bash
python examples/make_phantom.py --output phantom.h5 --noise 0.02
ptycho-align phantom.h5
```

See [`docs/ptycho_align.md`](docs/ptycho_align.md) — including the sign conventions
(the single most likely source of a diverging alignment) and a troubleshooting guide.

## Build order

1. **Scaffold** — package + `pyproject.toml`, module interfaces (this commit).
2. `io.hdf5_loader` + synthetic Shepp–Logan phantom for testing without data.
3. `colormaps.registry` + shared `ui/common/imageview.py`.
4. Three viewer apps (sinogram, projection, tomogram + reproject).
5. `align` methods (phase-correlation, TomoPy-joint) + the alignment app.
6. `messaging.bus` (ZeroMQ) wiring alignment → sinogram live update.
