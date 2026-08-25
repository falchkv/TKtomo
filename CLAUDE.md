# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TKtomo is a pre-processing and inspection toolkit for **ptycho-tomography** reconstruction. It assumes ptycho projections already exist and focuses on pre-processing (chiefly image alignment) and visual inspection, delegating heavy reconstruction to external libraries (TomoPy, optionally ASTRA). It ships **four independent, plain-`python`-runnable PySide6 desktop UIs** plus a library layer that passes images and parameters between them.

## Commands

```bash
# Core library + tests only (no GUI, no heavy reconstruction stack):
pip install -e ".[dev]"
# To also run the UIs (PySide6 + pyqtgraph + colormap libs):
pip install -e ".[ui,dev]"
# Reconstruction backends (best via conda-forge, not pip):
conda install -c conda-forge tomopy astra-toolbox

python -m pytest                                   # full suite
python -m pytest tests/test_io.py -q               # one file
python -m pytest tests/test_io.py::test_phantom_shapes   # one test

# Run a UI (each is standalone; sinogram/projection accept an optional .h5 path):
python -m tktomo.ui.sinogram_app [file.h5]
python -m tktomo.ui.tomogram_app
python -m tktomo.ui.projection_app
python -m tktomo.ui.alignment_app
# Equivalent installed console scripts: tktomo-sinogram, -tomogram, -projection, -alignment

# Blender X-ray simulator (needs a Python with bpy — see docs/blender-sim_instructions.md):
python examples/launch_blender.py             # open Blender GUI: demo scene + sidebar panel
blender --background --python tktomo/blender_sim/cli.py -- --output out.h5
python -m tktomo.blender_sim --output out.h5   # with `pip install -e ".[blender]"`
```

GUI tests use `pytest-qt`. Lint/format with ruff (the `# noqa` codes in source follow ruff conventions):

```bash
ruff check .        # lint
ruff check --fix .  # lint and autofix
ruff format .       # format
```

Note: ruff is not currently declared as a dependency or configured in `pyproject.toml` — `pip install ruff` to use it.

## Architecture

### Dependency layering (deliberate — preserve it)
The package is split so the **core library imports and unit-tests run in a light environment**: `io`, `align`, `recon` interfaces, and `messaging` depend only on the core deps in `pyproject.toml`. The GUI stack (`PySide6`, `pyqtgraph`, colormap libs) is the optional `ui` extra; TomoPy/pystackreg are the optional `recon` extra.

To keep this true, **heavy/optional dependencies are imported lazily inside functions**, never at module top level — skimage, scipy, zmq, msgpack, matplotlib/colorcet/cmocean/cmasher, and pyqtgraph are all imported at call time. Follow this pattern when adding code: a bare `import tktomo.io` or `import tktomo.align` must not pull in the GUI or reconstruction stack.

### Name→instance registries + Protocols (the modularity mechanism)
Interchangeable methods/backends are selected by string name through registries, and the UIs enumerate the registry to build their dropdowns — so adding a method makes it appear in the UI with no UI changes:

- **Aligners** (`tktomo/align/base.py`): `register_aligner` / `get_aligner` / `available_aligners`. `Aligner` is a `runtime_checkable` Protocol with `name` and `estimate(fixed, moving, initial) -> Transform`. Importing `tktomo.align` triggers a **side-effect import of `tktomo.align.methods`** which registers the built-ins (phase-correlation, StackReg). The alignment UI's method dropdown is populated from `available_aligners()`.
- **Recon backends** (`tktomo/recon/backend.py`): same pattern (`register_backend` / `get_backend`, default `"tomopy"`). `ReconBackend` Protocol has `reconstruct(...)` and `reproject(...)`, used by the tomogram UI's "reproject" button and by joint alignment.
- **Colormaps** (`tktomo/colormaps/registry.py`): a curated `display_name -> (source_lib, source_name)` table; only maps whose source lib is installed are advertised by `available_colormaps()`, so the dropdown never lists a map it cannot build. Grayscale is always available as a dependency-free fallback.

### Central data type
`ProjectionData` (`tktomo/io/data.py`) is the single container passed everywhere: `data` shape **`(n_angles, height, width)`**, `angles` in **radians** (TomoPy's convention — no conversion needed at the backend), plus free-form `metadata`. Helpers `sinogram(row) -> (n_angles, width)` and `projection(index) -> (height, width)`. The sinogram viewer reshapes the stack for scrolling via `np.moveaxis(data, 1, 0)` so detector **row** becomes the scroll axis.

### Transforms and alignment
`Transform` (`tktomo/align/transform.py`) is a frozen rigid-2D transform `(dx, dy, rotation°)`; `dx` is columns/width, `dy` is rows/height, rotation is CCW about the image centre, applied via `scipy.ndimage.affine_transform`. `TransformHistory` is the undo stack backing the alignment UI's undo button / Ctrl+Z. `apply_projection_transform` (`tktomo/align/apply.py`) is kept **Qt-free and array-based** so the sinogram viewer can re-warp one projection in its owned stack and re-derive the affected sinograms.

### Cross-app messaging (ZeroMQ PUB/SUB)
`tktomo/messaging/bus.py` is a thin ZeroMQ bus. A `Message` carries a `topic`, a JSON-friendly `params` dict, and named numpy `arrays`; the wire format is multipart (topic / msgpack header / raw array buffers, zero-copy). The **alignment app is the Publisher**: "Send to sinogram" publishes a `Transform.as_dict()` plus a `projection_index` on `TOPIC_ALIGNMENT`. The **sinogram app is the Subscriber**: it polls the bus non-blocking from a Qt `QTimer` (to keep the UI responsive), and on receipt either swaps in a pushed `sinogram` array or applies the transform to its projection stack via `apply_projection_transform`. Both `zmq` and `msgpack` are imported lazily, so importing `tktomo.messaging` never requires them.

### UIs
The apps live in `tktomo/ui/*_app.py` and share `tktomo/ui/common/`: `SliceViewer` (a pyqtgraph `ImageView` that scrolls along axis 0 with a colormap dropdown) and `run_app` (creates/reuses a `QApplication`, shows the window, runs the loop). When no data file is given, apps fall back to synthetic data from `tktomo/io/phantom.py` (`generate_phantom` builds a projection stack; `generate_volume` builds a 3D volume) so every UI runs with no acquisitions present. Beyond the four viewers/aligner, `feature_alignment_app` aligns a pair of images by hand-placed marks (labelled → least-squares Procrustes, unlabelled → RANSAC); its Qt-free geometry lives in `tktomo/align/feature.py` and is documented in `docs/feature_alignment.md`. The manual feature-tracking pair `feature_isolation_app` (hand-track one feature, export a moving crop with coordinate provenance) and `track_model_app` (label features, fit rotation axis/tilt-drift/per-view-shift model with fixed-free masks, residual plots, live gridrec slice, File menu with session save/load and exports incl. ASTRA `parallel3d_vec`) keeps ALL math and file formats Qt-free in `tktomo/tracking/` (coords/labels/model/diagnostics/stackio/export/sessionio); specs in `docs/feature_tracking_instructions.md`, user docs `docs/feature_isolation.md` + `docs/track_model.md`. The fit runs in raw detector coordinates through `tracking.coords.CoordinateChain`; the gauge handling in `tracking/model.py` is load-bearing, read its docstring before touching the solver.

The projection app also drives reconstruction: "Load projections…" reads HDF5 stacks (NXtomo/DXchange layouts or the blender-sim per-output groups — `tktomo.io.list_projection_groups` finds them, with a group picker when a file holds several outputs), and "Reconstruct →" runs the selected recon backend (algorithm/filter/iterations/centre controls; filter applies to gridrec/fbp, iterations to the iterative algorithms), writes the tomogram to HDF5 via `tktomo.io.save_volume` (`/tomogram/data` + `angles` + provenance attrs), and opens it in a `TomogramWindow` (which accepts the scan `angles` so "Reproject" uses the real geometry).

### Blender X-ray simulator
`tktomo/blender_sim/` simulates X-ray tomography projections from a Blender scene (spec: `docs/blender-sim_instructions.md`). Same layering discipline: `materials` (δ/β/μ/E edit rules), `propagation` (propagator registry: fresnel / angular_spectrum / fraunhofer / fresnel_scaling) and `multislice` (exit wave + beam-propagation loop) are **numpy-only**; `scene.py` is the sole `bpy` consumer with `bpy`/`mathutils` imported lazily inside functions, so importing anything under `tktomo.blender_sim` never requires Blender. The camera and source are fixed; the sample rotates via a `sample_root` Empty (bodies tagged with `xray_delta`/`xray_beta` custom properties — include meshes via the "X-ray Sim" sidebar panel (`ui_panel.register()`), `scene.add_selected(delta=…, beta=…)`, or `scene.add_body(…)`, which auto-creates the root and preserves transforms). `runner.simulate()` loops orientations and returns `ProjectionData` per requested output; `cli.py` runs headless (`blender --background --python tktomo/blender_sim/cli.py -- …`), under pip `bpy`, or in a live Blender session, building a demo scene when none is loaded. Blender scene coordinates are **millimetres** (`scene.UNIT_SCALE = 1e-3`); the physics APIs (line integrals, pixel pitch, propagation distances) are SI metres with conversion inside the scene layer; photon energy keV.
