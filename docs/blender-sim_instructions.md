# Blender X-ray tomography simulation

I want to use Blender to simulate X-ray tomography projections, defining per body the
X-ray refractive index (n = 1 − δ + iβ) of the different bodies in the scene.

The result is a Python **package** (`tktomo/blender_sim/`, implemented) with the
features below — not a single script pasted into Blender's scripting window.

## Deliverable: package layout and run modes

- **Physics layer (numpy-only).** Propagators, the multislice loop, and the
  material/edit rules are pure numpy — importable and unit-testable with no Blender
  installed, matching the repo's light-core layering.
- **Scene/geometry layer (lazy bpy).** Only this layer touches `bpy`, imported
  **lazily inside functions** (the repo's existing lazy-import pattern). It handles
  scene definition and extraction of the per-slab projected δ/μ maps.
- **Run modes** — the same code runs three ways, all editable from a normal IDE
  rather than Blender's scripting window:
  1. headless CLI: `blender --background --python tktomo/blender_sim/cli.py --
     --output out.h5` (Blender's bundled Python ships numpy);
  2. `pip install bpy` (the `[blender]` extra; note each bpy release pins an exact
     Python version): `python -m tktomo.blender_sim --output out.h5` or the
     `tktomo-blender-sim` console script;
  3. a live Blender session (scripting window or the Blender MCP addon) for
     interactive scene building and inspection.
- **Output** remains `ProjectionData`, so results plug directly into the existing
  viewers.
- **Units**: **Blender coordinates are millimetres** — one Blender unit = 1 mm
  (`scene.UNIT_SCALE = 1e-3` metres), so samples are modelled at a comfortable
  viewport scale; the panel's length fields (distance, Δz, r1) are mm too. The
  **physics-layer APIs stay SI**: line integrals, pixel pitch, propagation
  distances and μ are metres / 1/m (conversion happens inside the scene layer),
  photon energy in keV.

### Module map (as implemented)

| Module | Layer | Contents |
| --- | --- | --- |
| `materials.py` | numpy-only | `Material`, `MaterialSet` (δ/β/μ/E edit rules), `wavelength`, `mu_from_beta`, … |
| `propagation.py` | numpy-only | propagator registry: `fresnel`, `angular_spectrum`, `fraunhofer`, `fresnel_scaling` |
| `multislice.py` | numpy-only | exit wave, multislice loop, `detector_wave`, output extraction |
| `runner.py` | numpy-only API, bpy at call time | `normalize_orientations`, `simulate()` → `ProjectionData` per output |
| `scene.py` | lazy bpy | `setup_scene`, `add_body`/`add_selected`/`remove_body`/`bodies`, `set_sample_orientation`, `extract_slab_integrals`, `build_demo_scene` |
| `ui_panel.py` | lazy bpy | "X-ray Sim" viewport sidebar panel (`register()`): body tagging + Projection section ("Project now" button, "Live update" toggle) |
| `viewer.py` | lazy bpy | current-pose projection viewer: `show_projection()` into an Image Editor (auto-splits the viewport), depsgraph-driven live update |
| `cli.py` / `__main__.py` | entry point | argparse CLI, HDF5 output, demo-scene fallback |

`examples/demo_projection.py` is a runnable demo: it projects the current scene
(building the demo cube + sphere if nothing is tagged), auto-fits the detector to
the sample bounds, and displays the attenuation map in a Blender Image Editor
(saves a PNG in `--background` mode).

`examples/launch_blender.py` is the session starter: `python
examples/launch_blender.py [--beam cone]` opens the Blender GUI with the demo
scene built, the "X-ray Sim" sidebar panel registered, and a viewer column opened
next to the viewport — the projection image on top, a camera's-eye 3D view (what
the X-ray camera sees) below it (nothing persists in Blender between sessions, so
run this after every restart). It keeps already
tagged bodies when given an existing `.blend`
(`blender my_scene.blend --python examples/launch_blender.py`), and finds Blender
via `--blender`, `$BLENDER`, or PATH.

## Acquisition geometry: rotate the sample, keep the camera fixed

Real lab/synchrotron CT keeps the source and detector fixed and rotates the sample on
a stage. This is mathematically equivalent to orbiting the camera (a sample
orientation `R` equals a camera orientation `R⁻¹`), but rotating the sample is
preferable here:

- It matches TomoPy's convention, where `angles` are the **sample** rotation angles,
  so projections map 1:1 onto `ProjectionData(data=(n_angles, height, width),
  angles=radians)` with no conversion.
- A fixed camera guarantees a **constant detector frame** (identical pixel grid, up
  vector, and roll) for every projection, eliminating the in-plane roll drift an
  orbiting camera would introduce.

Concretely:

- **Fixed camera = the detector, and it defines the ray geometry.** The camera
  points at the world origin at a fixed distance and does not move between
  projections. **Cone beam vs parallel beam is defined by the camera
  type/parameters** (perspective = cone, orthographic = parallel), and the
  projection's **field of view and pixel grid match a render from that camera
  exactly**: orthographic FOV = `ortho_scale`; perspective FOV from the focal
  length/sensor (`camera.data.angle`), with the cone apex at the **camera
  pinhole** and the pixel grid sampled on the sample plane through the origin
  (pitch = `scene.camera_pixel_size(detector_shape)`). Assumes zero lens shift
  and square pixel aspect. Because the beam direction is fixed relative to the
  detector, the beam frame is constant across all projections; physically the
  same ray set describes a source on the far side of the sample shining toward
  the detector.
- **Rotate the sample.** All sample bodies are parented to a single **Empty at the
  world origin**; projections are generated by setting the Empty's orientation, so the
  bodies rotate as one rigid group about the common origin.
- **Orientation array (arbitrary).** The input is an array of **sample orientations** —
  arbitrary 3D poses, not just a single axis. Single-axis tomography is the subset
  where all orientations share one axis; arbitrary poses cover laminography /
  tilted-axis geometries. Accepted forms (`runner.normalize_orientations`): shape
  `(n,)` — radians about the vertical (Z) tomo axis, passed through as
  `ProjectionData.angles`; `(n, 3)` — tilted-axis angles `(x_tilt, y_tilt, scan)`
  in radians, applied Z-first (R = Rx·Ry·Rz): **x/y tilt the scan axis, z rotates
  about that tilted axis**, and the scan column becomes `ProjectionData.angles`;
  `(n, 3, 3)` — rotation matrices (`angles` zeros, poses in metadata). The
  `sample_root` Empty defaults to Blender's matching `ZYX` rotation mode, so
  hand-editing its X/Y/Z rotation in the UI behaves the same way.

## Adding sample bodies

Any closed mesh becomes part of the simulation once it is tagged and parented to
``sample_root`` — three equivalent ways, most convenient first:

1. **Sidebar panel (no code).** Run once:

       from tktomo.blender_sim import ui_panel
       ui_panel.register()

   An **"X-ray Sim"** tab appears in the 3D-viewport sidebar (``N``): set default
   δ/β, select meshes, click *Include selected*. The active body's δ/β stay
   editable there, *Exclude active* removes it, and the panel lists everything
   currently included. Its **Projection** section shows the current-pose
   projection in an Image Editor — *Project now* for a manual update, *Live
   update* to re-project automatically whenever the sample, its pose, or the
   camera changes; panel-setting edits also re-project live. Fields: energy,
   resolution, **output** (attenuation / phase / intensity — enabling Propagate
   switches to intensity, what a detector sees) and an optional **Propagate**
   stage with the **method** dropdown enumerated
   from the propagator registry (fresnel, angular_spectrum, fraunhofer,
   fresnel_scaling), detector distance, multislice Δz, and r1 for
   fresnel_scaling.

2. **One call for the selection** (scripting window / MCP):

       from tktomo.blender_sim import scene
       scene.add_selected(delta=1e-6, beta=1e-9)

3. **Explicit, per object** — accepts an object or its name, a ``Material``, bare
   ``delta``/``beta``, or ``delta`` + ``mu`` with ``energy_kev`` (β derived):

       scene.add_body("MyMesh", delta=1e-6, mu=250.0, energy_kev=17.0)

The ``sample_root`` Empty is created on demand, world transforms are preserved
(bodies stay where they were modelled, even mid-scan), and nested children are
found. Tags are the ``xray_delta``/``xray_beta`` custom properties, so values are
also editable in Object Properties → Custom Properties. ``scene.remove_body(obj)``
excludes a body again; ``scene.bodies()`` lists what is included.

## Material parameters: δ, β, photon energy E, and editable μ

Each body carries both **δ** (real decrement) and **β** (imaginary part) of the
complex refractive index n = 1 − δ + iβ. A single **scene-level photon energy E**
converts β to the linear attenuation coefficient μ that drives Beer–Lambert absorption
contrast in the render:

    μ = 4πβ / λ,   λ = hc / E,   hc ≈ 1.23984 keV·nm    ⟹    μ = 4πβE / (hc)

μ is exposed as an editable value. The parameters are linked by these edit rules:

- editing **β** recomputes μ (E and δ held fixed);
- editing **μ** recomputes **β**, *not* E — β = μλ/(4π) = μ·hc/(4πE) at fixed E
  (δ unchanged);
- editing **δ** affects δ only (no effect on β or μ);
- editing **E** rescales the material's optical constants by the standard
  far-from-edge dispersion laws: **δ ∝ 1/E²** (δ_new = δ·(E_old/E_new)²) and
  **β ∝ 1/E³** (β_new = β·(E_old/E_new)³); μ is then recomputed from the new β and E,
  so it follows μ ∝ 1/E².

δ is retained per body as the phase decrement (~1e-6); X-ray ray bending is negligible,
so a straight-ray line-integral transmission model is used for image formation.

## Wave formation and optional propagation

For each detector pixel the simulator forms the complex **exit wave** through the
sample under the projection approximation:

    ψ_exit = exp(−½·∫μ dz − i·φ),   φ = (2π/λ)·∫δ dz,   ∫μ dz = (4π/λ)·∫β dz

(The minus sign on φ follows the `e^{+ikz}` carrier convention of the propagators
and the slab transmission below: n = 1 − δ, so δ *reduces* the optical path.) The
magnitude carries the **absorbance** ∫μ dz (μ from β, per the material rules above)
and the argument carries the **accumulated phase** φ (from δ). Both are ray line
integrals through the sample along the fixed beam direction, so |ψ| = exp(−½·∫μ dz)
and φ = −arg ψ.

**Optional multislice propagation.** When enabled, the wave is transported to the
detector plane by the **multislice / beam-propagation method** instead of being read
off at the sample. The sample's extent along the fixed beam axis is divided into slabs
whose thickness is set by a **slice-spacing Δz** parameter — *not* a slice count.
Because the sample rotates against a fixed camera, its extent along the beam varies per
projection; fixing Δz (so the slice count `ceil(extent/Δz)` adapts while the spacing
stays constant) keeps the propagation physics consistent across all orientations. For
each slab, from its projected δ and μ:

    transmit:   ψ ← ψ · exp(−i·k·δ·Δz − k·β·Δz)     (k = 2π/λ)
    propagate:  ψ ← NearField_Δz{ ψ }               (to next slab)

The slab-to-slab step always uses a **near-field** propagator (Fresnel or
angular-spectrum, see below). After the last slab, a single free-space propagation
over the remaining **propagation distance z** carries the wave to the detector, giving
ψ_det — and *this* final hop uses the method chosen by the propagation-method selector.
A Δz at least as large as the sample extent collapses to a single slab (the projection
approximation); with propagation disabled, ψ_det = ψ_exit (the contact plane).

## Propagation methods

A **scene-level propagation-method selector** picks how the exit wave is carried to the
detector (the final hop above). The multislice slab-to-slab steps always use a
near-field kernel (Fresnel or angular-spectrum); when the selected method is itself
near-field, the same kernel is reused for the slab steps.

- **fresnel** — paraxial near-field. Transfer function `H = exp(i·k·z)·exp(−iπλz(f_x²+f_y²))`;
  `ψ_z = 𝔽⁻¹{ 𝔽{ψ}·H }`. Fast and accurate for small Fresnel numbers; the default.
- **angular_spectrum** (near-field spectral method) — the exact, non-paraxial
  transfer function `H = exp(i·k·z·√(1 − (λf_x)² − (λf_y)²))` (evanescent components
  band-limited/zeroed); `ψ_z = 𝔽⁻¹{ 𝔽{ψ}·H }`. Valid at wider angles than Fresnel.
- **fraunhofer** — far-field, single FFT. `ψ_z ∝ 𝔽{ψ}` with the usual leading
  quadratic phase and `1/(λz)` detector-scale factor; valid only for large z (small
  far-field Fresnel number). Single-stage — not used as a slab step.
- **fresnel_scaling** — cone-beam **effective propagation** via the Fresnel scaling
  theorem. From the cone geometry (R₁ = source→sample, R₂ = sample→detector) it forms
  the magnification **M = (R₁+R₂)/R₁** and the **effective distance
  z_eff = R₂/M = R₁R₂/(R₁+R₂)**, runs a near-field parallel-beam propagator (Fresnel or
  angular-spectrum) at z_eff, then scales the result onto the detector by M (intensity
  ×1/M²). This is selected explicitly and expects cone geometry parameters; pair it
  with a perspective (cone) camera for physical consistency.

## Output

A **scene-level output selector** chooses which quantity(ies) each run returns; any
combination is allowed:

- **attenuation** — the absorbance ∫μ dz (negative log of transmittance; linear in
  path length, ready as an absorption sinogram). Recovered from the field as
  −2·ln|ψ_det|, which equals the direct line integral when propagation is off.
- **accumulated phase** — φ = (2π/λ)·∫δ dz. Taken as −arg(ψ_det) (wrapped) when
  propagation is on, equal to the direct — unwrapped — line integral when off.
- **both** — attenuation and accumulated phase together.
- **intensity** — |ψ_det|², the transmitted intensity a detector measures. Equal to
  the Beer–Lambert transmittance exp(−∫μ dz) when propagation is off; with
  propagation it carries the edge-enhanced fringes.
- **complex** — the complex field ψ_det itself: the physical exit wave
  exp(−½·∫μ dz + i·φ), or the propagated detector field when multislice propagation is
  enabled.

Each run produces a projection stack shaped `(n_angles, height, width)` — real-valued
for attenuation/phase (two stacks when "both"), complex-valued for the complex option —
with the orientation/angle array retained as metadata, so real outputs drop directly
into `ProjectionData` (angles in radians, TomoPy convention) with no conversion.
