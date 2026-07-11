"""In-Blender projection viewer: compute, display, and live-update projections.

Shows the attenuation projection of the **current scene pose** (it never touches
the sample orientation — rotate ``sample_root`` by hand and re-project) in a Blender
Image Editor, as the image datablock ``xray_projection``:

- :func:`show_projection` — one shot: compute, write the image, make sure an Image
  Editor shows it (splitting the 3D viewport on first use);
- :func:`enable_live_update` / :func:`disable_live_update` — a depsgraph handler
  marks the projection dirty whenever a sample body, the sample root, or the
  camera (pose *or* intrinsics such as focal length / ortho scale) changes, and
  a ~0.3 s timer re-projects outside the handler (cheap when nothing changed).

Defaults (photon energy, resolution) come from the "X-ray Sim" panel's fields when
:mod:`~tktomo.blender_sim.ui_panel` is registered, else module defaults. Lazy-bpy
like the rest of the scene layer: importing this module never needs Blender.
"""

from __future__ import annotations

import numpy as np

from tktomo.blender_sim import scene as scene_layer
from tktomo.blender_sim.multislice import (
    detector_wave,
    projection_outputs,
    wave_outputs,
)

IMAGE_NAME = "xray_projection"
DEFAULT_ENERGY_KEV = 17.0
DEFAULT_RESOLUTION = 128
DEFAULT_OUTPUT = "attenuation"  # or "phase" / "intensity"
DEFAULT_METHOD = "fresnel"
DEFAULT_DISTANCE = 0.05  # exit-wave → detector hop, metres
DEFAULT_SLICE_SPACING = 0.0  # 0 = single slab (projection approximation)
DEFAULT_R1 = 0.5  # source→sample distance for fresnel_scaling, metres

_live = False
_dirty = False
_updating = False


def compute_projection(
    energy_kev: float = DEFAULT_ENERGY_KEV,
    resolution: int = DEFAULT_RESOLUTION,
    output: str = DEFAULT_OUTPUT,
    propagate: bool = False,
    method: str = DEFAULT_METHOD,
    distance: float = DEFAULT_DISTANCE,
    slice_spacing: float | None = None,
    method_kwargs: dict | None = None,
) -> np.ndarray:
    """Projection of the scene at its current pose: ``output`` map, real-valued.

    ``output`` is ``"attenuation"`` (∫μ dz), ``"phase"`` (φ) or ``"intensity"``
    (|ψ|², what a detector measures — the right view of a propagated wave). Without
    ``propagate`` these are the exact projected line integrals; with it, the wave
    is multislice-propagated (slab thickness ``slice_spacing``; ``None``/0 =
    single slab) and carried to the detector over ``distance`` by the propagator
    named ``method`` (see ``available_propagators()``), and the map is read off
    the detector field (phase wrapped). Field of view and pixel grid come from
    the active camera's intrinsics, so the image matches a camera render at this
    resolution — frame with the camera, not with a pixel-size parameter.
    """
    slice_spacing = slice_spacing or None  # 0 means single slab
    slab_delta, slab_beta = scene_layer.extract_slab_integrals(
        detector_shape=(resolution, resolution),
        slice_spacing=slice_spacing if propagate else None,
    )
    if not propagate:
        return projection_outputs(
            slab_delta.sum(axis=0), slab_beta.sum(axis=0), energy_kev, (output,)
        )[output]
    psi = detector_wave(
        slab_delta,
        slab_beta,
        energy_kev=energy_kev,
        pixel_size=scene_layer.camera_pixel_size((resolution, resolution)),
        slice_spacing=slice_spacing,
        distance=distance,
        method=method,
        method_kwargs=method_kwargs,
    )
    return wave_outputs(psi, (output,))[output]


def to_image(array: np.ndarray, name: str = IMAGE_NAME):
    """Write a 2D array into a grayscale Blender image datablock."""
    import bpy

    height, width = array.shape
    span = np.ptp(array)
    normalized = (array - array.min()) / (span if span > 0 else 1.0)
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3] = normalized[..., None]
    rgba[..., 3] = 1.0
    rgba = np.flipud(rgba)  # our row 0 is the top; Blender's is the bottom

    image = bpy.data.images.get(name)
    if image is None or image.size[0] != width or image.size[1] != height:
        if image is not None:
            bpy.data.images.remove(image)
        image = bpy.data.images.new(name, width, height, alpha=True, float_buffer=True)
    image.pixels.foreach_set(rgba.ravel())
    image.update()
    return image


def _split_area(window, area, direction: str, factor: float):
    """Split ``area``; return the newly created area (or None on failure)."""
    import bpy

    screen = window.screen
    before = set(screen.areas[:])
    try:
        with bpy.context.temp_override(window=window, area=area):
            bpy.ops.screen.area_split(direction=direction, factor=factor)
        return next(a for a in screen.areas if a not in before)
    except Exception:  # noqa: BLE001  (window-manager quirks: caller falls back)
        return None


def ensure_image_editor(image) -> str:
    """Make an Image Editor show ``image``, splitting the 3D viewport if needed."""
    import bpy

    if bpy.app.background:
        return "background mode: image datablock updated"

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "IMAGE_EDITOR":
                area.spaces.active.image = image
                return "shown in existing Image Editor"

    window = bpy.context.window_manager.windows[0]
    screen = window.screen
    viewports = [area for area in screen.areas if area.type == "VIEW_3D"]
    if viewports:
        viewport = max(viewports, key=lambda a: a.width * a.height)
        new_area = _split_area(window, viewport, "VERTICAL", 0.4)
        if new_area is not None:
            new_area.type = "IMAGE_EDITOR"
            new_area.spaces.active.image = image
            return "opened a new Image Editor next to the viewport"

    others = [area for area in screen.areas if area.type != "VIEW_3D"]
    if others:
        area = max(others, key=lambda a: a.width * a.height)
        area.type = "IMAGE_EDITOR"
        area.spaces.active.image = image
        return "converted an area to an Image Editor"
    return f"no editor area available — open an Image Editor and pick {image.name!r}"


def setup_viewer_layout() -> str:
    """Launcher layout: projection Image Editor on top, camera's-eye view below.

    Shows the first projection (creating the side Image Editor if needed), then
    splits that editor with a horizontal line: the top pane keeps the projection,
    the bottom pane becomes a 3D viewport locked to the scene camera's view — the
    beam's-eye preview of what is being projected. GUI only (no-op headless).
    """
    import bpy

    show_projection()  # creates/updates the image and an Image Editor area
    if bpy.app.background:
        return "background mode: no layout"

    for window in bpy.context.window_manager.windows:
        editors = [a for a in window.screen.areas if a.type == "IMAGE_EDITOR"]
        if not editors:
            continue
        editor = max(editors, key=lambda a: a.width * a.height)
        # already stacked above a camera view? (idempotent across re-runs)
        for area in window.screen.areas:
            if (
                area.type == "VIEW_3D"
                and area.x == editor.x
                and area.spaces.active.region_3d.view_perspective == "CAMERA"
            ):
                return "viewer layout already present"
        new_area = _split_area(window, editor, "HORIZONTAL", 0.5)
        if new_area is None:
            return "could not split the Image Editor; projection shown alone"
        top, bottom = (
            (editor, new_area) if editor.y > new_area.y else (new_area, editor)
        )
        image = bpy.data.images.get(IMAGE_NAME)
        top.type = "IMAGE_EDITOR"
        top.spaces.active.image = image
        bottom.type = "VIEW_3D"
        bottom.spaces.active.region_3d.view_perspective = "CAMERA"
        return "projection above, camera view below"
    return "no Image Editor available for the viewer layout"


def _settings() -> dict:
    """compute_projection kwargs from the panel fields (module defaults if absent)."""
    import bpy

    wm = bpy.context.window_manager
    method = str(getattr(wm, "tktomo_method", DEFAULT_METHOD))
    kwargs = {
        "energy_kev": float(getattr(wm, "tktomo_energy_kev", DEFAULT_ENERGY_KEV)),
        "resolution": int(getattr(wm, "tktomo_resolution", DEFAULT_RESOLUTION)),
        "output": str(getattr(wm, "tktomo_output", DEFAULT_OUTPUT)),
        "propagate": bool(getattr(wm, "tktomo_propagate", False)),
        "method": method,
        "distance": float(getattr(wm, "tktomo_distance", DEFAULT_DISTANCE)),
        "slice_spacing": float(getattr(wm, "tktomo_slice_spacing", DEFAULT_SLICE_SPACING)),
    }
    if method == "fresnel_scaling":
        kwargs["method_kwargs"] = {"r1": float(getattr(wm, "tktomo_r1", DEFAULT_R1))}
    return kwargs


def show_projection(**overrides) -> np.ndarray:
    """Compute the current-pose projection and display it; returns the array.

    Settings come from the "X-ray Sim" panel fields (or module defaults);
    keyword ``overrides`` are passed through to :func:`compute_projection`.
    """
    global _updating

    settings = {**_settings(), **overrides}
    _updating = True
    try:
        projection = compute_projection(**settings)
        ensure_image_editor(to_image(projection))
    finally:
        _updating = False
    return projection


# -- live update ---------------------------------------------------------------


def notify_settings_changed() -> None:
    """Mark the projection dirty after a panel-setting edit.

    WindowManager property edits never enter the depsgraph, so the depsgraph
    handler cannot see them — the panel's ``update=`` callbacks call this instead.
    The live timer picks it up on its next tick (no-op while live update is off).
    """
    global _dirty
    _dirty = True


def _mark_dirty(scene, depsgraph) -> None:  # depsgraph_update_post handler
    global _dirty
    if _updating:
        return
    relevant = {obj.name for obj in scene_layer.bodies()}
    relevant.add(scene_layer.SAMPLE_ROOT)
    if scene.camera is not None:
        # the camera defines FOV/pixel grid: re-project on moves (object update)
        # and on intrinsics edits like focal length / ortho scale (data update)
        relevant.add(scene.camera.name)
        relevant.add(scene.camera.data.name)
    for update in depsgraph.updates:
        if getattr(update.id, "name", None) in relevant:
            _dirty = True
            return


def _live_timer() -> float | None:
    """Re-project when dirty; heavy work happens here, not in the handler."""
    global _dirty
    if not _live:
        return None  # unregisters the timer
    if _dirty:
        _dirty = False
        try:
            show_projection()
        except ValueError as exc:  # e.g. all bodies removed
            print(f"live projection skipped: {exc}")
    return 0.3


def enable_live_update() -> None:
    """Re-project automatically whenever the sample or its pose changes."""
    global _live, _dirty
    import bpy

    if _live:
        return
    _live = True
    _dirty = True  # project right away
    if _mark_dirty not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_mark_dirty)
    if not bpy.app.timers.is_registered(_live_timer):
        bpy.app.timers.register(_live_timer, first_interval=0.1)


def disable_live_update() -> None:
    global _live
    import bpy

    _live = False  # the timer sees this and unregisters itself
    if _mark_dirty in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_mark_dirty)
