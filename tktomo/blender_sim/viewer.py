"""In-Blender projection viewer: compute, display, and live-update projections.

Shows the attenuation projection of the **current scene pose** (it never touches
the sample orientation — rotate ``sample_root`` by hand and re-project) in a Blender
Image Editor, as the image datablock ``xray_projection``:

- :func:`show_projection` — one shot: compute, write the image, make sure an Image
  Editor shows it (splitting the 3D viewport on first use);
- :func:`enable_live_update` / :func:`disable_live_update` — a depsgraph handler
  marks the projection dirty whenever a sample body or the sample root changes, and
  a ~0.3 s timer re-projects outside the handler (cheap when nothing changed).

Defaults (photon energy, resolution) come from the "X-ray Sim" panel's fields when
:mod:`~tktomo.blender_sim.ui_panel` is registered, else module defaults. Lazy-bpy
like the rest of the scene layer: importing this module never needs Blender.
"""

from __future__ import annotations

import numpy as np

from tktomo.blender_sim import scene as scene_layer
from tktomo.blender_sim.multislice import projection_outputs

IMAGE_NAME = "xray_projection"
DEFAULT_ENERGY_KEV = 17.0
DEFAULT_RESOLUTION = 128
FIELD_MARGIN = 1.3  # detector field of view = margin × sample bounding radius

_live = False
_dirty = False
_updating = False


def auto_pixel_size(resolution: int, margin: float = FIELD_MARGIN) -> float:
    """Pixel pitch so the detector covers the sample at any rotation."""
    import bpy
    import mathutils

    depsgraph = bpy.context.evaluated_depsgraph_get()
    corners = []
    for obj in scene_layer._bodies():  # raises with guidance when nothing is tagged
        evaluated = obj.evaluated_get(depsgraph)
        corners += [
            evaluated.matrix_world @ mathutils.Vector(c) for c in evaluated.bound_box
        ]
    radius = max(corner.length for corner in corners)
    return margin * 2.0 * radius / resolution


def compute_projection(
    energy_kev: float = DEFAULT_ENERGY_KEV,
    resolution: int = DEFAULT_RESOLUTION,
    pixel_size: float | None = None,
) -> np.ndarray:
    """Attenuation projection (∫μ dz) of the scene at its current pose."""
    if pixel_size is None:
        pixel_size = auto_pixel_size(resolution)
    slab_delta, slab_beta = scene_layer.extract_slab_integrals(
        detector_shape=(resolution, resolution), pixel_size=pixel_size
    )
    return projection_outputs(
        slab_delta.sum(axis=0), slab_beta.sum(axis=0), energy_kev, ("attenuation",)
    )["attenuation"]


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
        before = set(screen.areas[:])
        try:
            with bpy.context.temp_override(window=window, area=viewport):
                bpy.ops.screen.area_split(direction="VERTICAL", factor=0.4)
            new_area = next(a for a in screen.areas if a not in before)
            new_area.type = "IMAGE_EDITOR"
            new_area.spaces.active.image = image
            return "opened a new Image Editor next to the viewport"
        except Exception:  # noqa: BLE001  (window-manager quirks: fall through)
            pass

    others = [area for area in screen.areas if area.type != "VIEW_3D"]
    if others:
        area = max(others, key=lambda a: a.width * a.height)
        area.type = "IMAGE_EDITOR"
        area.spaces.active.image = image
        return "converted an area to an Image Editor"
    return f"no editor area available — open an Image Editor and pick {image.name!r}"


def _settings() -> tuple[float, int]:
    """Panel fields when registered, module defaults otherwise."""
    import bpy

    wm = bpy.context.window_manager
    energy = float(getattr(wm, "tktomo_energy_kev", DEFAULT_ENERGY_KEV))
    resolution = int(getattr(wm, "tktomo_resolution", DEFAULT_RESOLUTION))
    return energy, resolution


def show_projection(
    energy_kev: float | None = None, resolution: int | None = None
) -> np.ndarray:
    """Compute the current-pose projection and display it; returns the array."""
    global _updating

    default_energy, default_resolution = _settings()
    _updating = True
    try:
        projection = compute_projection(
            energy_kev if energy_kev is not None else default_energy,
            resolution if resolution is not None else default_resolution,
        )
        ensure_image_editor(to_image(projection))
    finally:
        _updating = False
    return projection


# -- live update ---------------------------------------------------------------


def _mark_dirty(scene, depsgraph) -> None:  # depsgraph_update_post handler
    global _dirty
    if _updating:
        return
    relevant = {obj.name for obj in scene_layer.bodies()}
    relevant.add(scene_layer.SAMPLE_ROOT)
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
