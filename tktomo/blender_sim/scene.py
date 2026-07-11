"""Blender scene layer: fixed camera/source, rotating sample, slab extraction.

This is the only module that touches ``bpy`` / ``mathutils``, and both are imported
**lazily inside functions** so ``import tktomo.blender_sim.scene`` stays safe in a
Blender-free environment.

Geometry convention (all lengths metres, Blender's default unit):

- The beam travels along **−Y**: the source sits at +Y, the camera (detector) at
  ``(0, −camera_distance, 0)`` looking toward the origin, up = +Z. Camera type picks
  the beam: orthographic = parallel, perspective = cone (with an ``xray_source``
  point at ``(0, +source_distance, 0)``).
- All sample bodies are parented to the ``sample_root`` Empty at the world origin
  and rotate as one rigid group; the camera and source never move. The root's
  rotation mode is **ZYX** (Z applied first), so its X/Y euler values tilt the scan
  axis and Z rotates about that tilted axis — hand-editing the root in Blender and
  the API's ``(x_tilt, y_tilt, scan)`` Euler input behave identically.
- Bodies carry their optical constants as custom properties ``xray_delta`` /
  ``xray_beta`` (set via :func:`add_body`).

Extraction ray-casts one ray per detector pixel through the evaluated scene and
accumulates per-body path lengths, binned into slabs of ``slice_spacing`` along the
beam axis. It is exact for closed meshes but O(pixels × crossings) with per-hit
Python overhead — keep detectors modest (≲128²) for interactive use.
"""

from __future__ import annotations

import math

import numpy as np

from tktomo.blender_sim.materials import Material, beta_from_mu

SAMPLE_ROOT = "sample_root"
CAMERA = "xray_camera"
SOURCE = "xray_source"
PROP_DELTA = "xray_delta"
PROP_BETA = "xray_beta"


def setup_scene(
    beam: str = "parallel",
    camera_distance: float = 0.5,
    source_distance: float = 0.5,
    detector_width: float = 0.1,
    clear: bool = True,
):
    """Create the fixed acquisition geometry; returns the ``sample_root`` Empty.

    ``beam`` is ``"parallel"`` (orthographic camera) or ``"cone"`` (perspective
    camera plus an ``xray_source`` Empty). ``detector_width`` only sizes the camera
    view (ortho scale / field of view) for visual inspection — the extraction uses
    the explicit pixel grid, not the render.
    """
    import bpy

    if beam not in ("parallel", "cone"):
        raise ValueError(f"beam must be 'parallel' or 'cone', got {beam!r}")
    if clear:
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)

    root = _ensure_root()

    camera_data = bpy.data.cameras.new(CAMERA)
    if beam == "parallel":
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = detector_width
    else:
        camera_data.type = "PERSP"
        camera_data.angle = 2.0 * math.atan(
            0.5 * detector_width / (camera_distance + source_distance)
        )
    camera = bpy.data.objects.new(CAMERA, camera_data)
    camera.location = (0.0, -camera_distance, 0.0)
    camera.rotation_euler = (math.pi / 2.0, 0.0, 0.0)  # look along +Y, up = +Z
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    if beam == "cone":
        source = bpy.data.objects.new(SOURCE, None)
        source.location = (0.0, source_distance, 0.0)
        bpy.context.collection.objects.link(source)

    bpy.context.view_layer.update()
    return root


def _ensure_root():
    """Return the ``sample_root`` Empty, creating it at the origin if missing."""
    import bpy

    root = bpy.data.objects.get(SAMPLE_ROOT)
    if root is None:
        root = bpy.data.objects.new(SAMPLE_ROOT, None)
        # ZYX order = Z applied first: the root's X/Y euler values tilt the scan
        # axis and Z rotates about that tilted axis (tilted-axis tomography).
        root.rotation_mode = "ZYX"
        bpy.context.collection.objects.link(root)
        bpy.context.view_layer.update()
    return root


def add_body(
    obj,
    material: Material | None = None,
    *,
    delta: float | None = None,
    beta: float | None = None,
    mu: float | None = None,
    energy_kev: float | None = None,
):
    """Include a mesh in the simulation: parent it to the sample root and tag δ/β.

    ``obj`` is a Blender object or its name. Optical constants come from either a
    :class:`~tktomo.blender_sim.materials.Material`, explicit ``delta``/``beta``, or
    ``delta`` + ``mu`` with ``energy_kev`` (β is derived at that energy). The sample
    root is created on demand and the object's world transform is preserved, so this
    works on any mesh already modelled in place. Values remain editable afterwards
    in Object Properties → Custom Properties (or the "X-ray Sim" sidebar panel, see
    :mod:`tktomo.blender_sim.ui_panel`).
    """
    import bpy

    if isinstance(obj, str):
        obj = bpy.data.objects[obj]
    if material is not None:
        delta, beta = material.delta, material.beta
    if delta is None:
        raise ValueError("Provide a Material or an explicit delta.")
    if beta is None:
        if mu is None or energy_kev is None:
            raise ValueError("Provide beta, or mu together with energy_kev.")
        beta = beta_from_mu(mu, energy_kev)

    root = _ensure_root()
    if obj.parent is not root:
        obj.parent = root
        # keep the object where the user modelled it, even if the sample is
        # currently rotated
        obj.matrix_parent_inverse = root.matrix_world.inverted()
    obj[PROP_DELTA] = float(delta)
    obj[PROP_BETA] = float(beta)
    return obj


def add_selected(**kwargs) -> list:
    """Tag every selected mesh object as a sample body (same kwargs as add_body)."""
    import bpy

    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError("No mesh objects selected — select the bodies to include.")
    return [add_body(obj, **kwargs) for obj in meshes]


def remove_body(obj) -> None:
    """Exclude ``obj`` (object or name) from the simulation; keeps its transform."""
    import bpy

    if isinstance(obj, str):
        obj = bpy.data.objects[obj]
    for prop in (PROP_DELTA, PROP_BETA):
        if prop in obj:
            del obj[prop]
    if obj.parent is not None and obj.parent.name == SAMPLE_ROOT:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world


def bodies() -> list:
    """All objects currently included in the simulation (possibly empty)."""
    import bpy

    root = bpy.data.objects.get(SAMPLE_ROOT)
    if root is None:
        return []
    return [
        obj
        for obj in root.children_recursive
        if PROP_DELTA in obj and PROP_BETA in obj
    ]


def set_sample_orientation(matrix: np.ndarray) -> None:
    """Set the sample root's rotation from a 3×3 rotation matrix."""
    import bpy
    import mathutils

    root = bpy.data.objects[SAMPLE_ROOT]
    rotation = mathutils.Matrix(np.asarray(matrix).tolist())
    if root.rotation_mode == "QUATERNION":
        root.rotation_quaternion = rotation.to_quaternion()
    elif root.rotation_mode == "AXIS_ANGLE":
        quaternion = rotation.to_quaternion()
        root.rotation_axis_angle = (quaternion.angle, *quaternion.axis)
    else:
        root.rotation_euler = rotation.to_euler(root.rotation_mode)
    bpy.context.view_layer.update()


def _bodies():
    tagged = bodies()
    if not tagged:
        raise ValueError(
            "No sample bodies found. Include meshes with scene.add_body(obj, "
            "delta=..., beta=...), scene.add_selected(...), or the 'X-ray Sim' "
            "sidebar panel (tktomo.blender_sim.ui_panel.register())."
        )
    return tagged


def _sample_y_bounds(depsgraph, bodies) -> tuple[float, float]:
    """World-space y extent of all (evaluated) sample bodies."""
    import mathutils

    y_values = []
    for obj in bodies:
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            y_values.append((evaluated.matrix_world @ mathutils.Vector(corner)).y)
    return min(y_values), max(y_values)


def _ray_segments(scene, depsgraph, origin, direction, t_max, epsilon):
    """March one ray through the scene; per-body (t_enter, t_exit) intervals."""
    segments: dict[str, list[tuple[float, float]]] = {}
    open_entries: dict[str, float] = {}
    position = origin.copy()
    travelled = 0.0
    while travelled < t_max:
        hit, location, normal, _, obj, _ = scene.ray_cast(
            depsgraph, position, direction, distance=t_max - travelled
        )
        if not hit:
            break
        travelled += (location - position).length
        name = obj.original.name if obj.original else obj.name
        if normal.dot(direction) < 0:  # front face: entering the body
            open_entries.setdefault(name, travelled)
        else:  # back face: leaving the body
            t_enter = open_entries.pop(name, None)
            if t_enter is not None:
                segments.setdefault(name, []).append((t_enter, travelled))
        position = location + direction * epsilon
        travelled += epsilon
    for name, t_enter in open_entries.items():  # never exited: clip to ray end
        segments.setdefault(name, []).append((t_enter, t_max))
    return segments


def extract_slab_integrals(
    detector_shape: tuple[int, int] = (64, 64),
    pixel_size: float = 1e-3,
    slice_spacing: float | None = None,
    margin: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-slab projected line integrals (∫δ dz, ∫β dz) for the current pose.

    Returns two ``(n_slabs, height, width)`` arrays in metres, slabs ordered
    entrance → exit along the beam. ``slice_spacing=None`` yields a single slab
    (the projection approximation). Beam type follows the active camera: ORTHO
    casts parallel rays; PERSP casts diverging rays from the ``xray_source`` point,
    so the maps are sampled on the demagnified sample-plane grid the Fresnel
    scaling propagator expects.
    """
    import bpy
    import mathutils

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bodies = _bodies()
    constants = {
        obj.name: (float(obj[PROP_DELTA]), float(obj[PROP_BETA])) for obj in bodies
    }
    y_min, y_max = _sample_y_bounds(depsgraph, bodies)
    extent = max(y_max - y_min, 0.0)

    if slice_spacing is None or slice_spacing >= extent or extent == 0.0:
        n_slabs = 1
        slab_edges = [(y_max, y_min)]
    else:
        n_slabs = math.ceil(extent / slice_spacing)
        slab_edges = [
            (y_max - k * slice_spacing, y_max - (k + 1) * slice_spacing)
            for k in range(n_slabs)
        ]

    camera = scene.camera
    cone = camera is not None and camera.data.type == "PERSP"
    if cone:
        try:
            source_location = bpy.data.objects[SOURCE].location.copy()
        except KeyError:
            raise ValueError(
                "Perspective camera needs an 'xray_source' object (see setup_scene)."
            ) from None
    detector_y = camera.location.y if camera is not None else y_min - margin

    height, width = detector_shape
    slab_delta = np.zeros((n_slabs, height, width))
    slab_beta = np.zeros((n_slabs, height, width))
    y_start = y_max + margin
    epsilon = max(1e-9, 1e-6 * (y_max - detector_y + margin))

    for i in range(height):
        z = ((height - 1) / 2.0 - i) * pixel_size
        for j in range(width):
            x = (j - (width - 1) / 2.0) * pixel_size
            if cone:
                origin = source_location.copy()
                target = mathutils.Vector((x, detector_y, z))
                direction = (target - origin).normalized()
            else:
                origin = mathutils.Vector((x, y_start, z))
                direction = mathutils.Vector((0.0, -1.0, 0.0))
            t_max = (origin.y - (y_min - margin)) / -direction.y
            segments = _ray_segments(scene, depsgraph, origin, direction, t_max, epsilon)
            for name, intervals in segments.items():
                if name not in constants:
                    continue  # non-sample object in the beam path
                delta, beta = constants[name]
                for t_enter, t_exit in intervals:
                    for k, (y_hi, y_lo) in enumerate(slab_edges):
                        # t range where the ray's y is inside [y_lo, y_hi]
                        t_hi = (origin.y - y_hi) / -direction.y
                        t_lo = (origin.y - y_lo) / -direction.y
                        overlap = min(t_exit, t_lo) - max(t_enter, t_hi)
                        if overlap > 0:
                            slab_delta[k, i, j] += delta * overlap
                            slab_beta[k, i, j] += beta * overlap
    return slab_delta, slab_beta


def build_demo_scene(
    beam: str = "parallel",
    camera_distance: float = 0.5,
    source_distance: float = 0.5,
    detector_width: float = 0.1,
):
    """Self-contained demo: a cube and a sphere with plausible X-ray constants.

    Lets every run mode work with no scene prepared, mirroring the phantom
    fallback the UIs use. Returns the created bodies.
    """
    import bpy

    setup_scene(
        beam=beam,
        camera_distance=camera_distance,
        source_distance=source_distance,
        detector_width=detector_width,
    )
    bpy.ops.mesh.primitive_cube_add(size=0.03, location=(0.008, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "demo_cube"
    add_body(cube, Material("demo_dense", delta=2e-6, beta=2e-9))

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.012, location=(-0.015, 0.0, 0.008))
    sphere = bpy.context.active_object
    sphere.name = "demo_sphere"
    add_body(sphere, Material("demo_light", delta=5e-7, beta=4e-10))

    bpy.context.view_layer.update()
    return [cube, sphere]
