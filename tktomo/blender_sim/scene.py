"""Blender scene layer: fixed camera, rotating sample, camera-matched extraction.

This is the only module that touches ``bpy`` / ``mathutils``, and both are imported
**lazily inside functions** so ``import tktomo.blender_sim.scene`` stays safe in a
Blender-free environment.

Geometry convention — **Blender coordinates are millimetres**: one Blender unit is
interpreted as 1 mm (``UNIT_SCALE`` metres), so samples are modelled at a
comfortable viewport scale. This module converts at the boundary: everything it
*returns* to the physics layer (line integrals, pixel pitch) is in **metres**, and
physics-side parameters it *receives* (``slice_spacing``) are in metres too. Only
raw Blender coordinates (object positions/sizes, ``camera_distance``,
``detector_width``, ``margin``) are mm:

- The camera is the detector and defines the whole ray geometry, exactly as a
  render would: **orthographic** = parallel beam with field of view =
  ``ortho_scale``; **perspective** = cone beam whose apex is the camera pinhole,
  with the field of view from the focal length / sensor (``camera.data.angle``).
  The default setup puts it at ``(0, −camera_distance, 0)`` looking at the origin,
  up = +Z, so the beam axis is Y.
- All sample bodies are parented to the ``sample_root`` Empty at the world origin
  and rotate as one rigid group; the camera never moves. The root's rotation mode
  is **ZYX** (Z applied first), so its X/Y euler values tilt the scan axis and Z
  rotates about that tilted axis — hand-editing the root in Blender and the API's
  ``(x_tilt, y_tilt, scan)`` Euler input behave identically.
- Bodies carry their optical constants as custom properties ``xray_delta`` /
  ``xray_beta`` (set via :func:`add_body`).

Extraction samples one ray per detector pixel — **the same rays a camera render
uses** (assuming zero lens shift and square pixel aspect) — casts it through the
evaluated scene and accumulates per-body path lengths, binned into slabs of
``slice_spacing`` along the camera view axis. It is exact for closed meshes but
O(pixels × crossings) with per-hit Python overhead — keep detectors modest
(≲128²) for interactive use.
"""

from __future__ import annotations

import math

import numpy as np

from tktomo.blender_sim.materials import Material, beta_from_mu

SAMPLE_ROOT = "sample_root"
CAMERA = "xray_camera"
PROP_DELTA = "xray_delta"
PROP_BETA = "xray_beta"

#: Metres per Blender unit: scene coordinates are interpreted as millimetres.
UNIT_SCALE = 1e-3


def setup_scene(
    beam: str = "parallel",
    camera_distance: float = 10.0,
    detector_width: float = 0.1,
    clear: bool = True,
):
    """Create the fixed acquisition geometry; returns the ``sample_root`` Empty.

    ``beam`` is ``"parallel"`` (orthographic camera) or ``"cone"`` (perspective
    camera — its pinhole is the cone apex). ``camera_distance`` and
    ``detector_width`` are Blender units (**mm**); ``detector_width`` is the field
    of view at the sample plane (through the origin) for both beam types; for cone
    it sets the focal angle as ``2·atan(detector_width / (2·camera_distance))``.
    The extraction derives its ray grid from these camera intrinsics, so the
    projection matches a render exactly.
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
        camera_data.angle = 2.0 * math.atan(0.5 * detector_width / camera_distance)
    camera = bpy.data.objects.new(CAMERA, camera_data)
    camera.location = (0.0, -camera_distance, 0.0)
    camera.rotation_euler = (math.pi / 2.0, 0.0, 0.0)  # look along +Y, up = +Z
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

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


def _camera_frame():
    """Active camera with its world-space (position, right, up, view) frame."""
    import bpy
    import mathutils

    camera = bpy.context.scene.camera
    if camera is None:
        raise ValueError("The scene has no active camera (see setup_scene).")
    rotation = camera.matrix_world.to_quaternion()
    right = rotation @ mathutils.Vector((1.0, 0.0, 0.0))
    up = rotation @ mathutils.Vector((0.0, 1.0, 0.0))
    view = rotation @ mathutils.Vector((0.0, 0.0, -1.0))  # cameras look along −Z
    return camera, camera.matrix_world.translation.copy(), right, up, view


def camera_pixel_size(detector_shape: tuple[int, int] = (64, 64)) -> float:
    """Sample-plane pixel pitch a render at this resolution would have, **metres**.

    The sample plane is the plane through the world origin perpendicular to the
    camera axis. ORTHO: field of view = ``ortho_scale`` everywhere. PERSP: field
    of view at the sample plane = ``2·d·tan(angle/2)`` with ``d`` the camera→
    origin distance and ``angle`` the focal angle (from focal length + sensor).
    Square pixels, sized by the larger detector dimension (Blender AUTO fit).
    Blender units (mm) are converted to metres via ``UNIT_SCALE``.
    """
    import math as _math

    camera, position, _, _, view = _camera_frame()
    if camera.data.type == "ORTHO":
        width = camera.data.ortho_scale
    else:
        distance = (-position).dot(view)
        if distance <= 0:
            raise ValueError("The camera must look toward the world origin.")
        width = 2.0 * distance * _math.tan(camera.data.angle / 2.0)
    return width * UNIT_SCALE / max(detector_shape)


def _sample_axis_bounds(depsgraph, bodies, view) -> tuple[float, float]:
    """Extent of all (evaluated) sample bodies along the camera view axis.

    Coordinates are signed distances s = P·view from the plane through the world
    origin perpendicular to the view axis (the sample plane).
    """
    import mathutils

    s_values = []
    for obj in bodies:
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            s_values.append(
                (evaluated.matrix_world @ mathutils.Vector(corner)).dot(view)
            )
    return min(s_values), max(s_values)


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
    slice_spacing: float | None = None,
    margin: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-slab projected line integrals (∫δ dz, ∫β dz) for the current pose.

    The rays are exactly the active camera's render rays for a ``detector_shape``
    image (assuming zero lens shift and square pixel aspect): ORTHO casts parallel
    rays across ``ortho_scale``; PERSP casts diverging rays from the camera
    pinhole with the field of view given by the focal length/sensor. The pixel
    grid lies on the sample plane through the world origin, with the pitch
    reported by :func:`camera_pixel_size` — for cone beams that is already the
    demagnified sample-plane grid the Fresnel scaling propagator expects.

    Returns two ``(n_slabs, height, width)`` arrays in **metres** (scene mm
    converted via ``UNIT_SCALE``), slabs of ``slice_spacing`` (**metres**, like
    every physics-side length) along the camera view axis ordered entrance → exit
    (the far side first, matching the physical beam travelling toward the
    camera). ``slice_spacing=None`` yields a single slab (the projection
    approximation). ``margin`` is in Blender units (mm).
    """
    import bpy

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bodies = _bodies()
    constants = {
        obj.name: (float(obj[PROP_DELTA]), float(obj[PROP_BETA])) for obj in bodies
    }
    # physics-side Δz (metres) → scene units for binning along the view axis
    scene_spacing = None if slice_spacing is None else slice_spacing / UNIT_SCALE

    camera, camera_position, right, up, view = _camera_frame()
    cone = camera.data.type == "PERSP"
    camera_distance = (-camera_position).dot(view)
    if cone and camera_distance <= 0:
        raise ValueError("The camera must look toward the world origin.")
    # grid pitch in scene units (mm); camera_pixel_size reports the same in metres
    scene_pixel = camera_pixel_size(detector_shape) / UNIT_SCALE
    plane_center = camera_position + camera_distance * view  # s = 0 on the axis

    s_min, s_max = _sample_axis_bounds(depsgraph, bodies, view)
    extent = max(s_max - s_min, 0.0)
    if scene_spacing is None or scene_spacing >= extent or extent == 0.0:
        slab_edges = [(s_max, s_min)]
    else:
        n_slabs = math.ceil(extent / scene_spacing)
        slab_edges = [
            (s_max - k * scene_spacing, s_max - (k + 1) * scene_spacing)
            for k in range(n_slabs)
        ]

    height, width = detector_shape
    slab_delta = np.zeros((len(slab_edges), height, width))
    slab_beta = np.zeros((len(slab_edges), height, width))
    epsilon = max(1e-9, 1e-6 * (extent + abs(camera_distance) + margin))

    for i in range(height):
        v = ((height - 1) / 2.0 - i) * scene_pixel  # row 0 = top of the render
        for j in range(width):
            u = (j - (width - 1) / 2.0) * scene_pixel
            pixel_point = plane_center + u * right + v * up
            if cone:
                origin = camera_position.copy()
                direction = (pixel_point - origin).normalized()
                cosine = direction.dot(view)  # ray obliquity: ds = cosine · dt
                s_origin = -camera_distance
            else:
                direction = view.copy()
                cosine = 1.0
                s_origin = s_min - margin
                origin = pixel_point + s_origin * view
            # t(s): arc length along the ray at which it crosses axis-coordinate s
            t_max = (s_max + margin - s_origin) / cosine
            segments = _ray_segments(scene, depsgraph, origin, direction, t_max, epsilon)
            for name, intervals in segments.items():
                if name not in constants:
                    continue  # non-sample object in the beam path
                delta, beta = constants[name]
                for t_enter, t_exit in intervals:
                    for k, (s_hi, s_lo) in enumerate(slab_edges):
                        t_lo = (s_lo - s_origin) / cosine
                        t_hi = (s_hi - s_origin) / cosine
                        overlap = min(t_exit, t_hi) - max(t_enter, t_lo)
                        if overlap > 0:  # path length: scene units (mm) → metres
                            slab_delta[k, i, j] += delta * overlap * UNIT_SCALE
                            slab_beta[k, i, j] += beta * overlap * UNIT_SCALE
    return slab_delta, slab_beta


def slab_count(slice_spacing: float | None) -> int:
    """Number of multislice slabs at the current pose (1 = single slab).

    ``slice_spacing`` is in **metres** (physics-side, like everywhere else);
    the count adapts to the sample's extent along the camera view axis, exactly
    as :func:`extract_slab_integrals` will slice it.
    """
    import bpy

    if not slice_spacing:
        return 1
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bodies = _bodies()
    _, _, _, _, view = _camera_frame()
    s_min, s_max = _sample_axis_bounds(depsgraph, bodies, view)
    extent = max(s_max - s_min, 0.0)
    scene_spacing = slice_spacing / UNIT_SCALE
    if extent == 0.0 or scene_spacing >= extent:
        return 1
    return math.ceil(extent / scene_spacing)


def build_demo_scene(
    beam: str = "parallel",
    camera_distance: float = 10.0,
    detector_width: float = 0.1,
):
    """Self-contained demo: a cube and a sphere with plausible X-ray constants.

    Lets every run mode work with no scene prepared, mirroring the phantom
    fallback the UIs use. All coordinates are Blender units = **mm**: a 30 µm
    cube and a 24 µm sphere inside a 100 µm × 100 µm orthographic field of view,
    camera 10 mm from the origin — a scale at which near-field propagation
    fringes are visible. Returns the created bodies.
    """
    import bpy

    setup_scene(
        beam=beam,
        camera_distance=camera_distance,
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
