"""Orchestrate a simulation run: orientations → per-output ProjectionData.

The heavy lifting is split between the numpy-only physics
(:mod:`~tktomo.blender_sim.multislice`) and the lazy-bpy scene layer
(:mod:`~tktomo.blender_sim.scene`); this module loops the sample orientations and
assembles the stacks. Importing it never requires ``bpy`` — only calling
:func:`simulate` does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tktomo.blender_sim.multislice import (
    detector_wave,
    projection_outputs,
    wave_outputs,
)
from tktomo.io.data import ProjectionData


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def normalize_orientations(orientations: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normalize sample orientations to rotation matrices plus tomo angles.

    Accepts, per the spec's "arbitrary orientations" input:

    - shape ``(n,)`` — single-axis tomography: rotation angles in **radians** about
      the vertical (world Z) tomo axis; returned unchanged as the angles array;
    - shape ``(n, 3)`` — tilted-axis angles ``(x_tilt, y_tilt, scan)`` in radians,
      applied Z-first (R = Rx·Ry·Rz, Blender's ``ZYX`` mode — the sample root's
      default): **x/y tilt the scan axis, z rotates about that tilted axis**. The
      scan (z) column is returned as the angles array;
    - shape ``(n, 3, 3)`` — rotation matrices used as-is (angles are zeros; the
      full poses belong in metadata).

    Returns ``(matrices (n, 3, 3), angles (n,))``.
    """
    array = np.asarray(orientations, dtype=np.float64)
    if array.ndim == 1:
        matrices = np.stack([_rotation_z(theta) for theta in array])
        return matrices, array
    if array.ndim == 2 and array.shape[1] == 3:
        matrices = np.stack(
            [_rotation_x(a) @ _rotation_y(b) @ _rotation_z(c) for a, b, c in array]
        )
        return matrices, array[:, 2].copy()
    if array.ndim == 3 and array.shape[1:] == (3, 3):
        return array, np.zeros(array.shape[0])
    raise ValueError(
        "orientations must have shape (n,) [angles about Z], (n, 3) [XYZ Euler] or "
        f"(n, 3, 3) [rotation matrices]; got shape {array.shape}"
    )


def simulate(
    orientations: Any,
    *,
    energy_kev: float,
    detector_shape: tuple[int, int] = (64, 64),
    pixel_size: float | None = None,
    outputs: tuple[str, ...] = ("attenuation",),
    propagate: bool = False,
    slice_spacing: float | None = None,
    distance: float = 0.0,
    method: str = "fresnel",
    slab_kernel: str = "fresnel",
    method_kwargs: dict | None = None,
) -> dict[str, ProjectionData]:
    """Run the simulation against the current Blender scene.

    Requires a scene prepared by :mod:`tktomo.blender_sim.scene` (a ``sample_root``
    Empty with material-tagged bodies and a fixed camera). For each orientation the
    sample Empty is rotated, per-slab line integrals are extracted, and the physics
    produces the selected outputs.

    The field of view and pixel grid come from the **camera intrinsics** (ortho
    scale, or focal length/sensor for cone beams), exactly matching a render at
    ``detector_shape`` resolution. ``pixel_size`` (sample-plane pitch, metres,
    used by the propagation physics) is derived from the camera when ``None``;
    pass a value only to override it. For cone beams the camera-derived grid is
    already the demagnified sample-plane grid, so ``fresnel_scaling`` uses it
    directly.

    Returns one :class:`ProjectionData` per requested output, with the orientation
    stack, energy and geometry recorded in metadata. Without ``propagate`` the
    attenuation/phase outputs are the exact (unwrapped) line integrals.
    """
    from tktomo.blender_sim import scene as scene_layer  # lazy: needs bpy at call time

    matrices, angles = normalize_orientations(orientations)
    kwargs = dict(method_kwargs or {})
    if pixel_size is None:
        pixel_size = scene_layer.camera_pixel_size(detector_shape)

    stacks: dict[str, list[np.ndarray]] = {name: [] for name in outputs}
    for matrix in matrices:
        scene_layer.set_sample_orientation(matrix)
        slab_delta_dz, slab_beta_dz = scene_layer.extract_slab_integrals(
            detector_shape=detector_shape,
            slice_spacing=slice_spacing if propagate else None,
        )
        if propagate:
            psi = detector_wave(
                slab_delta_dz,
                slab_beta_dz,
                energy_kev=energy_kev,
                pixel_size=pixel_size,
                slice_spacing=slice_spacing,
                distance=distance,
                method=method,
                slab_kernel=slab_kernel,
                method_kwargs=kwargs,
            )
            projection = wave_outputs(psi, outputs)
        else:
            projection = projection_outputs(
                slab_delta_dz.sum(axis=0), slab_beta_dz.sum(axis=0), energy_kev, outputs
            )
        for name in outputs:
            stacks[name].append(projection[name])

    metadata = {
        "orientations": matrices,
        "energy_kev": energy_kev,
        "pixel_size": pixel_size,
        "propagate": propagate,
        "slice_spacing": slice_spacing,
        "distance": distance,
        "method": method,
    }
    return {
        name: ProjectionData(
            data=np.stack(stacks[name]),
            angles=angles,
            metadata={**metadata, "output": name},
        )
        for name in outputs
    }
