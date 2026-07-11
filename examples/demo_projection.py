"""Demo: simulate one X-ray projection of the current Blender scene and display it.

Run inside any Python that can import ``bpy`` and ``tktomo`` — the Blender scripting
window, the Blender MCP session, ``blender --python examples/demo_projection.py``,
or a ``pip install bpy`` environment.

What it does:

1. Uses the scene's tagged sample bodies (see :mod:`tktomo.blender_sim.scene`); if
   none exist, builds the built-in demo scene (cube + sphere) first.
2. Rotates the sample to ``ANGLE`` and simulates one attenuation projection
   (∫μ dz) through the full :func:`~tktomo.blender_sim.runner.simulate` pipeline;
   the field of view and pixel grid come from the camera intrinsics, matching a
   render exactly.
3. Displays it via :mod:`tktomo.blender_sim.viewer` as the ``xray_projection``
   image in an Image Editor; in ``blender --background`` mode it saves
   ``xray_projection.png`` instead.

For interactive use, prefer the "X-ray Sim" sidebar panel's Projection section
("Project now" / "Live update") — this script demonstrates the scripted route.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:  # make tktomo importable from Blender's Python
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from tktomo.blender_sim import scene as xscene
from tktomo.blender_sim import simulate, viewer

ANGLE = 0.4  # sample rotation about the vertical (Z) tomo axis, radians
ENERGY_KEV = 17.0
RESOLUTION = 128  # square detector, pixels


def main() -> None:
    import bpy

    bodies = xscene.bodies()
    if not bodies:
        print("No tagged sample bodies found — building the demo scene.")
        xscene.build_demo_scene()
        bodies = xscene.bodies()

    results = simulate(
        np.array([ANGLE]),
        energy_kev=ENERGY_KEV,
        detector_shape=(RESOLUTION, RESOLUTION),
        outputs=("attenuation",),
    )
    projection = results["attenuation"].projection(0)
    pixel_size = results["attenuation"].metadata["pixel_size"]
    image = viewer.to_image(projection)
    if bpy.app.background:
        path = os.path.abspath(f"{viewer.IMAGE_NAME}.png")
        image.filepath_raw = path
        image.file_format = "PNG"
        image.save()
        where = f"saved {path}"
    else:
        where = viewer.ensure_image_editor(image)

    print(
        f"Projected {len(bodies)} bodies at {np.degrees(ANGLE):.1f}° "
        f"({RESOLUTION}×{RESOLUTION} px, {pixel_size * 1e3:.3f} mm pixels, "
        f"{ENERGY_KEV} keV): attenuation ∫μdz range "
        f"[{projection.min():.3g}, {projection.max():.3g}] — {where}."
    )


if __name__ == "__main__":
    main()
