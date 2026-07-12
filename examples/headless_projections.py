"""Generate X-ray projections headlessly from a .blend file into HDF5.

The scene must contain tagged sample bodies and an X-ray camera (e.g. a scene
saved from a ``launch_blender.py`` session). Orientations are **z rotation
angles only** — a single-axis scan about the vertical tomo axis, so the angles
drop straight into TomoPy conventions.

Run with headless Blender (h5py must be importable from Blender's Python)::

    blender --background my_scene.blend --python examples/headless_projections.py -- \
        --angles 0 180 64 --output projections.h5

or in a ``pip install bpy`` environment (bpy + h5py in the same Python)::

    python examples/headless_projections.py --blend my_scene.blend \
        --angles 0 180 64 --output projections.h5

The HDF5 file contains one group per requested output with ``data``
(n_angles, height, width) and ``angles`` (radians) datasets, plus the relevant
metadata as attributes (energy, sample-plane pixel size in metres, propagation
settings, source .blend).
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:  # make tktomo importable from Blender's Python
    sys.path.insert(0, _REPO_ROOT)

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless X-ray projections from a .blend file into HDF5."
    )
    parser.add_argument(
        "--blend", default=None,
        help="Path to the .blend file to load (omit when Blender already opened "
        "one, e.g. 'blender --background scene.blend --python ...').",
    )
    parser.add_argument("--output", required=True, help="Output HDF5 file path.")
    parser.add_argument(
        "--angles", nargs=3, type=float, default=(0.0, 180.0, 32),
        metavar=("START", "STOP", "N"),
        help="Z rotation angles in degrees: start stop n, stop excluded "
        "(default: 0 180 32).",
    )
    parser.add_argument("--energy", type=float, default=17.0, help="Photon energy, keV.")
    parser.add_argument(
        "--detector", type=int, nargs=2, default=(128, 128), metavar=("H", "W"),
        help="Detector shape in pixels; field of view comes from the camera "
        "(default: 128 128).",
    )
    parser.add_argument(
        "--outputs", nargs="+", default=["attenuation"],
        choices=["attenuation", "phase", "intensity", "complex"],
        help="Quantities to write (any combination).",
    )
    parser.add_argument(
        "--propagate", action="store_true",
        help="Propagate the exit wave to the detector.",
    )
    parser.add_argument(
        "--method", default="fresnel",
        choices=["fresnel", "angular_spectrum", "fraunhofer", "fresnel_scaling"],
        help="Propagation method for the exit-wave → detector hop.",
    )
    parser.add_argument(
        "--distance", type=float, default=1e-3,
        help="Sample → detector propagation distance, metres (default 1 mm).",
    )
    parser.add_argument(
        "--slice-spacing", type=float, default=None,
        help="Enable multislice with this slab thickness Δz in metres "
        "(default: single slab).",
    )
    parser.add_argument(
        "--r1", type=float, default=None,
        help="Source → sample distance for fresnel_scaling, metres.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    import bpy

    try:
        import h5py
    except ImportError:
        sys.exit(
            "h5py is not importable from this Python. Either install it into "
            "Blender's bundled Python, or run in a pip-bpy environment: "
            "python examples/headless_projections.py --blend scene.blend …"
        )

    if argv is None:
        argv = sys.argv[1:]
        if "--" in argv:  # blender ... --python script.py -- <our args>
            argv = argv[argv.index("--") + 1 :]
        elif bpy.app.binary_path:
            argv = []  # real Blender process without '--': argv is Blender's own
    args = _build_parser().parse_args(argv)

    if args.blend:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))

    from tktomo.blender_sim import scene
    from tktomo.blender_sim.runner import simulate

    bodies = scene.bodies()
    if not bodies:
        sys.exit(
            "The scene contains no tagged sample bodies (xray_delta/xray_beta); "
            "prepare it with launch_blender.py / scene.add_body first."
        )

    start, stop, n_angles = args.angles
    angles = np.linspace(
        np.radians(start), np.radians(stop), int(n_angles), endpoint=False
    )
    method_kwargs = {"r1": args.r1} if args.r1 is not None else {}
    results = simulate(
        angles,
        energy_kev=args.energy,
        detector_shape=tuple(args.detector),
        outputs=tuple(args.outputs),
        propagate=args.propagate,
        slice_spacing=args.slice_spacing,
        distance=args.distance,
        method=args.method,
        method_kwargs=method_kwargs,
    )

    blend_name = args.blend or bpy.data.filepath or "<current scene>"
    with h5py.File(args.output, "w") as h5:
        for name, projection_data in results.items():
            group = h5.create_group(name)
            group.create_dataset("data", data=projection_data.data)
            group.create_dataset("angles", data=projection_data.angles)
            group["angles"].attrs["units"] = "radians"
            group["angles"].attrs["axis"] = "z rotation of the sample (tomo axis)"
            group.attrs["energy_kev"] = args.energy
            group.attrs["pixel_size_m"] = projection_data.metadata["pixel_size"]
            group.attrs["detector_shape"] = args.detector
            group.attrs["propagate"] = args.propagate
            group.attrs["method"] = args.method
            group.attrs["distance_m"] = args.distance
            group.attrs["slice_spacing_m"] = args.slice_spacing or 0.0
            group.attrs["bodies"] = [obj.name for obj in bodies]
            group.attrs["blend_file"] = str(blend_name)
            group.attrs["blender_unit"] = "mm (UNIT_SCALE = 1e-3 m)"
    print(
        f"Wrote {', '.join(results)} ({int(n_angles)} angles, "
        f"{args.detector[0]}×{args.detector[1]} px) from {blend_name} "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
