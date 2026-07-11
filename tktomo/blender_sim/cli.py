"""Command-line entry point for the Blender X-ray simulator.

Run modes (same code, pick whichever Python has ``bpy``):

- headless Blender:
  ``blender --background --python tktomo/blender_sim/cli.py -- --output out.h5``
  (tktomo must be importable from Blender's bundled Python, e.g. installed into it
  or on ``PYTHONPATH``; the arguments after ``--`` are ours);
- ``pip install bpy`` environment: ``python -m tktomo.blender_sim --output out.h5``
  or the ``tktomo-blender-sim`` console script;
- live Blender: ``from tktomo.blender_sim.cli import main; main([...])``.

By default a demo scene (cube + sphere) is built; pass ``--use-scene`` to run
against whatever scene is already loaded (e.g. a ``.blend`` opened by Blender).
Results are written as HDF5: one group per output with ``data`` and ``angles``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tktomo-blender-sim",
        description="Simulate X-ray tomography projections from a Blender scene.",
    )
    parser.add_argument("--output", required=True, help="Output HDF5 file path.")
    parser.add_argument(
        "--angles",
        nargs=3,
        type=float,
        default=(0.0, 180.0, 32),
        metavar=("START", "STOP", "N"),
        help="Single-axis scan in degrees: start stop n (default: 0 180 32).",
    )
    parser.add_argument("--energy", type=float, default=17.0, help="Photon energy, keV.")
    parser.add_argument(
        "--pixel-size", type=float, default=1e-3, help="Detector pixel pitch, metres."
    )
    parser.add_argument(
        "--detector", type=int, nargs=2, default=(64, 64), metavar=("H", "W"),
        help="Detector shape in pixels (default: 64 64).",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["attenuation"],
        choices=["attenuation", "phase", "complex"],
        help="Quantities to return (any combination).",
    )
    parser.add_argument(
        "--beam", choices=["parallel", "cone"], default="parallel",
        help="Demo-scene beam geometry (ignored with --use-scene).",
    )
    parser.add_argument(
        "--use-scene", action="store_true",
        help="Run against the currently loaded Blender scene instead of the demo.",
    )
    parser.add_argument(
        "--propagate", action="store_true", help="Enable multislice propagation."
    )
    parser.add_argument(
        "--slice-spacing", type=float, default=None,
        help="Multislice slab thickness Δz, metres (default: single slab).",
    )
    parser.add_argument(
        "--distance", type=float, default=0.0,
        help="Sample→detector propagation distance z, metres.",
    )
    parser.add_argument(
        "--method",
        default="fresnel",
        choices=["fresnel", "angular_spectrum", "fraunhofer", "fresnel_scaling"],
        help="Propagation method for the exit-wave→detector hop.",
    )
    parser.add_argument(
        "--r1", type=float, default=None,
        help="Source→sample distance for fresnel_scaling, metres.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
        if "--" in argv:  # blender --python script.py -- <our args>
            argv = argv[argv.index("--") + 1 :]
    args = _build_parser().parse_args(argv)

    import h5py

    from tktomo.blender_sim import scene, simulate

    if not args.use_scene:
        scene.build_demo_scene(beam=args.beam)

    start, stop, n_angles = args.angles
    angles = np.linspace(np.radians(start), np.radians(stop), int(n_angles), endpoint=False)
    method_kwargs = {"r1": args.r1} if args.r1 is not None else {}

    results = simulate(
        angles,
        energy_kev=args.energy,
        pixel_size=args.pixel_size,
        detector_shape=tuple(args.detector),
        outputs=tuple(args.outputs),
        propagate=args.propagate,
        slice_spacing=args.slice_spacing,
        distance=args.distance,
        method=args.method,
        method_kwargs=method_kwargs,
    )

    with h5py.File(args.output, "w") as h5:
        for name, projection_data in results.items():
            group = h5.create_group(name)
            group.create_dataset("data", data=projection_data.data)
            group.create_dataset("angles", data=projection_data.angles)
            group.attrs["energy_kev"] = args.energy
            group.attrs["pixel_size"] = args.pixel_size
            group.attrs["method"] = args.method
    print(f"Wrote {', '.join(results)} to {args.output}")


if __name__ == "__main__":
    main()
