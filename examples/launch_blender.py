"""Open Blender with the X-ray simulation scene ready to use.

Run it with plain Python::

    python examples/launch_blender.py [--beam cone] [--blender /path/to/blender]

It finds the Blender executable (``--blender`` flag, else the ``BLENDER``
environment variable, else ``blender`` on PATH) and opens the GUI with:

- the **demo scene** built (cube + sphere tagged as sample bodies) — skipped when
  the opened scene already contains tagged bodies, so ``blender my_scene.blend
  --python examples/launch_blender.py`` keeps your own sample;
- the **"X-ray Sim" sidebar panel** registered (N-key sidebar in the 3D viewport).

The same file is the Blender-side startup script, so it also works directly::

    blender [file.blend] --python examples/launch_blender.py -- [--beam cone]
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open Blender with the TKtomo X-ray sim scene set up."
    )
    parser.add_argument(
        "--beam", choices=["parallel", "cone"], default="parallel",
        help="Demo-scene beam geometry (default: parallel).",
    )
    parser.add_argument(
        "--blender", default=None,
        help="Blender executable (default: $BLENDER or 'blender' on PATH).",
    )
    return parser.parse_args(argv)


def _setup_inside_blender(argv: list[str]) -> None:
    """Blender-side role: register the panel and build the demo scene if needed."""
    import bpy

    if REPO_ROOT not in sys.path:  # make tktomo importable from Blender's Python
        sys.path.insert(0, REPO_ROOT)
    args = _parse_args(argv)

    from tktomo.blender_sim import scene, ui_panel

    if not bpy.app.background:  # panels need a UI
        ui_panel.register()
    if scene.bodies():
        print("Tagged sample bodies already present — keeping the loaded scene.")
    else:
        scene.build_demo_scene(beam=args.beam)

    if not bpy.app.background:
        # build the viewer layout via a one-shot timer, once the UI is fully up:
        # projection Image Editor on top, camera's-eye 3D view below it, and the
        # "X-ray Sim" sidebar opened in the main viewport
        def _first_projection():
            from tktomo.blender_sim import viewer

            try:
                print(f"Viewer layout: {viewer.setup_viewer_layout()}")
            except Exception as exc:  # noqa: BLE001  (startup must not crash)
                print(f"Initial projection failed: {exc}")
            _open_sidebar()
            return None

        def _open_sidebar():
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type != "VIEW_3D":
                        continue
                    area.spaces.active.show_region_ui = True  # the N sidebar
                    for region in area.regions:  # best effort: focus our tab
                        if region.type == "UI":
                            try:
                                region.active_panel_category = "X-ray Sim"
                            except (AttributeError, TypeError):
                                pass

        bpy.app.timers.register(_first_projection, first_interval=0.5)

    names = ", ".join(obj.name for obj in scene.bodies())
    print(
        f"TKtomo X-ray sim ready ({args.beam} beam): bodies [{names}]; "
        "'X-ray Sim' tab in the viewport sidebar (press N) — use its Projection "
        "section ('Project now' / 'Live update')."
    )


def _launch_blender(argv: list[str]) -> int:
    """Plain-Python role: spawn the Blender GUI running this file as startup."""
    import shutil
    import subprocess

    args = _parse_args(argv)
    executable = args.blender or os.environ.get("BLENDER") or shutil.which("blender")
    if not executable:
        sys.exit(
            "Blender not found. Pass --blender /path/to/blender or set $BLENDER."
        )
    command = [executable, "--python", os.path.abspath(__file__)]
    if args.beam != "parallel":
        command += ["--", "--beam", args.beam]
    print("Launching:", " ".join(command))
    return subprocess.call(command)


def main() -> None:
    # We are "inside Blender" only in a real Blender process, not merely when the
    # pip-installed bpy module is importable (there bpy.app.binary_path is empty) —
    # in a pip-bpy environment this script must still *launch* the Blender GUI.
    try:
        import bpy
    except ImportError:
        in_blender = False
    else:
        in_blender = bool(bpy.app.binary_path)

    argv = sys.argv[1:]
    if "--" in argv:  # blender ... --python file.py -- <our args>
        argv = argv[argv.index("--") + 1 :]
    elif in_blender:
        argv = []  # whatever is left is Blender's own argv, not ours

    if in_blender:
        _setup_inside_blender(argv)
    else:
        sys.exit(_launch_blender(argv))


if __name__ == "__main__":
    main()
