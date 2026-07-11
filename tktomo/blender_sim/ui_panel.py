"""Viewport sidebar panel for tagging sample bodies without writing code.

Enable it from the Blender scripting window (or MCP session):

    from tktomo.blender_sim import ui_panel
    ui_panel.register()

An **"X-ray Sim"** tab appears in the 3D viewport sidebar (press ``N``):

- set default δ/β, select meshes, click **Include selected** — done, the meshes are
  parented to ``sample_root`` and picked up by the simulation;
- the active body's δ/β stay editable in the panel (they are the ``xray_delta`` /
  ``xray_beta`` custom properties, so Object Properties → Custom Properties works
  too), and **Exclude active** removes it;
- the panel lists everything currently included;
- a **Projection** section with a **Project now** button and a **Live update**
  toggle that re-projects whenever the sample, its pose, or the camera changes —
  both display in an Image Editor via :mod:`~tktomo.blender_sim.viewer`. Fields:
  energy, resolution, **output** (attenuation or phase), and an optional
  **Propagate** stage with **method** (enumerated from the propagator registry),
  detector distance, multislice slice spacing Δz, and r1 for ``fresnel_scaling``.

Following the repo's lazy-import rule, ``bpy`` is only imported inside
:func:`register` / :func:`unregister`, so importing this module never needs Blender.
"""

from __future__ import annotations

_classes: list = []


def register() -> None:
    """Register the panel, its operators and the default-value properties."""
    import bpy

    from tktomo.blender_sim import scene as scene_layer
    from tktomo.blender_sim import viewer

    if _classes:  # already registered by this module instance
        return
    # drop leftovers from a previous module instance (script reloads)
    for name in (
        "TKTOMO_OT_include_selected",
        "TKTOMO_OT_exclude_active",
        "TKTOMO_OT_project_now",
        "TKTOMO_PT_xray_sim",
    ):
        stale = getattr(bpy.types, name, None)
        if stale is not None:
            try:
                bpy.utils.unregister_class(stale)
            except RuntimeError:
                pass

    class TKTOMO_OT_include_selected(bpy.types.Operator):
        """Parent selected meshes to sample_root and tag them with the defaults."""

        bl_idname = "tktomo.include_selected"
        bl_label = "Include selected"
        bl_options = {"REGISTER", "UNDO"}

        def execute(self, context):
            wm = context.window_manager
            try:
                added = scene_layer.add_selected(
                    delta=wm.tktomo_delta, beta=wm.tktomo_beta
                )
            except ValueError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            self.report({"INFO"}, f"Included {len(added)} mesh(es) in the simulation")
            return {"FINISHED"}

    class TKTOMO_OT_exclude_active(bpy.types.Operator):
        """Remove the active object from the simulation."""

        bl_idname = "tktomo.exclude_active"
        bl_label = "Exclude active"
        bl_options = {"REGISTER", "UNDO"}

        @classmethod
        def poll(cls, context):
            obj = context.active_object
            return obj is not None and scene_layer.PROP_DELTA in obj

        def execute(self, context):
            scene_layer.remove_body(context.active_object)
            return {"FINISHED"}

    class TKTOMO_OT_project_now(bpy.types.Operator):
        """Project the scene at its current pose and show it in an Image Editor."""

        bl_idname = "tktomo.project_now"
        bl_label = "Project now"
        bl_options = {"REGISTER"}

        def execute(self, context):
            try:
                projection = viewer.show_projection()
            except ValueError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            self.report(
                {"INFO"},
                f"Projection updated (∫μdz max {projection.max():.3g})",
            )
            return {"FINISHED"}

    class TKTOMO_PT_xray_sim(bpy.types.Panel):
        bl_label = "TKtomo X-ray bodies"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "X-ray Sim"

        def draw(self, context):
            layout = self.layout
            wm = context.window_manager

            column = layout.column(align=True)
            column.label(text="Defaults for new bodies:")
            column.prop(wm, "tktomo_delta")
            column.prop(wm, "tktomo_beta")
            column.operator(TKTOMO_OT_include_selected.bl_idname, icon="ADD")

            obj = context.active_object
            if obj is not None and scene_layer.PROP_DELTA in obj:
                box = layout.box()
                box.label(text=f"Active: {obj.name}", icon="MESH_DATA")
                box.prop(obj, f'["{scene_layer.PROP_DELTA}"]', text="delta")
                box.prop(obj, f'["{scene_layer.PROP_BETA}"]', text="beta")
                box.operator(TKTOMO_OT_exclude_active.bl_idname, icon="X")

            included = scene_layer.bodies()
            layout.label(text=f"In simulation ({len(included)}):")
            for body in included:
                layout.label(text=body.name, icon="CHECKMARK")

            box = layout.box()
            box.label(text="Projection", icon="IMAGE_DATA")
            column = box.column(align=True)
            column.prop(wm, "tktomo_energy_kev")
            column.prop(wm, "tktomo_resolution")
            column.prop(wm, "tktomo_output")
            box.prop(wm, "tktomo_propagate", toggle=True, icon="OUTLINER_OB_LIGHTPROBE")
            if wm.tktomo_propagate:
                column = box.column(align=True)
                column.prop(wm, "tktomo_method")
                column.prop(wm, "tktomo_distance")
                column.prop(wm, "tktomo_slice_spacing")
                if wm.tktomo_method == "fresnel_scaling":
                    column.prop(wm, "tktomo_r1")
            box.operator(TKTOMO_OT_project_now.bl_idname, icon="RENDER_STILL")
            box.prop(wm, "tktomo_live", toggle=True, icon="FILE_REFRESH")

    def _toggle_live(self, context):
        if self.tktomo_live:
            viewer.enable_live_update()
        else:
            viewer.disable_live_update()

    def _settings_changed(self, context):
        # WindowManager props never enter the depsgraph, so tell the viewer
        # directly; the live timer re-projects on its next tick.
        viewer.notify_settings_changed()

    def _propagate_toggled(self, context):
        if self.tktomo_propagate:
            # a detector measures |ψ|²: show intensity when propagating
            self.tktomo_output = "intensity"
        viewer.notify_settings_changed()

    from tktomo.blender_sim.propagation import available_propagators

    bpy.types.WindowManager.tktomo_energy_kev = bpy.props.FloatProperty(
        name="Energy (keV)", description="Photon energy used for the projection",
        default=viewer.DEFAULT_ENERGY_KEV, min=0.001, update=_settings_changed,
    )
    bpy.types.WindowManager.tktomo_resolution = bpy.props.IntProperty(
        name="Resolution (px)",
        description="Detector size; field of view comes from the camera intrinsics",
        default=viewer.DEFAULT_RESOLUTION, min=8, max=1024, update=_settings_changed,
    )
    bpy.types.WindowManager.tktomo_output = bpy.props.EnumProperty(
        name="Output",
        description="Quantity shown in the projection view",
        items=[
            ("attenuation", "Attenuation", "Absorbance ∫μ dz (from β)"),
            ("phase", "Phase", "Accumulated phase φ = (2π/λ)·∫δ dz (from δ); "
             "wrapped to (−π, π] when propagation is on"),
            ("intensity", "Intensity", "|ψ|² — the transmitted intensity a detector "
             "measures; shows propagation fringes"),
        ],
        default=viewer.DEFAULT_OUTPUT, update=_settings_changed,
    )
    bpy.types.WindowManager.tktomo_propagate = bpy.props.BoolProperty(
        name="Propagate",
        description="Multislice-propagate the exit wave to the detector instead of "
        "showing the raw projected line integrals (switches the output to "
        "intensity, what a detector sees)",
        default=False, update=_propagate_toggled,
    )
    bpy.types.WindowManager.tktomo_method = bpy.props.EnumProperty(
        name="Method",
        description="Propagator for the exit-wave → detector hop "
        "(multislice slab steps always use a near-field kernel)",
        items=[(name, name, "") for name in available_propagators()],
        default=viewer.DEFAULT_METHOD, update=_settings_changed,
    )
    bpy.types.WindowManager.tktomo_distance = bpy.props.FloatProperty(
        name="Distance (m)",
        description="Sample → detector free-space propagation distance z",
        default=viewer.DEFAULT_DISTANCE, min=0.0, update=_settings_changed,
    )
    bpy.types.WindowManager.tktomo_slice_spacing = bpy.props.FloatProperty(
        name="Slice Δz (m)",
        description="Multislice slab thickness along the beam; 0 = single slab "
        "(projection approximation)",
        default=viewer.DEFAULT_SLICE_SPACING, min=0.0, update=_settings_changed,
    )
    bpy.types.WindowManager.tktomo_r1 = bpy.props.FloatProperty(
        name="r1 (m)",
        description="Source → sample distance for fresnel_scaling (cone beam); "
        "≈ the camera distance in this geometry",
        default=viewer.DEFAULT_R1, min=1e-9, update=_settings_changed,
    )
    bpy.types.WindowManager.tktomo_live = bpy.props.BoolProperty(
        name="Live update",
        description="Re-project automatically when the sample or its pose changes",
        default=False, update=_toggle_live,
    )
    bpy.types.WindowManager.tktomo_delta = bpy.props.FloatProperty(
        name="delta", description="Refractive-index decrement δ of n = 1 − δ + iβ",
        default=1e-6, min=0.0, precision=6, step=1,
    )
    bpy.types.WindowManager.tktomo_beta = bpy.props.FloatProperty(
        name="beta", description="Imaginary part β of n = 1 − δ + iβ (sets μ = 4πβ/λ)",
        default=1e-9, min=0.0, precision=6, step=1,
    )

    for cls in (
        TKTOMO_OT_include_selected,
        TKTOMO_OT_exclude_active,
        TKTOMO_OT_project_now,
        TKTOMO_PT_xray_sim,
    ):
        bpy.utils.register_class(cls)
        _classes.append(cls)


def unregister() -> None:
    import bpy

    from tktomo.blender_sim import viewer

    viewer.disable_live_update()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    _classes.clear()
    for prop in ("tktomo_delta", "tktomo_beta", "tktomo_energy_kev",
                 "tktomo_resolution", "tktomo_output", "tktomo_propagate",
                 "tktomo_method", "tktomo_distance", "tktomo_slice_spacing",
                 "tktomo_r1", "tktomo_live"):
        if hasattr(bpy.types.WindowManager, prop):
            delattr(bpy.types.WindowManager, prop)
