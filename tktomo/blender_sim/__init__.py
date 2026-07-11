"""Blender-based X-ray tomography projection simulator.

Layering (see ``docs/blender-sim_instructions.md``):

- **Physics layer (numpy-only)** — :mod:`~tktomo.blender_sim.materials`,
  :mod:`~tktomo.blender_sim.propagation`, :mod:`~tktomo.blender_sim.multislice`.
  Importable and unit-testable with no Blender installed.
- **Scene/geometry layer (lazy bpy)** — :mod:`~tktomo.blender_sim.scene`. The only
  code touching ``bpy``, imported lazily inside functions.
- **Orchestration** — :mod:`~tktomo.blender_sim.runner` loops sample orientations,
  extracts per-slab line integrals from the Blender scene and returns
  :class:`~tktomo.io.data.ProjectionData` per requested output.

Units: all lengths in **metres** (Blender's default unit), photon energy in **keV**,
μ in 1/m. A bare ``import tktomo.blender_sim`` must never pull in ``bpy``.
"""

from tktomo.blender_sim.materials import (
    HC_KEV_M,
    Material,
    MaterialSet,
    beta_from_mu,
    mu_from_beta,
    wavelength,
    wavenumber,
)
from tktomo.blender_sim.multislice import (
    VALID_OUTPUTS,
    detector_wave,
    exit_wave,
    multislice_wave,
    projection_outputs,
    wave_outputs,
)
from tktomo.blender_sim.propagation import (
    available_propagators,
    get_propagator,
    register_propagator,
)
from tktomo.blender_sim.runner import normalize_orientations, simulate

__all__ = [
    "HC_KEV_M",
    "Material",
    "MaterialSet",
    "VALID_OUTPUTS",
    "available_propagators",
    "beta_from_mu",
    "detector_wave",
    "exit_wave",
    "get_propagator",
    "mu_from_beta",
    "multislice_wave",
    "normalize_orientations",
    "projection_outputs",
    "register_propagator",
    "simulate",
    "wave_outputs",
    "wavelength",
    "wavenumber",
]
