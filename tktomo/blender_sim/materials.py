"""X-ray material parameters: δ, β, scene-level photon energy E and the linked μ.

The complex refractive index is n = 1 − δ + iβ. A single scene-level photon energy
converts β to the linear attenuation coefficient μ:

    μ = 4πβ/λ,   λ = hc/E   ⟹   μ = 4πβE/(hc)

Edit rules (from ``docs/blender-sim_instructions.md``):

- editing **β** recomputes μ (E and δ fixed);
- editing **μ** recomputes **β**, never E — β = μλ/(4π) at fixed E (δ unchanged);
- editing **δ** affects δ only;
- editing **E** rescales every material by the far-from-edge dispersion laws
  δ ∝ 1/E² and β ∝ 1/E³ (so μ follows ∝ 1/E²).

Units: lengths in metres, photon energy in keV, μ in 1/m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator

#: hc in keV·m (≈ 1.23984 keV·nm).
HC_KEV_M = 1.23984198e-9


def wavelength(energy_kev: float) -> float:
    """Photon wavelength in metres for ``energy_kev`` in keV."""
    if energy_kev <= 0:
        raise ValueError(f"photon energy must be positive, got {energy_kev}")
    return HC_KEV_M / energy_kev


def wavenumber(energy_kev: float) -> float:
    """Wavenumber k = 2π/λ in 1/m."""
    return 2.0 * math.pi / wavelength(energy_kev)


def mu_from_beta(beta: float, energy_kev: float) -> float:
    """Linear attenuation coefficient μ = 4πβ/λ in 1/m."""
    return 4.0 * math.pi * beta / wavelength(energy_kev)


def beta_from_mu(mu: float, energy_kev: float) -> float:
    """Imaginary part β = μλ/(4π) at fixed photon energy."""
    return mu * wavelength(energy_kev) / (4.0 * math.pi)


@dataclass
class Material:
    """Optical constants of one body: n = 1 − δ + iβ."""

    name: str
    delta: float
    beta: float


class MaterialSet:
    """Named materials plus the scene-level photon energy, enforcing the edit rules.

    Mutations must go through the ``set_*`` methods (or the ``energy_kev`` setter)
    so the δ/β/μ/E linkage stays consistent.
    """

    def __init__(self, energy_kev: float, materials: Iterable[Material] = ()) -> None:
        if energy_kev <= 0:
            raise ValueError(f"photon energy must be positive, got {energy_kev}")
        self._energy_kev = float(energy_kev)
        self._materials: dict[str, Material] = {}
        for material in materials:
            self._materials[material.name] = material

    # -- container protocol -------------------------------------------------
    def __getitem__(self, name: str) -> Material:
        try:
            return self._materials[name]
        except KeyError:
            raise KeyError(
                f"Unknown material {name!r}. Available: {sorted(self._materials)}"
            ) from None

    def __iter__(self) -> Iterator[Material]:
        return iter(self._materials.values())

    def __len__(self) -> int:
        return len(self._materials)

    def __contains__(self, name: object) -> bool:
        return name in self._materials

    def names(self) -> list[str]:
        return sorted(self._materials)

    def add(self, name: str, delta: float, beta: float) -> Material:
        material = Material(name=name, delta=float(delta), beta=float(beta))
        self._materials[name] = material
        return material

    # -- linked-parameter edit rules ----------------------------------------
    @property
    def energy_kev(self) -> float:
        return self._energy_kev

    @energy_kev.setter
    def energy_kev(self, new_energy_kev: float) -> None:
        """Change E; rescale all materials by δ ∝ 1/E², β ∝ 1/E³."""
        if new_energy_kev <= 0:
            raise ValueError(f"photon energy must be positive, got {new_energy_kev}")
        ratio = self._energy_kev / float(new_energy_kev)
        for material in self._materials.values():
            material.delta *= ratio**2
            material.beta *= ratio**3
        self._energy_kev = float(new_energy_kev)

    def set_delta(self, name: str, delta: float) -> None:
        """Edit δ only; β and μ are unaffected."""
        self[name].delta = float(delta)

    def set_beta(self, name: str, beta: float) -> None:
        """Edit β; μ follows implicitly (it is always derived as 4πβE/hc)."""
        self[name].beta = float(beta)

    def mu(self, name: str) -> float:
        """Linear attenuation coefficient of ``name`` at the current energy, 1/m."""
        return mu_from_beta(self[name].beta, self._energy_kev)

    def set_mu(self, name: str, mu: float) -> None:
        """Edit μ: updates β at the *fixed* current energy; E and δ unchanged."""
        self[name].beta = beta_from_mu(mu, self._energy_kev)
