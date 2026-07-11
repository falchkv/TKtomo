"""Exit-wave formation, the multislice loop, and output extraction (numpy-only).

For each detector pixel the exit wave through the sample under the projection
approximation is

    ψ_exit = exp(−½·∫μ dz − i·φ),   φ = (2π/λ)·∫δ dz,   ∫μ dz = 2k·∫β dz

with the ``e^{+ikz}`` carrier convention shared with
:mod:`~tktomo.blender_sim.propagation` (δ *reduces* the optical path, so the
accumulated phase enters with a minus sign; the reported "accumulated phase"
output is the positive φ = −arg ψ).

Multislice: the sample is divided into slabs along the beam by a fixed
slice-spacing Δz (count adapts per orientation). Per slab, from the slab's
projected line integrals, transmit then propagate to the next slab plane with a
near-field kernel; the final hop to the detector uses the selected propagation
method. Δz ≥ sample extent ⇒ a single slab ⇒ the projection approximation.

Inputs are per-slab *line integrals* ``∫δ dz`` / ``∫β dz`` in metres (arrays of
shape ``(n_slabs, height, width)``), as produced by
:func:`tktomo.blender_sim.scene.extract_slab_integrals`.
"""

from __future__ import annotations

import numpy as np

from tktomo.blender_sim.materials import wavelength, wavenumber
from tktomo.blender_sim.propagation import get_propagator

#: Quantities the output selector can return, in any combination.
VALID_OUTPUTS = ("attenuation", "phase", "complex")


def _validate_outputs(outputs: tuple[str, ...]) -> tuple[str, ...]:
    outputs = tuple(outputs)
    unknown = [name for name in outputs if name not in VALID_OUTPUTS]
    if unknown or not outputs:
        raise ValueError(f"outputs must be a non-empty subset of {VALID_OUTPUTS}, got {outputs}")
    return outputs


def transmission(delta_dz: np.ndarray, beta_dz: np.ndarray, energy_kev: float) -> np.ndarray:
    """Slab transmission t = exp(−i·k·∫δ dz − k·∫β dz)."""
    k = wavenumber(energy_kev)
    return np.exp(-k * np.asarray(beta_dz) - 1j * k * np.asarray(delta_dz))


def exit_wave(delta_dz: np.ndarray, beta_dz: np.ndarray, energy_kev: float) -> np.ndarray:
    """Exit wave from *total* line integrals (projection approximation)."""
    return transmission(delta_dz, beta_dz, energy_kev)


def multislice_wave(
    slab_delta_dz: np.ndarray,
    slab_beta_dz: np.ndarray,
    *,
    energy_kev: float,
    pixel_size: float,
    slice_spacing: float,
    kernel: str = "fresnel",
) -> np.ndarray:
    """Multislice / beam-propagation through the sample; returns the exit-plane wave.

    Slabs are ordered entrance → exit. Each slab transmits, then a near-field
    ``kernel`` propagates Δz to the next slab plane (n_slabs − 1 propagations), so a
    single slab reduces exactly to the projection-approximation exit wave.
    """
    slab_delta_dz = np.asarray(slab_delta_dz)
    slab_beta_dz = np.asarray(slab_beta_dz)
    if slab_delta_dz.shape != slab_beta_dz.shape or slab_delta_dz.ndim != 3:
        raise ValueError(
            "slab integrals must both have shape (n_slabs, height, width); got "
            f"{slab_delta_dz.shape} and {slab_beta_dz.shape}"
        )
    propagator = get_propagator(kernel)
    if not propagator.near_field:
        raise ValueError(f"multislice slab steps need a near-field kernel, got {kernel!r}")
    lam = wavelength(energy_kev)
    n_slabs = slab_delta_dz.shape[0]
    psi = np.ones(slab_delta_dz.shape[1:], dtype=complex)
    for index in range(n_slabs):
        psi = psi * transmission(slab_delta_dz[index], slab_beta_dz[index], energy_kev)
        if index < n_slabs - 1:
            psi = propagator.propagate(
                psi, distance=slice_spacing, wavelength=lam, pixel_size=pixel_size
            )
    return psi


def detector_wave(
    slab_delta_dz: np.ndarray,
    slab_beta_dz: np.ndarray,
    *,
    energy_kev: float,
    pixel_size: float,
    slice_spacing: float | None = None,
    distance: float = 0.0,
    method: str = "fresnel",
    slab_kernel: str = "fresnel",
    method_kwargs: dict | None = None,
) -> np.ndarray:
    """Wave at the detector plane: multislice through the sample + final hop.

    ``slice_spacing=None`` (or a single slab) means the projection approximation;
    ``distance`` is the free-space hop from the sample exit plane to the detector,
    carried by the propagator named ``method`` (the scene-level selector). The
    slab-to-slab steps always use the near-field ``slab_kernel``.
    """
    slab_delta_dz = np.asarray(slab_delta_dz)
    slab_beta_dz = np.asarray(slab_beta_dz)
    if slab_delta_dz.ndim == 2:  # accept collapsed integrals as one slab
        slab_delta_dz = slab_delta_dz[None]
        slab_beta_dz = slab_beta_dz[None]
    if slab_delta_dz.shape[0] > 1 and slice_spacing is None:
        raise ValueError("slice_spacing is required when more than one slab is given")
    psi = multislice_wave(
        slab_delta_dz,
        slab_beta_dz,
        energy_kev=energy_kev,
        pixel_size=pixel_size,
        slice_spacing=slice_spacing if slice_spacing is not None else 0.0,
        kernel=slab_kernel,
    )
    if distance:
        psi = get_propagator(method).propagate(
            psi,
            distance=distance,
            wavelength=wavelength(energy_kev),
            pixel_size=pixel_size,
            **(method_kwargs or {}),
        )
    return psi


def wave_outputs(
    psi_det: np.ndarray, outputs: tuple[str, ...] = ("attenuation",)
) -> dict[str, np.ndarray]:
    """Extract the selected quantities from a detector field.

    - ``attenuation`` — absorbance ∫μ dz recovered as −2·ln|ψ|;
    - ``phase`` — accumulated phase φ = −arg ψ (wrapped to (−π, π]; for unwrapped
      values without propagation use :func:`projection_outputs`);
    - ``complex`` — the field itself.
    """
    outputs = _validate_outputs(outputs)
    psi_det = np.asarray(psi_det, dtype=complex)
    result: dict[str, np.ndarray] = {}
    if "attenuation" in outputs:
        magnitude = np.abs(psi_det)
        result["attenuation"] = -2.0 * np.log(np.clip(magnitude, np.finfo(float).tiny, None))
    if "phase" in outputs:
        result["phase"] = -np.angle(psi_det)
    if "complex" in outputs:
        result["complex"] = psi_det
    return result


def projection_outputs(
    delta_dz: np.ndarray,
    beta_dz: np.ndarray,
    energy_kev: float,
    outputs: tuple[str, ...] = ("attenuation",),
) -> dict[str, np.ndarray]:
    """Outputs straight from total line integrals (no propagation, no phase wrap).

    - ``attenuation`` — ∫μ dz = 2k·∫β dz;
    - ``phase`` — φ = k·∫δ dz;
    - ``complex`` — the exit wave exp(−½∫μ dz − iφ).
    """
    outputs = _validate_outputs(outputs)
    k = wavenumber(energy_kev)
    result: dict[str, np.ndarray] = {}
    if "attenuation" in outputs:
        result["attenuation"] = 2.0 * k * np.asarray(beta_dz)
    if "phase" in outputs:
        result["phase"] = k * np.asarray(delta_dz)
    if "complex" in outputs:
        result["complex"] = exit_wave(delta_dz, beta_dz, energy_kev)
    return result
