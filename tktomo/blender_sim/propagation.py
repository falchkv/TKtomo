"""Free-space propagators and their registry.

A :class:`Propagator` carries a complex field sampled on a regular grid over a
distance. Methods register under a short name (mirroring the aligner / recon-backend
registries) so callers select them by string:

- ``fresnel`` — paraxial near-field transfer function (the default);
- ``angular_spectrum`` — exact non-paraxial near-field spectral method;
- ``fraunhofer`` — far-field single FFT (large-z only, not usable as a slab step);
- ``fresnel_scaling`` — cone-beam effective propagation via the Fresnel scaling
  theorem (M = (R₁+R₂)/R₁, z_eff = R₁R₂/(R₁+R₂)).

Sign convention: forward carrier ``e^{+ikz}``, matching the slab transmission
``exp(−i·k·δ·Δz − k·β·Δz)`` in :mod:`~tktomo.blender_sim.multislice`.

Units: lengths in metres. ``pixel_size`` is the pitch of the *input* field grid.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Propagator(Protocol):
    #: Registry name.
    name: str
    #: True if valid at short distances (usable as the multislice slab step).
    near_field: bool

    def propagate(
        self,
        psi: np.ndarray,
        *,
        distance: float,
        wavelength: float,
        pixel_size: float,
        **kwargs,
    ) -> np.ndarray:
        """Return the field after propagating ``distance`` metres."""
        ...


_REGISTRY: dict[str, Propagator] = {}


def register_propagator(propagator: Propagator) -> Propagator:
    """Register a propagator instance under ``propagator.name``. Returns it unchanged."""
    if not getattr(propagator, "name", None):
        raise ValueError("Propagator must define a non-empty 'name'.")
    _REGISTRY[propagator.name] = propagator
    return propagator


def available_propagators() -> list[str]:
    """Names of registered propagators, for populating a method dropdown."""
    return sorted(_REGISTRY)


def get_propagator(name: str = "fresnel") -> Propagator:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown propagator {name!r}. Available: {available_propagators()}"
        ) from None


def _frequency_grids(shape: tuple[int, int], pixel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """fftfreq grids (fy, fx) in 1/m for a (height, width) field."""
    fy = np.fft.fftfreq(shape[0], d=pixel_size)
    fx = np.fft.fftfreq(shape[1], d=pixel_size)
    return fy[:, None], fx[None, :]


class FresnelPropagator:
    """Paraxial near-field transfer function H = e^{ikz}·exp(−iπλz(fx²+fy²))."""

    name = "fresnel"
    near_field = True

    def propagate(self, psi, *, distance, wavelength, pixel_size, **kwargs):
        psi = np.asarray(psi, dtype=complex)
        if distance == 0:
            return psi.copy()
        fy, fx = _frequency_grids(psi.shape, pixel_size)
        carrier = np.exp(1j * 2 * np.pi / wavelength * distance)
        transfer = carrier * np.exp(-1j * np.pi * wavelength * distance * (fx**2 + fy**2))
        return np.fft.ifft2(np.fft.fft2(psi) * transfer)


class AngularSpectrumPropagator:
    """Exact near-field spectral method H = exp(ikz·√(1 − (λfx)² − (λfy)²)).

    Evanescent components (negative argument under the root) are zeroed.
    """

    name = "angular_spectrum"
    near_field = True

    def propagate(self, psi, *, distance, wavelength, pixel_size, **kwargs):
        psi = np.asarray(psi, dtype=complex)
        if distance == 0:
            return psi.copy()
        fy, fx = _frequency_grids(psi.shape, pixel_size)
        argument = 1.0 - (wavelength * fx) ** 2 - (wavelength * fy) ** 2
        propagating = argument > 0
        transfer = np.zeros(np.broadcast_shapes(fy.shape, fx.shape), dtype=complex)
        transfer[propagating] = np.exp(
            1j
            * (2 * np.pi / wavelength)
            * distance
            * np.sqrt(np.broadcast_to(argument, transfer.shape)[propagating])
        )
        return np.fft.ifft2(np.fft.fft2(psi) * transfer)


class FraunhoferPropagator:
    """Far-field single-FFT propagator (valid only for large z).

    Output is sampled on the far-field grid x' = λ·z·f (centred), so the output
    pixel pitch is ``λ·z / (n · pixel_size)`` per axis. Single-stage: not usable
    as a multislice slab step.
    """

    name = "fraunhofer"
    near_field = False

    def propagate(self, psi, *, distance, wavelength, pixel_size, **kwargs):
        psi = np.asarray(psi, dtype=complex)
        if distance <= 0:
            raise ValueError("fraunhofer requires a positive propagation distance")
        k = 2 * np.pi / wavelength
        fy = np.fft.fftshift(np.fft.fftfreq(psi.shape[0], d=pixel_size))[:, None]
        fx = np.fft.fftshift(np.fft.fftfreq(psi.shape[1], d=pixel_size))[None, :]
        x = wavelength * distance * fx
        y = wavelength * distance * fy
        spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psi))) * pixel_size**2
        prefactor = np.exp(1j * k * distance) / (1j * wavelength * distance)
        return prefactor * np.exp(1j * k * (x**2 + y**2) / (2 * distance)) * spectrum


class FresnelScalingPropagator:
    """Cone-beam effective propagation via the Fresnel scaling theorem.

    From the cone geometry (``r1`` = source→sample, ``r2`` = sample→detector, both
    metres) it forms the magnification M = (r1+r2)/r1 and the effective distance
    z_eff = r1·r2/(r1+r2), runs a parallel-beam *near-field* propagator (``kernel``)
    at z_eff on the input (sample-plane) grid, then scales the amplitude by 1/M so
    intensity carries the 1/M² factor. The output is sampled at ``M · pixel_size``
    on the detector. ``r2`` defaults to ``distance`` if not given. Pair with a
    perspective (cone) camera for physical consistency.
    """

    name = "fresnel_scaling"
    near_field = False

    def propagate(
        self,
        psi,
        *,
        distance=None,
        wavelength,
        pixel_size,
        r1=None,
        r2=None,
        kernel="fresnel",
        **kwargs,
    ):
        if r1 is None:
            raise ValueError("fresnel_scaling requires r1 (source→sample distance)")
        if r2 is None:
            r2 = distance
        if r2 is None:
            raise ValueError(
                "fresnel_scaling requires r2 (sample→detector distance) or distance"
            )
        near_field_kernel = get_propagator(kernel)
        if not near_field_kernel.near_field:
            raise ValueError(f"fresnel_scaling kernel must be near-field, got {kernel!r}")
        magnification = (r1 + r2) / r1
        z_eff = r1 * r2 / (r1 + r2)
        propagated = near_field_kernel.propagate(
            psi, distance=z_eff, wavelength=wavelength, pixel_size=pixel_size
        )
        return propagated / magnification


register_propagator(FresnelPropagator())
register_propagator(AngularSpectrumPropagator())
register_propagator(FraunhoferPropagator())
register_propagator(FresnelScalingPropagator())
