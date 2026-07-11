"""Physics-layer tests for tktomo.blender_sim — must run with no Blender installed."""

import sys

import numpy as np
import pytest

from tktomo.blender_sim import (
    MaterialSet,
    available_propagators,
    beta_from_mu,
    detector_wave,
    exit_wave,
    get_propagator,
    mu_from_beta,
    multislice_wave,
    normalize_orientations,
    projection_outputs,
    wave_outputs,
    wavelength,
    wavenumber,
)

ENERGY = 17.0  # keV
LAM = wavelength(ENERGY)


def test_import_does_not_pull_in_bpy():
    import tktomo.blender_sim  # noqa: F401
    import tktomo.blender_sim.scene  # noqa: F401  (bpy is lazy inside functions)
    import tktomo.blender_sim.ui_panel  # noqa: F401
    import tktomo.blender_sim.viewer  # noqa: F401

    assert "bpy" not in sys.modules


# -- materials ---------------------------------------------------------------


def test_mu_beta_roundtrip():
    mu = mu_from_beta(3e-9, ENERGY)
    assert mu == pytest.approx(4 * np.pi * 3e-9 / LAM)
    assert beta_from_mu(mu, ENERGY) == pytest.approx(3e-9)


def test_set_mu_updates_beta_not_energy():
    materials = MaterialSet(ENERGY)
    materials.add("gold", delta=1e-6, beta=1e-8)
    materials.set_mu("gold", 250.0)
    assert materials.energy_kev == ENERGY  # E untouched
    assert materials["gold"].beta == pytest.approx(beta_from_mu(250.0, ENERGY))
    assert materials["gold"].delta == 1e-6  # δ untouched
    assert materials.mu("gold") == pytest.approx(250.0)


def test_energy_edit_rescales_by_dispersion_laws():
    materials = MaterialSet(10.0)
    materials.add("m", delta=2e-6, beta=4e-9)
    mu_before = materials.mu("m")
    materials.energy_kev = 20.0  # double the energy
    assert materials["m"].delta == pytest.approx(2e-6 / 4)  # δ ∝ 1/E²
    assert materials["m"].beta == pytest.approx(4e-9 / 8)  # β ∝ 1/E³
    assert materials.mu("m") == pytest.approx(mu_before / 4)  # μ ∝ 1/E²


def test_delta_edit_touches_delta_only():
    materials = MaterialSet(ENERGY)
    materials.add("m", delta=1e-6, beta=1e-9)
    mu_before = materials.mu("m")
    materials.set_delta("m", 9e-6)
    assert materials["m"].delta == 9e-6
    assert materials["m"].beta == 1e-9
    assert materials.mu("m") == pytest.approx(mu_before)


# -- propagators -------------------------------------------------------------


def _gaussian(n=64, pixel=1e-6, waist=8e-6):
    x = (np.arange(n) - n / 2) * pixel
    xx, yy = np.meshgrid(x, x)
    return np.exp(-(xx**2 + yy**2) / waist**2).astype(complex)


def test_registry_lists_all_methods():
    names = available_propagators()
    for expected in ("fresnel", "angular_spectrum", "fraunhofer", "fresnel_scaling"):
        assert expected in names


def test_unknown_propagator_raises():
    with pytest.raises(KeyError, match="Unknown propagator"):
        get_propagator("nope")


@pytest.mark.parametrize("name", ["fresnel", "angular_spectrum"])
def test_near_field_zero_distance_is_identity(name):
    psi = _gaussian()
    out = get_propagator(name).propagate(
        psi, distance=0.0, wavelength=LAM, pixel_size=1e-6
    )
    np.testing.assert_allclose(out, psi)


@pytest.mark.parametrize("name", ["fresnel", "angular_spectrum"])
def test_near_field_conserves_energy(name):
    psi = _gaussian()
    out = get_propagator(name).propagate(
        psi, distance=5e-3, wavelength=LAM, pixel_size=1e-6
    )
    assert np.sum(np.abs(out) ** 2) == pytest.approx(np.sum(np.abs(psi) ** 2), rel=1e-9)


@pytest.mark.parametrize("name", ["fresnel", "angular_spectrum"])
def test_plane_wave_is_invariant_up_to_global_phase(name):
    psi = np.ones((32, 32), dtype=complex)
    out = get_propagator(name).propagate(
        psi, distance=1e-2, wavelength=LAM, pixel_size=1e-6
    )
    np.testing.assert_allclose(np.abs(out), 1.0, atol=1e-12)
    # uniform phase across the field
    np.testing.assert_allclose(np.angle(out / out[0, 0]), 0.0, atol=1e-9)


def test_fresnel_matches_angular_spectrum_in_paraxial_regime():
    psi = _gaussian()
    kwargs = dict(distance=2e-3, wavelength=LAM, pixel_size=1e-6)
    fresnel = get_propagator("fresnel").propagate(psi, **kwargs)
    angular = get_propagator("angular_spectrum").propagate(psi, **kwargs)
    # allow a global phase offset between the two conventions
    correlation = np.vdot(fresnel, angular) / (
        np.linalg.norm(fresnel) * np.linalg.norm(angular)
    )
    assert abs(correlation) == pytest.approx(1.0, abs=1e-6)


def test_fraunhofer_requires_positive_distance_and_conserves_power():
    psi = _gaussian()
    propagator = get_propagator("fraunhofer")
    with pytest.raises(ValueError):
        propagator.propagate(psi, distance=0.0, wavelength=LAM, pixel_size=1e-6)
    distance, pixel = 1.0, 1e-6
    out = propagator.propagate(psi, distance=distance, wavelength=LAM, pixel_size=pixel)
    out_pixel = LAM * distance / (psi.shape[0] * pixel)
    power_in = np.sum(np.abs(psi) ** 2) * pixel**2
    power_out = np.sum(np.abs(out) ** 2) * out_pixel**2
    assert power_out == pytest.approx(power_in, rel=1e-9)


def test_fresnel_scaling_reduces_to_fresnel_for_distant_source():
    psi = _gaussian()
    r1, r2 = 1e6, 5e-3  # effectively parallel illumination
    scaled = get_propagator("fresnel_scaling").propagate(
        psi, wavelength=LAM, pixel_size=1e-6, r1=r1, r2=r2
    )
    z_eff = r1 * r2 / (r1 + r2)
    plain = get_propagator("fresnel").propagate(
        psi, distance=z_eff, wavelength=LAM, pixel_size=1e-6
    )
    magnification = (r1 + r2) / r1
    np.testing.assert_allclose(scaled, plain / magnification, rtol=1e-9)
    assert magnification == pytest.approx(1.0, abs=1e-8)


def test_fresnel_scaling_requires_cone_geometry():
    psi = _gaussian(16)
    propagator = get_propagator("fresnel_scaling")
    with pytest.raises(ValueError, match="r1"):
        propagator.propagate(psi, distance=1e-2, wavelength=LAM, pixel_size=1e-6)
    with pytest.raises(ValueError, match="r2"):
        propagator.propagate(psi, wavelength=LAM, pixel_size=1e-6, r1=0.5)


def test_fresnel_scaling_applies_intensity_demagnification():
    psi = np.ones((16, 16), dtype=complex)
    r1 = r2 = 0.5  # M = 2
    out = get_propagator("fresnel_scaling").propagate(
        psi, wavelength=LAM, pixel_size=1e-6, r1=r1, r2=r2
    )
    np.testing.assert_allclose(np.abs(out) ** 2, 0.25, atol=1e-12)


# -- exit wave / multislice ---------------------------------------------------


def test_exit_wave_magnitude_and_phase():
    delta_dz = np.full((8, 8), 3e-12)  # φ = k·∫δdz ≈ 0.26 rad — safely unwrapped
    beta_dz = np.full((8, 8), 2e-12)
    k = wavenumber(ENERGY)
    psi = exit_wave(delta_dz, beta_dz, ENERGY)
    np.testing.assert_allclose(np.abs(psi), np.exp(-k * beta_dz))
    np.testing.assert_allclose(-np.angle(psi), k * delta_dz)  # φ = −arg ψ


def test_single_slab_multislice_is_projection_approximation():
    delta = np.random.default_rng(0).random((1, 16, 16)) * 1e-9
    beta = np.random.default_rng(1).random((1, 16, 16)) * 1e-11
    psi_multi = multislice_wave(
        delta, beta, energy_kev=ENERGY, pixel_size=1e-6, slice_spacing=1.0
    )
    np.testing.assert_allclose(psi_multi, exit_wave(delta[0], beta[0], ENERGY))


def test_uniform_slabs_multislice_equals_product_of_transmissions():
    # plane-wave symmetry: propagation between uniform slabs is a global phase,
    # so the magnitude and transverse phase match the collapsed exit wave.
    delta = np.full((4, 8, 8), 1e-9)
    beta = np.full((4, 8, 8), 5e-12)
    psi = multislice_wave(
        delta, beta, energy_kev=ENERGY, pixel_size=1e-6, slice_spacing=1e-3
    )
    collapsed = exit_wave(delta.sum(axis=0), beta.sum(axis=0), ENERGY)
    np.testing.assert_allclose(np.abs(psi), np.abs(collapsed), rtol=1e-9)
    relative = psi / collapsed
    np.testing.assert_allclose(relative, relative[0, 0], rtol=1e-9)


def test_detector_wave_without_distance_is_exit_wave():
    delta = np.random.default_rng(2).random((16, 16)) * 1e-9
    beta = np.random.default_rng(3).random((16, 16)) * 1e-11
    psi = detector_wave(delta, beta, energy_kev=ENERGY, pixel_size=1e-6)
    np.testing.assert_allclose(psi, exit_wave(delta, beta, ENERGY))


def test_multiple_slabs_require_slice_spacing():
    slabs = np.zeros((3, 8, 8))
    with pytest.raises(ValueError, match="slice_spacing"):
        detector_wave(slabs, slabs, energy_kev=ENERGY, pixel_size=1e-6)


def test_multislice_rejects_far_field_slab_kernel():
    slabs = np.zeros((2, 8, 8))
    with pytest.raises(ValueError, match="near-field"):
        multislice_wave(
            slabs, slabs, energy_kev=ENERGY, pixel_size=1e-6,
            slice_spacing=1e-3, kernel="fraunhofer",
        )


# -- outputs ------------------------------------------------------------------


def test_projection_outputs_match_wave_outputs_for_small_phase():
    delta_dz = np.full((8, 8), 1e-12)  # small enough that −arg ψ does not wrap
    beta_dz = np.full((8, 8), 1e-12)
    direct = projection_outputs(delta_dz, beta_dz, ENERGY, ("attenuation", "phase", "complex"))
    from_wave = wave_outputs(direct["complex"], ("attenuation", "phase"))
    np.testing.assert_allclose(direct["attenuation"], from_wave["attenuation"], rtol=1e-9)
    np.testing.assert_allclose(direct["phase"], from_wave["phase"], rtol=1e-9)
    k = wavenumber(ENERGY)
    np.testing.assert_allclose(direct["attenuation"], 2 * k * beta_dz)
    np.testing.assert_allclose(direct["phase"], k * delta_dz)


def test_outputs_selector_rejects_unknown_and_empty():
    with pytest.raises(ValueError):
        projection_outputs(np.zeros((2, 2)), np.zeros((2, 2)), ENERGY, ("intensity",))
    with pytest.raises(ValueError):
        wave_outputs(np.ones((2, 2), dtype=complex), ())


# -- orientations --------------------------------------------------------------


def test_scalar_orientations_are_z_rotations_with_angles_passthrough():
    angles = np.array([0.0, np.pi / 2])
    matrices, out_angles = normalize_orientations(angles)
    np.testing.assert_allclose(out_angles, angles)
    np.testing.assert_allclose(matrices[0], np.eye(3), atol=1e-15)
    # 90° about Z sends +X to +Y
    np.testing.assert_allclose(matrices[1] @ [1, 0, 0], [0, 1, 0], atol=1e-15)


def test_euler_z_only_matches_scalar_form():
    theta = 0.7
    from_euler, _ = normalize_orientations([[0.0, 0.0, theta]])
    from_scalar, _ = normalize_orientations([theta])
    np.testing.assert_allclose(from_euler, from_scalar)


def test_euler_xy_tilt_axis_and_z_scans_about_it():
    # x/y set the scan axis; z rotates about it: the tilted axis (the image of
    # world Z) must not depend on the scan angle.
    tilt_x, tilt_y = 0.3, -0.2
    scans = [0.0, 0.9, 2.4]
    matrices, angles = normalize_orientations([[tilt_x, tilt_y, s] for s in scans])
    np.testing.assert_allclose(angles, scans)  # scan column passes through
    axes = matrices @ [0.0, 0.0, 1.0]
    np.testing.assert_allclose(axes[1], axes[0], atol=1e-15)
    np.testing.assert_allclose(axes[2], axes[0], atol=1e-15)
    # and with zero tilt the axis is world Z itself
    untilted, _ = normalize_orientations([[0.0, 0.0, 1.1]])
    np.testing.assert_allclose(untilted[0] @ [0, 0, 1], [0, 0, 1], atol=1e-15)


def test_matrix_orientations_pass_through():
    matrix = np.eye(3)[None]
    matrices, angles = normalize_orientations(matrix)
    np.testing.assert_allclose(matrices, matrix)
    np.testing.assert_allclose(angles, [0.0])


def test_bad_orientation_shape_raises():
    with pytest.raises(ValueError, match="orientations"):
        normalize_orientations(np.zeros((2, 4)))
