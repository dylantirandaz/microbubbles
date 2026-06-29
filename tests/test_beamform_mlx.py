from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from ultratrace_ulm.beamform_core import (  # noqa: E402
    NeutralConfig,
    _beamform_single_transmit_mlx,
    beamform_iq,
)


def _tukey_weight(r_norm: float, alpha: float) -> float:
    if alpha <= 0.0:
        return 1.0
    if r_norm < 0.0 or r_norm > 1.0:
        return 0.0
    if r_norm <= 1.0 - alpha:
        return 1.0
    return float(0.5 - 0.5 * np.cos(np.pi * (1.0 - r_norm) / alpha))


def _reference_single_transmit(
    channel_data: np.ndarray,
    rx_coords_m: np.ndarray,
    scan_coords_m: np.ndarray,
    tx_wave_arrivals_s: np.ndarray,
    *,
    rx_start_s: float,
    sampling_freq_hz: float,
    f_number: float,
    sound_speed_m_s: float,
    modulation_freq_hz: float,
    tukey_alpha: float,
) -> np.ndarray:
    n_rx, n_samples, n_frames = channel_data.shape
    out = np.zeros((scan_coords_m.shape[0], n_frames), dtype=np.complex64)

    for scan_idx, voxel in enumerate(scan_coords_m):
        aperture_radius = voxel[2] / (2.0 * f_number)
        aperture_radius_sq = aperture_radius * aperture_radius
        for rx_idx in range(n_rx):
            delta = rx_coords_m[rx_idx] - voxel
            horizontal_sq = float(delta[0] * delta[0] + delta[1] * delta[1])
            if horizontal_sq > aperture_radius_sq:
                continue

            rx_distance = float(np.sqrt(horizontal_sq + delta[2] * delta[2]))
            tau = float(tx_wave_arrivals_s[scan_idx] + rx_distance / sound_speed_m_s)
            sample_idx = (tau - rx_start_s) * sampling_freq_hz
            if sample_idx < 0.0 or sample_idx > n_samples - 1:
                continue

            sample_floor = int(np.floor(sample_idx))
            sample_ceil = int(np.ceil(sample_idx))
            lerp_alpha = sample_idx - sample_floor
            sample = (
                channel_data[rx_idx, sample_floor]
                + (channel_data[rx_idx, sample_ceil] - channel_data[rx_idx, sample_floor])
                * lerp_alpha
            )

            if tukey_alpha > 0.0:
                r_norm = float(np.sqrt(horizontal_sq) / aperture_radius)
                sample = sample * _tukey_weight(r_norm, tukey_alpha)

            if modulation_freq_hz:
                sample = sample * np.exp(1j * 2.0 * np.pi * modulation_freq_hz * tau)

            out[scan_idx] += sample.astype(np.complex64)

    return out


def test_mlx_single_transmit_matches_reference():
    rng = np.random.default_rng(7)
    channel_data = (
        rng.normal(size=(4, 32, 3)) + 1j * rng.normal(size=(4, 32, 3))
    ).astype(np.complex64)
    rx_coords_m = np.array(
        [
            [-1.0e-3, 0.0, 0.0],
            [-0.3e-3, 0.0, 0.0],
            [0.3e-3, 0.0, 0.0],
            [1.0e-3, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scan_coords_m = np.array(
        [
            [-0.5e-3, 0.0, 4.0e-3],
            [0.0, 0.0, 6.0e-3],
            [1.5e-3, 0.0, 5.0e-3],
            [0.0, 0.4e-3, 7.0e-3],
            [2.0e-3, 0.0, 3.0e-3],
        ],
        dtype=np.float32,
    )
    tx_wave_arrivals_s = np.array([0.0, 0.2e-6, 0.1e-6, 0.3e-6, 0.0], dtype=np.float32)
    params = dict(
        rx_start_s=0.0,
        sampling_freq_hz=4.0e6,
        f_number=0.9,
        sound_speed_m_s=1540.0,
        modulation_freq_hz=2.0e6,
        tukey_alpha=0.5,
    )

    actual = _beamform_single_transmit_mlx(
        channel_data,
        rx_coords_m,
        scan_coords_m,
        tx_wave_arrivals_s,
        scan_chunk=2,
        receive_chunk=2,
        frame_chunk=1,
        **params,
    )
    expected = _reference_single_transmit(
        channel_data,
        rx_coords_m,
        scan_coords_m,
        tx_wave_arrivals_s,
        **params,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_beamform_iq_mlx_returns_compound_grid_shape():
    rng = np.random.default_rng(11)
    iq = (
        rng.normal(size=(2, 1, 1, 4, 32)) + 1j * rng.normal(size=(2, 1, 1, 4, 32))
    ).astype(np.complex64)
    config = NeutralConfig(
        sampling_rate_hz=4.0e6,
        tx_freq_hz=5.0e6,
        speed_of_sound_m_s=1540.0,
        start_depth_m=4.0e-3,
        end_depth_m=4.5e-3,
        rx_offset_s=0.0,
        rx_el_bf_time_offset_s=0.0,
        element_pitch_x_m=0.2e-3,
        element_pitch_y_m=0.2e-3,
        ana_mux=False,
        num_elev_planes=1,
        row_index=-1,
        mean_subtract_channels=False,
        large_fov=False,
        xlarge_fov=False,
        z_coarseness=1.0,
        x_coarseness=1.0,
        f_number=0.9,
        y_offset=0.0,
        y_tilt_angle_deg=0.0,
        tx_az_half_aperture=2,
        elev_switch_matrix_mode=0,
        en_saturation=True,
        num_noise_loops=0,
    )

    compound, grid = beamform_iq(
        iq,
        np.zeros((1, 4), dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
        config,
        backend="mlx",
        scan_chunk=3,
        receive_chunk=2,
        frame_chunk=1,
    )

    assert compound.shape == (2, 1, grid.depth_pixels, grid.width_pixels)
    assert compound.dtype == np.complex64
    assert np.isfinite(compound).all()
