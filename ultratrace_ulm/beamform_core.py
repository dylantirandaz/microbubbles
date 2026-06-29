"""Standalone beamforming core for neutral ultratrace data.

This module is a sanitized, dependency-free port of the pieces of the reference
imaging stack required to beamform demodulated IQ data with the
``mach`` CUDA kernel, plus a native MLX backend for Apple Silicon. It
deliberately contains NO external imaging-SDK import and does NOT decode raw
frames -- it consumes the neutral ultratrace schema written by
``export_neutral_ultratrace.py`` (see that converter / the README for the
schema).

The numerics here mirror the reference:
  * ``imaging/preprocess.py``                  -> :func:`preprocess`
  * ``imaging/beamform.py::compute_grid_params`` /
    ``BeamformingParameters.create`` grid math -> :func:`build_grid`
  * ``imaging/beamform.py::_prepare_tx_delays_for_row`` -> :func:`prepare_tx_delays_for_row`
  * ``imaging/mach_beamform.py``               -> :func:`beamform_iq_mach`
  * ``misc/ulm/beamform_tgc_n4.py``            -> :func:`compute_global_tgc`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# mach is a GPU-only optional dependency. Guard the import so the package still
# imports (and ``beamform --help`` works) without it.
try:  # pragma: no cover - exercised only on GPU hosts
    from mach.experimental import beamform as _mach_beamform

    MACH_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means mach is unavailable
    _mach_beamform = None
    MACH_AVAILABLE = False

# MLX is an optional Apple Silicon backend. Guard the import for the same reason
# as MACH: tracking/viewer imports should work without any GPU package installed.
try:  # pragma: no cover - import availability depends on host platform
    import mlx.core as _mx

    MLX_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means mlx is unavailable
    _mx = None
    MLX_AVAILABLE = False

BEAMFORM_BACKENDS = ("auto", "mach", "mlx")


def resolve_beamform_backend(backend: str = "mach") -> str:
    """Resolve ``auto`` and validate requested beamforming backend."""
    backend = backend.lower()
    if backend not in BEAMFORM_BACKENDS:
        choices = ", ".join(BEAMFORM_BACKENDS)
        raise ValueError(f"Unknown beamforming backend '{backend}'. Choose one of: {choices}")
    if backend == "auto":
        if MACH_AVAILABLE:
            return "mach"
        if MLX_AVAILABLE:
            return "mlx"
        raise ImportError(
            "No beamforming backend is available. Install the CUDA backend with "
            "pip install 'ultratrace-ulm-pipeline[mach]' or the Apple Silicon "
            "backend with pip install 'ultratrace-ulm-pipeline[mlx]'."
        )
    if backend == "mach" and not MACH_AVAILABLE:
        raise ImportError(
            "mach is not importable. MACH beamforming requires the GPU-only "
            "'mach-beamform' package. Install with: pip install "
            "'ultratrace-ulm-pipeline[mach]'"
        )
    if backend == "mlx" and not MLX_AVAILABLE:
        raise ImportError(
            "mlx is not importable. MLX beamforming requires Apple Silicon/macOS "
            "and the optional 'mlx' package. Install with: pip install "
            "'ultratrace-ulm-pipeline[mlx]'"
        )
    return backend


# --------------------------------------------------------------------------- #
# Neutral config container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NeutralConfig:
    """Scalar acquisition parameters needed to beamform neutral IQ.

    Mirrors the ``/config`` attrs of the neutral ultratrace schema. Field names
    match the schema attr names.
    """

    sampling_rate_hz: float
    tx_freq_hz: float
    speed_of_sound_m_s: float
    start_depth_m: float
    end_depth_m: float
    rx_offset_s: float
    rx_el_bf_time_offset_s: float
    element_pitch_x_m: float
    element_pitch_y_m: float
    ana_mux: bool
    num_elev_planes: int
    row_index: int
    mean_subtract_channels: bool
    large_fov: bool
    xlarge_fov: bool
    z_coarseness: float
    x_coarseness: float
    f_number: float
    y_offset: float
    y_tilt_angle_deg: float
    tx_az_half_aperture: int
    elev_switch_matrix_mode: int
    en_saturation: bool
    num_noise_loops: int

    @classmethod
    def from_h5_attrs(cls, attrs) -> "NeutralConfig":
        def get(name, default=None):
            if name in attrs:
                return attrs[name]
            if default is not None:
                return default
            raise KeyError(f"Neutral config missing required attr '{name}'")

        return cls(
            sampling_rate_hz=float(get("sampling_rate_hz")),
            tx_freq_hz=float(get("tx_freq_hz")),
            speed_of_sound_m_s=float(get("speed_of_sound_m_s")),
            start_depth_m=float(get("start_depth_m")),
            end_depth_m=float(get("end_depth_m")),
            rx_offset_s=float(get("rx_offset_s")),
            rx_el_bf_time_offset_s=float(get("rx_el_bf_time_offset_s", 0.0)),
            element_pitch_x_m=float(get("element_pitch_x_m")),
            element_pitch_y_m=float(get("element_pitch_y_m")),
            ana_mux=bool(get("ana_mux")),
            num_elev_planes=int(get("num_elev_planes")),
            row_index=int(get("row_index", -1)),
            mean_subtract_channels=bool(get("mean_subtract_channels")),
            large_fov=bool(get("large_fov")),
            xlarge_fov=bool(get("xlarge_fov", False)),
            z_coarseness=float(get("z_coarseness")),
            x_coarseness=float(get("x_coarseness")),
            f_number=float(get("f_number")),
            y_offset=float(get("y_offset", 0.0)),
            y_tilt_angle_deg=float(get("y_tilt_angle_deg", 0.0)),
            tx_az_half_aperture=int(get("tx_az_half_aperture")),
            elev_switch_matrix_mode=int(get("elev_switch_matrix_mode", 0)),
            en_saturation=bool(get("en_saturation", True)),
            num_noise_loops=int(get("num_noise_loops", 0)),
        )


# Element-pitch reference constant (reference DEFAULT_ELEMENT_PITCH_M). The
# create()-grid x-span uses this nominal pitch, not the per-config pitch.
DEFAULT_ELEMENT_PITCH_M = 208e-6


# --------------------------------------------------------------------------- #
# I/Q preprocessing (port of reference preprocess)
# --------------------------------------------------------------------------- #
def _unwrap_12bit(x: np.ndarray) -> np.ndarray:
    x_i32 = x.astype(np.int32, copy=False)
    d = np.diff(x_i32, axis=-1)
    jumps = np.round(d / 4096).astype(np.int32)
    correction = np.cumsum(jumps, axis=-1) * 4096
    out = x_i32.copy()
    out[..., 1:] -= correction
    return out


def _unwrap_iq_12bit(iq: np.ndarray) -> np.ndarray:
    return (_unwrap_12bit(np.real(iq)) + 1j * _unwrap_12bit(np.imag(iq))).astype(
        np.complex64, copy=False
    )


def preprocess(
    iq_signals: np.ndarray,  # (loops, angles, rows, cols, time)
    config: NeutralConfig,
    row_index: int | None = -1,
    mean_subtract_channels: bool = False,
) -> np.ndarray:
    """Sanitized port of the reference preprocess (numpy-only path).

    Returns ``(frames, angles, rows_kept, cols, time)``. Only the non-elevational
    focusing branches are ported (el_focus_depth_m is not supported here).
    """
    if not config.en_saturation:
        iq_signals = _unwrap_iq_12bit(iq_signals)

    # tx delays and iq data are flipped relative to each other along channels.
    iq_signals = np.flip(iq_signals, axis=3)

    loops_total = iq_signals.shape[0]
    if loops_total - config.num_noise_loops <= 0:
        raise RuntimeError("No useful loops available after removing noise frames.")
    iq = iq_signals[config.num_noise_loops:]

    num_rows = iq.shape[2]
    if row_index == -2:
        iq = np.sum(iq, axis=2)[:, :, np.newaxis, :, :]
    elif row_index == -1 or row_index is None:
        pass  # keep all rows
    else:
        iq = iq[:, :, row_index: row_index + 1, :, :]

    if mean_subtract_channels:
        iq = iq - iq.mean(axis=3, keepdims=True)

    return np.ascontiguousarray(iq)


# --------------------------------------------------------------------------- #
# Imaging grid (port of BeamformingParameters.create grid math)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid:
    """Imaging grid; arrays have shape (depth=z, height=elev, width=x)."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def depth_pixels(self) -> int:
        return self.x.shape[0]

    @property
    def height_pixels(self) -> int:
        return self.x.shape[1]

    @property
    def width_pixels(self) -> int:
        return self.x.shape[2]


def build_grid(config: NeutralConfig) -> Grid:
    """Build the imaging grid, mirroring ``BeamformingParameters.create``.

    Z/X use lambda/2 spacing driven by the TX carrier (scaled by coarseness),
    independent of decimation / element pitch. X half-span is
    ``DEFAULT_ELEMENT_PITCH_M * half_aperture * fov_mult``. Y spans
    ``+-(DEFAULT_ELEMENT_PITCH_Y_M * 4 / 2**elev_mode)`` with ``num_elev_planes``.
    Grid arrays are ordered (z, elev, x) to match the reference output.
    """
    c = config.speed_of_sound_m_s
    lambda_over_2 = c / (2.0 * config.tx_freq_hz)

    rx_depth_offset = config.rx_el_bf_time_offset_s * c / 2.0
    z_start = config.start_depth_m + rx_depth_offset
    z_end = config.end_depth_m + rx_depth_offset
    z_pixels = max(1, int(round((z_end - z_start) / (lambda_over_2 * config.z_coarseness))))

    if config.xlarge_fov:
        fov_mult = 4
    elif config.large_fov:
        fov_mult = 2
    else:
        fov_mult = 1

    x_half_span = DEFAULT_ELEMENT_PITCH_M * config.tx_az_half_aperture * fov_mult
    x_pixels = max(1, int(round((2 * x_half_span) / (lambda_over_2 * config.x_coarseness))))

    num_y_planes = config.num_elev_planes
    half_height = DEFAULT_ELEMENT_PITCH_M * 8 * 4 / (2 ** config.elev_switch_matrix_mode)
    if num_y_planes <= 1:
        y_start = y_end = config.y_offset
        num_y_planes = 1
    else:
        y_start = -half_height + config.y_offset
        y_end = half_height + config.y_offset

    x = np.linspace(-x_half_span, x_half_span, x_pixels)
    y = np.linspace(y_start, y_end, num_y_planes)
    z = np.linspace(z_start, z_end, z_pixels)

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    if config.y_tilt_angle_deg:
        Y = Y + Z * np.tan(np.deg2rad(config.y_tilt_angle_deg))
    return Grid(X.astype(np.float64), Y.astype(np.float64), Z.astype(np.float64))


# --------------------------------------------------------------------------- #
# Transmit delays (port of _prepare_tx_delays_for_row)
# --------------------------------------------------------------------------- #
def prepare_tx_delays_for_row(
    tx_delays_list,  # list/array of per-angle arrays
    num_channels: int,
    row_index: int = 0,
) -> np.ndarray:
    """Normalize per-angle azimuthal transmit delays to (angles, num_channels)."""
    angles = len(tx_delays_list)
    out = np.zeros((angles, num_channels), dtype=np.float64)
    for ai in range(angles):
        d = np.asarray(tx_delays_list[ai]).astype(np.float64)
        if d.size == num_channels:
            out[ai] = d
        elif d.size == 2 * num_channels:
            out[ai] = d[:num_channels] if row_index == 0 else d[-num_channels:]
        elif d.size > num_channels:
            start = (d.size - num_channels) // 2
            out[ai] = d[start: start + num_channels]
        else:
            out[ai, : d.size] = d
    return out


# --------------------------------------------------------------------------- #
# MACH beamform (port of the reference MACH wrapper)
# --------------------------------------------------------------------------- #
def beamform_iq_mach(
    iq_frames: np.ndarray,  # (loops, angles, rows, cols, time) complex
    tx_delays: np.ndarray,  # (angles, channels) seconds
    tx_delays_elev: np.ndarray,  # (angles, rows) seconds
    config: NeutralConfig,
) -> tuple[np.ndarray, Grid]:
    """Beamform neutral IQ with the mach experimental kernel.

    Returns ``(compound_image, grid)`` where compound_image is
    ``(frames, elev, z, x)`` complex64 and ``grid`` arrays are ``(z, elev, x)``.
    """
    resolve_beamform_backend("mach")
    import cupy as cp

    iq = preprocess(
        iq_frames,
        config,
        row_index=config.row_index,
        mean_subtract_channels=config.mean_subtract_channels,
    )
    num_frames, num_angles, num_rows, num_cols, num_time = iq.shape

    grid = build_grid(config)

    # Azimuthal + elevational transmit delays normalized to the kept rows/cols.
    txd = prepare_tx_delays_for_row(
        list(tx_delays), num_cols, row_index=config.row_index
    )
    txd_elev = np.asarray(tx_delays_elev, dtype=np.float64)
    if config.row_index is not None and config.row_index >= 0:
        txd_elev = txd_elev[:, [config.row_index]]
    elif num_rows == 1:
        txd_elev = txd_elev[:, [txd_elev.shape[1] // 2]]
    elif txd_elev.shape[1] != num_rows:
        # Align metadata rows to kept rows by even subsampling.
        if txd_elev.shape[1] > num_rows:
            stride = max(1, txd_elev.shape[1] // num_rows)
            txd_elev = txd_elev[:, ::stride][:, :num_rows]
        else:
            pad = np.zeros((txd_elev.shape[0], num_rows), dtype=np.float64)
            pad[:, : txd_elev.shape[1]] = txd_elev
            txd_elev = pad

    # Receive element positions (n_rx=rows*cols, 3) in meters.
    x_channels = (np.arange(num_cols, dtype=np.float32) - (num_cols - 1) / 2) * config.element_pitch_x_m
    y_rows = (np.arange(num_rows, dtype=np.float32) - (num_rows - 1) / 2) * config.element_pitch_y_m
    rx_coords_np = np.stack(
        (
            np.tile(x_channels, num_rows),
            np.repeat(y_rows, num_cols),
            np.zeros(num_rows * num_cols, dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    rx_coords_cp = cp.asarray(rx_coords_np)

    scan_coords_np = np.stack(
        (grid.x.ravel(), grid.y.ravel(), grid.z.ravel()), axis=1
    ).astype(np.float32)
    scan_coords_cp = cp.asarray(scan_coords_np)

    # Combined per-element transmit delays (angles, rows*cols).
    tx_delays_full = cp.asarray(
        (txd[:, np.newaxis, :] + txd_elev[:, :, np.newaxis]).reshape(txd.shape[0], -1),
        dtype=cp.float32,
    )

    # Transmit-wave arrival per scan point and angle, computed in scan-point
    # chunks so the full (n_scan, n_rx) distance matrix is never materialized.
    n_scan = scan_coords_cp.shape[0]
    inv_c = 1.0 / float(config.speed_of_sound_m_s)
    tx_arrivals_all = cp.empty((num_angles, n_scan), dtype=cp.float32)
    chunk = 200000
    for s0 in range(0, n_scan, chunk):
        s1 = min(s0 + chunk, n_scan)
        dc = cp.linalg.norm(
            rx_coords_cp[cp.newaxis, :, :] - scan_coords_cp[s0:s1, cp.newaxis, :], axis=2
        ) * inv_c
        for a in range(num_angles):
            tx_arrivals_all[a, s0:s1] = cp.min(tx_delays_full[a, cp.newaxis, :] + dc, axis=1)
        del dc

    iq_cp = cp.asarray(iq)
    all_beamformed = []
    for angle_idx in range(num_angles):
        ch = iq_cp[:, angle_idx, :, :, :].reshape(num_frames, num_rows * num_cols, num_time)
        ch = cp.transpose(ch, (1, 2, 0))[cp.newaxis, ...]  # (1, n_rx, n_samples, n_frames)

        tx_arrivals = tx_arrivals_all[angle_idx][cp.newaxis, :]

        bf = _mach_beamform(
            channel_data=ch,
            rx_coords_m=rx_coords_cp,
            scan_coords_m=scan_coords_cp,
            tx_wave_arrivals_s=tx_arrivals,
            rx_start_s=config.rx_offset_s,
            sampling_freq_hz=config.sampling_rate_hz,
            f_number=config.f_number if config.f_number else 2.0,
            sound_speed_m_s=config.speed_of_sound_m_s,
            modulation_freq_hz=config.tx_freq_hz,
            tukey_alpha=0.5,
        )  # (n_points, n_frames)
        bf = bf.reshape(grid.depth_pixels, grid.height_pixels, grid.width_pixels, num_frames)
        all_beamformed.append(cp.asnumpy(bf))

    # (A, z, elev, x, F) -> compound over angles -> (F, elev, z, x)
    beamformed_all = np.stack(all_beamformed, axis=0).astype(np.complex64)
    compound = beamformed_all.sum(axis=0)  # (z, elev, x, F)
    compound = np.transpose(compound, (3, 1, 0, 2)).astype(np.complex64)  # (F, elev, z, x)
    return compound, grid


# --------------------------------------------------------------------------- #
# MLX beamform (Apple Silicon delay-and-sum backend)
# --------------------------------------------------------------------------- #
def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _tukey_apodization_mlx(r_norm, tukey_alpha: float, mx):
    """Half Tukey receive-apodization window matching the MACH CUDA kernel."""
    if tukey_alpha <= 0.0:
        return mx.ones(r_norm.shape, dtype=mx.float32)
    flat = r_norm <= (1.0 - tukey_alpha)
    taper = 0.5 - 0.5 * mx.cos(np.pi * (1.0 - r_norm) / tukey_alpha)
    return mx.where(r_norm <= 1.0, mx.where(flat, 1.0, taper), 0.0)


def _linear_interpolate_mlx(channel_data, sample_idx, valid, mx):
    """Linear interpolation over channel_data's sample axis.

    ``channel_data`` is ``(rx, samples, frames)`` and ``sample_idx`` is
    ``(scan, rx)``. The result is ``(scan, rx, frames)``.
    """
    n_scan, n_rx = sample_idx.shape
    _, n_samples, n_frames = channel_data.shape

    sample_floor = mx.floor(sample_idx)
    sample_ceil = mx.ceil(sample_idx)
    idx0 = mx.clip(sample_floor, 0.0, float(n_samples - 1)).astype(mx.int32)
    idx1 = mx.clip(sample_ceil, 0.0, float(n_samples - 1)).astype(mx.int32)
    alpha = sample_idx - sample_floor

    idx_shape = (n_scan, n_rx, 1, n_frames)
    idx0 = mx.broadcast_to(idx0[:, :, None, None], idx_shape)
    idx1 = mx.broadcast_to(idx1[:, :, None, None], idx_shape)
    expanded = channel_data[None, :, :, :]
    v0 = mx.take_along_axis(expanded, idx0, axis=2)
    v1 = mx.take_along_axis(expanded, idx1, axis=2)
    samples = mx.squeeze(v0 + (v1 - v0) * alpha[:, :, None, None], axis=2)
    return mx.where(valid[:, :, None], samples, 0.0)


def _compute_tx_arrivals_mlx(
    rx_coords,
    scan_coords,
    tx_delays_full,
    speed_of_sound_m_s: float,
    receive_chunk: int,
    mx,
):
    """Compute per-scan transmit-wave arrivals by min over emitting elements."""
    n_scan = scan_coords.shape[0]
    n_rx = rx_coords.shape[0]
    inv_c = 1.0 / float(speed_of_sound_m_s)
    arrivals = mx.full((n_scan,), np.inf, dtype=mx.float32)

    for r0 in range(0, n_rx, receive_chunk):
        r1 = min(r0 + receive_chunk, n_rx)
        rx = rx_coords[r0:r1]
        dist = mx.linalg.norm(rx[None, :, :] - scan_coords[:, None, :], axis=2)
        candidate = dist * inv_c + tx_delays_full[r0:r1][None, :]
        arrivals = mx.minimum(arrivals, mx.min(candidate, axis=1))
        mx.eval(arrivals)

    return arrivals


def _beamform_single_transmit_mlx(
    channel_data: np.ndarray,  # (rx, samples, frames) complex64
    rx_coords_m: np.ndarray,  # (rx, 3) float32
    scan_coords_m: np.ndarray,  # (scan, 3) float32
    tx_wave_arrivals_s: np.ndarray,  # (scan,) float32
    *,
    rx_start_s: float,
    sampling_freq_hz: float,
    f_number: float,
    sound_speed_m_s: float,
    modulation_freq_hz: float,
    tukey_alpha: float = 0.5,
    scan_chunk: int = 2048,
    receive_chunk: int = 128,
    frame_chunk: int = 16,
) -> np.ndarray:
    """MLX delay-and-sum for one transmit angle.

    The implementation mirrors MACH's linear-interpolation path:
    dynamic F-number aperture, optional Tukey apodization, and complex IQ phase
    correction by the physical transmit+receive travel time.
    """
    resolve_beamform_backend("mlx")
    mx = _mx
    assert mx is not None

    scan_chunk = _positive_int(scan_chunk, "scan_chunk")
    receive_chunk = _positive_int(receive_chunk, "receive_chunk")
    frame_chunk = _positive_int(frame_chunk, "frame_chunk")

    channel_np = np.ascontiguousarray(channel_data, dtype=np.complex64)
    rx_coords_np = np.ascontiguousarray(rx_coords_m, dtype=np.float32)
    scan_coords_np = np.ascontiguousarray(scan_coords_m, dtype=np.float32)
    tx_arrivals_np = np.ascontiguousarray(tx_wave_arrivals_s, dtype=np.float32)

    n_rx, n_samples, n_frames = channel_np.shape
    n_scan = scan_coords_np.shape[0]
    if rx_coords_np.shape != (n_rx, 3):
        raise ValueError(f"rx_coords_m must have shape ({n_rx}, 3); got {rx_coords_np.shape}")
    if scan_coords_np.shape != (n_scan, 3):
        raise ValueError(f"scan_coords_m must have shape ({n_scan}, 3); got {scan_coords_np.shape}")
    if tx_arrivals_np.shape != (n_scan,):
        raise ValueError(
            f"tx_wave_arrivals_s must have shape ({n_scan},); got {tx_arrivals_np.shape}"
        )

    rx_coords = mx.array(rx_coords_np, dtype=mx.float32)
    scan_coords = mx.array(scan_coords_np, dtype=mx.float32)
    tx_arrivals = mx.array(tx_arrivals_np, dtype=mx.float32)
    out = np.zeros((n_scan, n_frames), dtype=np.complex64)

    inv_c = 1.0 / float(sound_speed_m_s)
    modulation_freq_rad = 2.0 * np.pi * float(modulation_freq_hz)
    f_number = float(f_number) if f_number else 2.0

    for s0 in range(0, n_scan, scan_chunk):
        s1 = min(s0 + scan_chunk, n_scan)
        scan = scan_coords[s0:s1]
        scan_tx = tx_arrivals[s0:s1]
        z = scan[:, 2]
        aperture_radius = z / (2.0 * f_number)
        aperture_radius_squared = aperture_radius * aperture_radius

        for f0 in range(0, n_frames, frame_chunk):
            f1 = min(f0 + frame_chunk, n_frames)
            accum = mx.zeros((s1 - s0, f1 - f0), dtype=mx.complex64)

            for r0 in range(0, n_rx, receive_chunk):
                r1 = min(r0 + receive_chunk, n_rx)
                rx = rx_coords[r0:r1]
                ch = mx.array(channel_np[r0:r1, :, f0:f1], dtype=mx.complex64)

                delta = rx[None, :, :] - scan[:, None, :]
                horizontal_sq = delta[:, :, 0] * delta[:, :, 0] + delta[:, :, 1] * delta[:, :, 1]
                inside = horizontal_sq <= aperture_radius_squared[:, None]
                rx_distance = mx.sqrt(horizontal_sq + delta[:, :, 2] * delta[:, :, 2])
                physical_tau = scan_tx[:, None] + rx_distance * inv_c
                sample_idx = (physical_tau - float(rx_start_s)) * float(sampling_freq_hz)
                valid = inside & (sample_idx >= 0.0) & (sample_idx <= float(n_samples - 1))

                samples = _linear_interpolate_mlx(ch, sample_idx, valid, mx)
                if tukey_alpha > 0.0:
                    horizontal = mx.sqrt(mx.maximum(horizontal_sq, 0.0))
                    r_norm = horizontal / aperture_radius[:, None]
                    samples = samples * _tukey_apodization_mlx(r_norm, tukey_alpha, mx)[:, :, None]

                if modulation_freq_hz:
                    phase = mx.exp((1j * modulation_freq_rad) * physical_tau)
                    samples = samples * phase[:, :, None]

                accum = accum + mx.sum(samples, axis=1)
                mx.eval(accum)

            out[s0:s1, f0:f1] = np.array(accum)

        mx.clear_cache()

    return out


def beamform_iq_mlx(
    iq_frames: np.ndarray,  # (loops, angles, rows, cols, time) complex
    tx_delays: np.ndarray,  # (angles, channels) seconds
    tx_delays_elev: np.ndarray,  # (angles, rows) seconds
    config: NeutralConfig,
    *,
    scan_chunk: int = 2048,
    receive_chunk: int = 128,
    frame_chunk: int = 16,
) -> tuple[np.ndarray, Grid]:
    """Beamform neutral IQ with an MLX delay-and-sum backend.

    Returns ``(compound_image, grid)`` where compound_image is
    ``(frames, elev, z, x)`` complex64 and ``grid`` arrays are ``(z, elev, x)``.
    """
    resolve_beamform_backend("mlx")
    mx = _mx
    assert mx is not None

    iq = preprocess(
        iq_frames,
        config,
        row_index=config.row_index,
        mean_subtract_channels=config.mean_subtract_channels,
    )
    num_frames, num_angles, num_rows, num_cols, _ = iq.shape

    grid = build_grid(config)

    txd = prepare_tx_delays_for_row(
        list(tx_delays), num_cols, row_index=config.row_index
    )
    txd_elev = np.asarray(tx_delays_elev, dtype=np.float64)
    if config.row_index is not None and config.row_index >= 0:
        txd_elev = txd_elev[:, [config.row_index]]
    elif num_rows == 1:
        txd_elev = txd_elev[:, [txd_elev.shape[1] // 2]]
    elif txd_elev.shape[1] != num_rows:
        if txd_elev.shape[1] > num_rows:
            stride = max(1, txd_elev.shape[1] // num_rows)
            txd_elev = txd_elev[:, ::stride][:, :num_rows]
        else:
            pad = np.zeros((txd_elev.shape[0], num_rows), dtype=np.float64)
            pad[:, : txd_elev.shape[1]] = txd_elev
            txd_elev = pad

    x_channels = (np.arange(num_cols, dtype=np.float32) - (num_cols - 1) / 2) * config.element_pitch_x_m
    y_rows = (np.arange(num_rows, dtype=np.float32) - (num_rows - 1) / 2) * config.element_pitch_y_m
    rx_coords_np = np.stack(
        (
            np.tile(x_channels, num_rows),
            np.repeat(y_rows, num_cols),
            np.zeros(num_rows * num_cols, dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)

    scan_coords_np = np.stack(
        (grid.x.ravel(), grid.y.ravel(), grid.z.ravel()), axis=1
    ).astype(np.float32)
    n_scan = scan_coords_np.shape[0]

    rx_coords_mx = mx.array(rx_coords_np, dtype=mx.float32)
    scan_coords_mx = mx.array(scan_coords_np, dtype=mx.float32)
    tx_delays_full_np = (
        txd[:, np.newaxis, :] + txd_elev[:, :, np.newaxis]
    ).reshape(txd.shape[0], -1).astype(np.float32)

    compound_flat = np.zeros((n_scan, num_frames), dtype=np.complex64)
    receive_chunk = _positive_int(receive_chunk, "receive_chunk")
    scan_chunk = _positive_int(scan_chunk, "scan_chunk")
    frame_chunk = _positive_int(frame_chunk, "frame_chunk")

    for angle_idx in range(num_angles):
        ch = iq[:, angle_idx, :, :, :].reshape(num_frames, num_rows * num_cols, -1)
        ch = np.transpose(ch, (1, 2, 0)).astype(np.complex64, copy=False)
        tx_full = mx.array(tx_delays_full_np[angle_idx], dtype=mx.float32)
        tx_arrivals_np = np.empty(n_scan, dtype=np.float32)

        for s0 in range(0, n_scan, scan_chunk):
            s1 = min(s0 + scan_chunk, n_scan)
            tx_arrivals = _compute_tx_arrivals_mlx(
                rx_coords_mx,
                scan_coords_mx[s0:s1],
                tx_full,
                config.speed_of_sound_m_s,
                receive_chunk,
                mx,
            )
            tx_arrivals_np[s0:s1] = np.array(tx_arrivals)

        angle_out = _beamform_single_transmit_mlx(
            ch,
            rx_coords_np,
            scan_coords_np,
            tx_arrivals_np,
            rx_start_s=config.rx_offset_s,
            sampling_freq_hz=config.sampling_rate_hz,
            f_number=config.f_number if config.f_number else 2.0,
            sound_speed_m_s=config.speed_of_sound_m_s,
            modulation_freq_hz=config.tx_freq_hz,
            tukey_alpha=0.5,
            scan_chunk=scan_chunk,
            receive_chunk=receive_chunk,
            frame_chunk=frame_chunk,
        )
        compound_flat += angle_out

    compound = compound_flat.reshape(
        grid.depth_pixels, grid.height_pixels, grid.width_pixels, num_frames
    )
    compound = np.transpose(compound, (3, 1, 0, 2)).astype(np.complex64)
    return compound, grid


def beamform_iq(
    iq_frames: np.ndarray,  # (loops, angles, rows, cols, time) complex
    tx_delays: np.ndarray,  # (angles, channels) seconds
    tx_delays_elev: np.ndarray,  # (angles, rows) seconds
    config: NeutralConfig,
    *,
    backend: str = "mach",
    scan_chunk: int = 2048,
    receive_chunk: int = 128,
    frame_chunk: int = 16,
) -> tuple[np.ndarray, Grid]:
    """Beamform neutral IQ with the selected backend."""
    resolved = resolve_beamform_backend(backend)
    if resolved == "mach":
        return beamform_iq_mach(iq_frames, tx_delays, tx_delays_elev, config)
    return beamform_iq_mlx(
        iq_frames,
        tx_delays,
        tx_delays_elev,
        config,
        scan_chunk=scan_chunk,
        receive_chunk=receive_chunk,
        frame_chunk=frame_chunk,
    )


# --------------------------------------------------------------------------- #
# Global spatial TGC (port of misc/ulm/beamform_tgc_n4.py)
# --------------------------------------------------------------------------- #
def compute_global_tgc(
    compounds: list[np.ndarray],
    grid: Grid,
    tx_freq_hz: float,
    svd_cut: float = 0.05,
    sigma_lambda: float = 9.0,
) -> np.ndarray:
    """Build a global spatial-TGC map from a set of beamformed compounds.

    Averages an SVD power map (using the full SVD filter), then
    Gaussian-blurs by ``sigma = sigma_lambda * lambda`` (lambda at c=1540) in the
    (elev, z, x) layout. Returns the inverse-sqrt normalization map (1, elev, z, x).
    """
    from scipy.ndimage import gaussian_filter

    from .svd import filter_svd_3d

    pd_sum = None
    for ci in compounds:
        out = filter_svd_3d(np.asarray(ci), low_cutoff=svd_cut, method="full")
        pd = (np.abs(out) ** 2).mean(0)  # (elev, z, x)
        pd_sum = pd if pd_sum is None else pd_sum + pd
    pd_mean = pd_sum / len(compounds)

    # Per-axis voxel sizes from the grid (z, elev, x ordering).
    z_ax = grid.z[:, 0, 0]
    y_ax = grid.y[0, :, 0]
    x_ax = grid.x[0, 0, :]
    wavelength = 1540.0 / tx_freq_hz
    sigma_m = sigma_lambda * wavelength
    dz = abs(z_ax[1] - z_ax[0]) if len(z_ax) > 1 else None
    dx = abs(x_ax[1] - x_ax[0]) if len(x_ax) > 1 else None
    dy = abs(y_ax[1] - y_ax[0]) if len(y_ax) > 1 else None
    sigma_z = sigma_m / dz if dz else 0.0
    sigma_x = sigma_m / dx if dx else 0.0
    sigma_y = sigma_m / dy if (dy and dy > 0) else 0.0

    tgc = gaussian_filter(pd_mean, sigma=(sigma_y, sigma_z, sigma_x)).astype(np.float32)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(tgc, np.finfo(np.float32).eps))
    return inv_sqrt[None]  # (1, elev, z, x)
