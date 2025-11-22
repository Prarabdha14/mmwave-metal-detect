# simulate/proc.py
import numpy as np
from scipy.signal.windows import hann
from .sim import N_samples, N_chirps, B, T_chirp, fs, lam

def compute_range_doppler(scene):
    """
    scene: (N_chirps, N_samples) complex
    returns: RD_db (doppler_bins x range_bins), Rpos (N_chirps x N_range_bins)
    """
    win_fast = hann(N_samples)
    scene_win = scene * win_fast[np.newaxis, :]

    R = np.fft.fft(scene_win, axis=1)
    N_range_bins = N_samples // 2
    R_pos = R[:, :N_range_bins]

    RD = np.fft.fft(R_pos, axis=0)
    RD_shift = np.fft.fftshift(RD, axes=0)

    RD_mag = np.abs(RD_shift)
    RD_db = 20 * np.log10(RD_mag + 1e-12)
    return RD_db, R_pos

def range_bin_to_meters(bin_idx):
    range_res = 3e8 / (2 * B)
    return bin_idx * range_res

def doppler_bins_to_velocity():
    f_d = np.fft.fftshift(np.fft.fftfreq(N_chirps, d=T_chirp))
    # velocity = f_d * lambda / 2
    return f_d * lam / 2
