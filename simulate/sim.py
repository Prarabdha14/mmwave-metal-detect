# simulate/sim.py
import numpy as np

# Constants & radar params (tweak if needed)
c = 3e8
fc = 60e9
lam = c / fc
fs = 2e6            # ADC sampling rate for beat signal (Hz)
T_chirp = 60e-6     # chirp duration (s)
N_samples = max(256, int(T_chirp * fs))  # fast-time samples (ensure at least 256)
N_chirps = 64       # slow-time (chirps per CPI)
B = 240e6           # sweep bandwidth
S = B / T_chirp

def generate_target_echo(range_m, vel_m_s, rcs=1.0, snr_db=20):
    """
    Generate complex baseband returns for one target.
    Returns array shape (N_chirps, N_samples) dtype complex64.
    """
    f_b = 2.0 * S * range_m / c
    f_d = 2.0 * vel_m_s / lam

    t_fast = np.arange(N_samples) / fs
    data = np.zeros((N_chirps, N_samples), dtype=np.complex64)

    amp = rcs / (range_m**2 + 1e-6)  # simple range attenuation
    for n in range(N_chirps):
        slow_time = n * T_chirp
        doppler_phase = np.exp(1j * 2.0 * np.pi * f_d * slow_time)
        beat = np.exp(1j * 2.0 * np.pi * f_b * t_fast)
        data[n, :] = amp * doppler_phase * beat

    # Add AWGN to produce requested SNR
    sig_pow = np.mean(np.abs(data)**2)
    snr_lin = 10**(snr_db/10.0)
    noise_power = sig_pow / (snr_lin + 1e-12)
    noise = np.sqrt(noise_power/2) * (np.random.randn(*data.shape) + 1j*np.random.randn(*data.shape))
    return data + noise

def simulate_scene(target_list):
    """
    target_list: list of dicts {'r': meters, 'v': m/s, 'rcs': val, 'snr_db': val}
    returns combined scene array (N_chirps, N_samples)
    """
    scene = np.zeros((N_chirps, N_samples), dtype=np.complex64)
    for t in target_list:
        scene += generate_target_echo(t['r'], t['v'], rcs=t.get('rcs',1.0), snr_db=t.get('snr_db',20))
    return scene
