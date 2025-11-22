# simulate/__init__.py
from .sim import simulate_scene, generate_target_echo
from .proc import compute_range_doppler, range_bin_to_meters, doppler_bins_to_velocity
from .detect import simple_detector
