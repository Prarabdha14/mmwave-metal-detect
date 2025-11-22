# deploy/inference_pipeline.py
import json
from simulate.sim import simulate_scene
from simulate.proc import compute_range_doppler, doppler_bins_to_velocity, range_bin_to_meters
from simulate.detect import simple_detector

def rd_to_detection_json(peaks, doppler_vs, range_res):
    detections = []
    for (d_idx, r_idx) in peaks:
        detections.append({
            "range_bin": int(r_idx),
            "range_m": float(round(r_idx * range_res, 3)),
            "doppler_idx": int(d_idx),
            "velocity_m_s": float(round(doppler_vs[d_idx], 3))
        })
    return detections

def run_demo():
    targets = [
        {'r': 5.0, 'v': 0.0, 'rcs': 3.0, 'snr_db': 28},
        {'r': 8.5, 'v': -1.1, 'rcs': 1.2, 'snr_db': 18},
        {'r': 12.0, 'v': 0.0, 'rcs': 0.6, 'snr_db': 8}
    ]
    scene = simulate_scene(targets)
    rd_db, _ = compute_range_doppler(scene)
    peaks, thresh = simple_detector(rd_db, threshold_factor=4.0)
    doppler_vs = doppler_bins_to_velocity()
    range_res = 3e8 / (2*240e6)
    detections = rd_to_detection_json(peaks, doppler_vs, range_res)
    out = {"detections": detections, "threshold_db": float(thresh)}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    run_demo()
