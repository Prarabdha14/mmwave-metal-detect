# simulate/detect.py
import numpy as np

def simple_detector(rd_db, threshold_factor=4.0, min_distance_bins=2):
    """
    Quick detector: median+std adaptive threshold.
    rd_db: 2D array (doppler_bins x range_bins)
    Returns filtered list of (doppler_idx, range_idx) and threshold value.
    """
    med = np.median(rd_db)
    sd = np.std(rd_db)
    thresh = med + threshold_factor * sd
    peaks = np.argwhere(rd_db > thresh)
    # filter close duplicates
    filtered = []
    for p in peaks:
        dp, rp = int(p[0]), int(p[1])
        too_close = False
        for q in filtered:
            if abs(q[0]-dp) <= min_distance_bins and abs(q[1]-rp) <= min_distance_bins:
                too_close = True
                break
        if not too_close:
            filtered.append((dp, rp))
    return filtered, thresh
