import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("PYTHONPATH=", sys.path[-1])

from scipy.ndimage import zoom
from flask import Flask, request, jsonify
import numpy as np
import joblib
from simulate.proc import compute_range_doppler
from simulate.detect import simple_detector
from simulate.sim import simulate_scene

# ---------------------------
# LOAD MODELS
# ---------------------------
# ----- ensure model paths are absolute, based on this file's location -----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
pca_path = os.path.join(MODEL_DIR, "pca_transform.pkl")
svm_path = os.path.join(MODEL_DIR, "svm_baseline.pkl")
cal_svm_path = os.path.join(MODEL_DIR, "svm_calibrated.pkl")
# ------------------------------------------------------------------------
print("MODEL_DIR resolved to:", MODEL_DIR)
print("pca_path:", pca_path)
print("svm_path:", svm_path)


pca = joblib.load(pca_path)

if os.path.exists(cal_svm_path):
    svm = joblib.load(cal_svm_path)
else:
    svm = joblib.load(svm_path)

THRESHOLD = 0.6   # probability threshold
RANGE_RES = 3e8 / (2 * 240e6)  # depends on bandwidth 240 MHz


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def crop_roi(rd_db, d_idx, r_idx, size=32):
    roi = np.zeros((size, size))
    H, W = rd_db.shape
    top = d_idx - size//2
    left = r_idx - size//2
    for i in range(size):
        for j in range(size):
            dd = top + i
            rr = left + j
            if 0 <= dd < H and 0 <= rr < W:
                roi[i,j] = rd_db[dd, rr]
    return roi

def preprocess_roi_resize(roi, rd_shape):
    """
    Resize roi (e.g., 32x32) up to rd_shape (e.g., 64x128), log1p, normalize,
    flatten, pad/truncate to pca.n_features_in_, then return pca.transform(...)
    """
    # log compress + normalize small patch
    x = np.log1p(np.abs(roi))
    mx = np.max(x)
    if mx > 0:
        x = x / (mx + 1e-12)

    target_h, target_w = rd_shape
    h, w = roi.shape

    if (h, w) != (target_h, target_w):
        zh = target_h / h
        zw = target_w / w
        # use order=1 for bilinear interpolation
        roi_resized = zoom(x, (zh, zw), order=1)
    else:
        roi_resized = x

    x_flat = roi_resized.flatten().reshape(1, -1)

    # Pad or truncate to PCA expected length
    if x_flat.shape[1] != pca.n_features_in_:
        vec = np.zeros((1, pca.n_features_in_), dtype=float)
        L = min(pca.n_features_in_, x_flat.shape[1])
        vec[0, :L] = x_flat[0, :L]
        x_flat = vec

    feat_p = pca.transform(x_flat)
    return feat_p

def classify_roi_resized(roi, rd_shape):
    feat_p = preprocess_roi_resize(roi, rd_shape)
    prob = float(svm.predict_proba(feat_p)[0][1])
    label = "metal" if prob >= THRESHOLD else "non-metal"
    return label, prob


# ---------------------------
# FLASK APP
# ---------------------------

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "mmWave Radar Inference API Running"})

@app.route("/infer", methods=["POST"])
def infer():
    """
    Accepts:
      Option A:  raw RD map array  (rd_db)
      Option B:  raw chirp data    (N_chirps x N_samples)
      Option C:  parameters to simulate a scene
    """
    data = request.json

    # CASE A: RD DB map directly
    if "rd_db" in data:
        rd_db = np.array(data["rd_db"])
    
    # CASE B: raw IQ scene (run FFT inside API)
    elif "scene" in data:
        scene = np.array(data["scene"])
        rd_db, _ = compute_range_doppler(scene)

    # CASE C: simulate scene from params
    elif "simulate" in data:
        params = data["simulate"]
        scene = simulate_scene(params)
        rd_db, _ = compute_range_doppler(scene)

    else:
        return jsonify({"error": "Invalid input. Provide rd_db or scene or simulate."}), 400

    # DETECTION
    peaks, thresh_val = simple_detector(rd_db, threshold_factor=4.0)

    detections = []
    for (d_idx, r_idx) in peaks:
        roi = crop_roi(rd_db, d_idx, r_idx, size=32)
        label, prob = classify_roi_resized(roi, rd_db.shape)

        detections.append({
            "doppler_bin": int(d_idx),
            "range_bin": int(r_idx),
            "range_m": round(r_idx * RANGE_RES, 3),
            "label": label,
            "prob": round(prob, 3)
        })

    return jsonify({
        "detections": detections,
        "threshold_db": float(thresh_val)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

