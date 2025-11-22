# tests/test_smoke.py
from simulate.sim import simulate_scene
from simulate.proc import compute_range_doppler

def test_sim_and_proc_runs():
    targets = [{'r':5.0,'v':0.0,'rcs':2.0,'snr_db':20}]
    scene = simulate_scene(targets)
    rd_db, Rpos = compute_range_doppler(scene)
    assert rd_db is not None
    assert rd_db.size > 0
