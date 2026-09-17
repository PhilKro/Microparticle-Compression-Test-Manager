import os
import sys
import json
import math
import shutil
import tempfile
from pathlib import Path
import numpy as np
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from models import Sample, CompressionSession, CompressionTestRecord
from compression_planner import (
    extract_all_particles, predict_smaract_translation,
    robust_refine_orientation_scale, CalibrationResult
)
from main import AppController

def test_rb_pse_algorithm():
    print("Testing Robust Bayesian Pose & Scale Estimator (RB-PSE) algorithm...")
    # True physical parameters: orientation = 35.0 deg, stage scale = 1.100 (piezo moves 10% further)
    true_phi = 35.0
    true_scale = 1.100
    rad = math.radians(true_phi)
    R_true = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])

    # Operator starts with visual coarse alignment of 30.0 deg and 1.00 scale
    prior_phi = 30.0
    prior_scale = 1.00

    # 1. Prior only (N=0)
    res0 = robust_refine_orientation_scale([], [], prior_rot_deg=prior_phi, prior_scale=prior_scale)
    assert abs(res0.rot_deg - 30.0) < 1e-4
    assert abs(res0.scale - 1.00) < 1e-4
    assert res0.confidence_pct == 0.0
    assert res0.inlier_count == 0

    # 2. Single short hop (2.5 um)
    p1 = (2.0, 1.5)
    s1_vec = true_scale * (R_true @ np.array(p1)) + np.array([0.05, -0.05])
    s1 = (float(s1_vec[0]), float(s1_vec[1]))
    res1 = robust_refine_orientation_scale([p1], [s1], prior_rot_deg=prior_phi, prior_scale=prior_scale)
    # Regularization should prevent erratic swing: should be between 30 and 35
    assert 30.0 <= res1.rot_deg <= 35.0, f"Expected 30 <= rot <= 35, got {res1.rot_deg}"
    assert 1.00 <= res1.scale <= 1.10, f"Expected 1.00 <= scale <= 1.10, got {res1.scale}"
    assert res1.rot_std_deg > 0.5, "Single short hop should have noticeable uncertainty"

    # 3. Multiple hops converging to true values
    p2 = (12.0, -9.0) # 15 um
    s2_vec = true_scale * (R_true @ np.array(p2)) + np.array([-0.06, 0.04])
    s2 = (float(s2_vec[0]), float(s2_vec[1]))

    p3 = (-16.0, 12.0) # 20 um
    s3_vec = true_scale * (R_true @ np.array(p3)) + np.array([0.04, -0.05])
    s3 = (float(s3_vec[0]), float(s3_vec[1]))

    res_multi = robust_refine_orientation_scale([p1, p2, p3], [s1, s2, s3], prior_rot_deg=prior_phi, prior_scale=prior_scale)
    assert abs(res_multi.rot_deg - true_phi) < 0.5, f"Expected rot near {true_phi}, got {res_multi.rot_deg}"
    assert abs(res_multi.scale - true_scale) < 0.02, f"Expected scale near {true_scale}, got {res_multi.scale}"
    assert res_multi.confidence_pct > 70.0, f"Expected high confidence, got {res_multi.confidence_pct}"
    assert res_multi.inlier_count == 3
    print(f"  Converged with 3 hops: phi={res_multi.rot_deg:.2f}+-{res_multi.rot_std_deg:.2f} deg, scale={res_multi.scale:.3f}+-{res_multi.scale_std:.3f}")

    # 4. Outlier blunder rejection
    p_blunder = (-5.0, 8.0)
    s_blunder = (25.0, 25.0) # Massive blunder: operator drove to wrong spot or bumped stage!
    
    p4 = (18.0, 14.0) # 22.8 um
    s4_vec = true_scale * (R_true @ np.array(p4)) + np.array([0.03, -0.03])
    s4 = (float(s4_vec[0]), float(s4_vec[1]))

    flats_with_blunder = [p1, p2, p_blunder, p4]
    acts_with_blunder = [s1, s2, s_blunder, s4]
    res_blunder = robust_refine_orientation_scale(flats_with_blunder, acts_with_blunder, prior_rot_deg=prior_phi, prior_scale=prior_scale)
    assert res_blunder.inlier_count == 3, f"Expected 3 inliers, got {res_blunder.inlier_count}"
    assert res_blunder.inlier_mask[2] is False, "Blunder hop must be marked as outlier"
    assert abs(res_blunder.rot_deg - true_phi) < 0.5, f"Blunder corrupted rotation: {res_blunder.rot_deg}"
    assert abs(res_blunder.scale - true_scale) < 0.02, f"Blunder corrupted scale: {res_blunder.scale}"
    print("  Outlier blunder successfully detected and pruned by RANSAC!")

    print("RB-PSE algorithm tests PASSED!")

def test_full_session_and_ui():
    print("Testing Full Session & UI Integration...")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication(sys.argv)

    temp_dir = tempfile.mkdtemp(prefix="sem_cal_test_")
    try:
        reg_dir = ROOT_DIR / 'Region2'
        if reg_dir.exists():
            for fname in os.listdir(reg_dir):
                if fname.startswith("compression_test_"):
                    continue
                src = reg_dir / fname
                dst = os.path.join(temp_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        else:
            fixture_path = ROOT_DIR / 'tests' / 'fixtures' / 'sample_test_data.json'
            shutil.copy2(fixture_path, os.path.join(temp_dir, 'sample_data.json'))

        controller = AppController()
        controller.sample_dir = temp_dir
        controller.load_from_json(os.path.join(temp_dir, 'sample_data.json'))
        controller.populate_tree()
        controller.init_compression_tab()

        controller.window.tabs.setCurrentIndex(2)
        controller.on_tab_changed(2)

        # 1. Verify UI controls
        assert hasattr(controller.window, 'spin_comp_scale')
        assert hasattr(controller.window, 'lbl_comp_calibration_badge')
        assert abs(controller.window.spin_comp_scale.value() - 1.000) < 1e-4

        # Test manual scale tweak
        controller.window.spin_comp_scale.setValue(1.050)
        assert abs(controller.comp_scale - 1.050) < 1e-4

        # Reset scale back to 1.000
        controller.window.spin_comp_scale.setValue(1.000)
        assert abs(controller.comp_scale - 1.000) < 1e-4

        # 2. Enter Translation Mode (Mode B)
        controller.set_comp_mode("trans")
        dist_label_text = controller.window.lbl_comp_distance.text()
        assert "scale" in dist_label_text
        print("Distance label:", dist_label_text.encode('ascii', 'backslashreplace').decode('ascii'))

        # 3. Simulate user inputting actual travel with scale = 1.100
        pred_dx = controller.window.spin_comp_act_dx.value()
        pred_dy = controller.window.spin_comp_act_dy.value()
        controller.window.spin_comp_act_dx.setValue(pred_dx * 1.10)
        controller.window.spin_comp_act_dy.setValue(pred_dy * 1.10)

        # Confirm move (Enter in Mode B)
        controller.on_comp_confirm_trans()

        # Check scale adapted
        assert controller.comp_scale > 1.02, f"Scale did not adapt upward: {controller.comp_scale}"
        assert abs(controller.window.spin_comp_scale.value() - controller.comp_scale) < 1e-3
        print(f"Scale successfully auto-adapted to: {controller.comp_scale:.4f}")

        # Check calibration badge updated
        badge_text = controller.window.lbl_comp_calibration_badge.text()
        print("Calibration Badge:", badge_text.encode('ascii', 'backslashreplace').decode('ascii'))
        assert "φ=" in badge_text or "\u03c6=" in badge_text or "Initial" in badge_text or "Converged" in badge_text

        # 4. Log test in Mode A (Enter in Mode A)
        curr_uid = controller.comp_current_uid
        controller.on_comp_log_test()

        # Verify JSON file has all comprehensive mechanical test fields
        session_file = os.path.join(temp_dir, controller.comp_session_filename)
        assert os.path.exists(session_file)
        with open(session_file, 'r', encoding='utf-8') as f:
            sess_json = json.load(f)

        assert sess_json['total_particles_tested'] >= 1
        assert sess_json['scale_factor'] > 1.0
        assert 'calibration_summary' in sess_json
        assert 'stage_settings' in sess_json

        test_rec = sess_json['tests'][-1]
        assert test_rec['particle_uid'] == curr_uid
        assert 'particle_properties' in test_rec
        props = test_rec['particle_properties']
        assert 'ecd_nm' in props
        assert 'top_facet_diameter_nm' in props
        assert 'top_facet_area_um2' in props
        assert 'pixel_x' in props
        assert 'pixel_y' in props

        assert 'stage_scale_factor' in test_rec
        assert test_rec['stage_scale_factor'] > 1.0
        assert 'calibration_snapshot' in test_rec
        assert 'cumulative_stage_dx_um' in test_rec
        assert 'cumulative_stage_dy_um' in test_rec
        assert 'smaract_actual_distance_um' in test_rec

        print("Full Session & UI Integration tests PASSED!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_rb_pse_algorithm()
    test_full_session_and_ui()
    print("ALL ROBUST CALIBRATION TESTS PASSED SUCCESSFULLY!")
