import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from models import Sample, CompressionSession, CompressionTestRecord
from compression_planner import (
    extract_all_particles, predict_smaract_translation,
    refine_orientation_scale, get_tilt_transform, map_point_tilt,
    optimize_particle_route
)
from main import AppController

def test_models_and_planner():
    print("Testing models and planner...")
    sample_json = ROOT_DIR / 'tests' / 'fixtures' / 'sample_test_data.json'
    if not sample_json.exists():
        sample_json = ROOT_DIR / 'Region2' / 'sample_data.json'
    with open(sample_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sample = Sample(**data)
    particles = extract_all_particles(sample)
    
    assert len(particles) == 61, f"Expected 61 particles, got {len(particles)}"
    assert '6_1' in particles, "Expected UID '6_1' for Site 6 Particle 1"
    assert '18_3' in particles, "Expected UID '18_3' for Site 18 Particle 3"
    assert '223_1' in particles, "Expected UID '223_1' for Site 223 Particle 1"
    
    p61 = particles['6_1']
    p62 = particles['6_2']
    assert p61.uid == '6_1'
    assert p61.site_number == 6
    assert p61.particle_id == 1
    
    # Test SmarAct prediction
    dx, dy, dist = predict_smaract_translation(
        (p61.flat_x_um, p61.flat_y_um),
        (p62.flat_x_um, p62.flat_y_um),
        rot_deg=0.0
    )
    # At rot=0, dx = p62.flat_x_um - p61.flat_x_um
    assert abs(dx - (p62.flat_x_um - p61.flat_x_um)) < 1e-4
    assert abs(dy - (p62.flat_y_um - p61.flat_y_um)) < 1e-4
    
    # Test Flip X / Y
    dx_f, dy_f, dist_f = predict_smaract_translation(
        (p61.flat_x_um, p61.flat_y_um),
        (p62.flat_x_um, p62.flat_y_um),
        rot_deg=0.0,
        flip_xy=True
    )
    assert abs(dx_f - dy) < 1e-4 and abs(dy_f - dx) < 1e-4, "Flip XY failed"
    
    # Test Kabsch refinement
    flats = [(10.0, 5.0), (-5.0, 20.0), (12.0, -8.0)]
    acts = [predict_smaract_translation((0,0), pt, rot_deg=35.0, scale=1.01)[:2] for pt in flats]
    rot_est, s_est, err = refine_orientation_scale(flats, acts)
    assert abs(rot_est - 35.0) < 0.1, f"Kabsch rot error: {rot_est}"
    assert abs(s_est - 1.01) < 0.01, f"Kabsch scale error: {s_est}"

    # Test TSP Route Optimizer
    all_uids = list(particles.keys())
    route, total_dist = optimize_particle_route('6_1', all_uids, particles)
    assert len(route) == 61, f"Expected 61 particles in route, got {len(route)}"
    assert route[0] == '6_1', "First particle in tour must be the chosen start particle"
    assert len(set(route)) == 61, "Tour must visit each particle exactly once"
    assert total_dist > 0.0, "Tour distance must be positive"
    print(f"TSP route computed: 61 particles, {total_dist:.1f} um total travel")

    print("Models and planner passed!")

def test_full_ui_integration():
    print("Testing full UI integration...")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Create temp directory with copy of Region2 so we don't dirty original files
    temp_dir = tempfile.mkdtemp(prefix="sem_comp_test_")
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
        
        # Verify Tab 3 setup
        assert controller.window.tabs.count() == 3, f"Expected 3 tabs, got {controller.window.tabs.count()}"
        assert controller.window.tabs.tabText(2) == "Micro-Compression"
        
        # Switch to Tab 3
        controller.window.tabs.setCurrentIndex(2)
        controller.on_tab_changed(2)
        
        # Verify particles loaded & TSP tour computed
        assert len(controller.comp_particles) == 61
        assert controller.comp_current_uid is not None
        assert controller.comp_next_uid is not None
        assert len(controller.comp_tour_order) == 61
        assert controller.comp_tour_order[0] == controller.comp_current_uid
        assert controller.comp_tour_order[1] == controller.comp_next_uid
        assert controller.comp_tour_distance_um > 0.0
        print(f"Current particle: {controller.comp_current_uid}, Auto-selected Next: {controller.comp_next_uid}")
        
        # Verify UI redesign elements
        # 1. Obsolete button removed
        assert not hasattr(controller.window, 'btn_comp_set_next'), "btn_comp_set_next should be removed"
        # 2. Queue summary label
        summary_text = controller.window.lbl_comp_queue_summary.text()
        assert "Tour:" in summary_text and "remaining" in summary_text
        # 3. Table columns: 6 columns
        expected_cols = ["Step", "UID", "Hop (µm)", "Shape", "Facet (nm)", "Status"]
        assert controller.window.comp_particle_model.columnCount() == 6
        for col_idx, col_name in enumerate(expected_cols):
            assert controller.window.comp_particle_model.headerData(col_idx, Qt.Horizontal) == col_name
        
        # 4. Filter buttons exist and work
        assert controller.window.btn_comp_filter_all.isChecked()
        controller.window.btn_comp_filter_untested.setChecked(True)
        controller.populate_compression_table()
        assert controller.window.comp_particle_model.rowCount() == 61  # Initially all 61 untested
        
        # Verify Mode A (Testing Mode) is active initially
        assert controller.comp_mode == "test"
        assert controller.window.comp_mode_stack.currentIndex() == 0
        assert controller.window.lbl_test_particle_uid.text() == f"Particle {controller.comp_current_uid}"
        assert controller.window.combo_comp_next.count() == 61
        
        # Verify initial site view is displaying the CURRENT particle's site
        curr_p = controller.comp_particles[controller.comp_current_uid]
        assert curr_p.site_filename in controller.comp_site_filename
        # Check title indicates Current Particle
        site_title_text = controller.window.lbl_comp_site_title.text()
        assert "Current Particle" in site_title_text
        assert controller.comp_current_uid in site_title_text
        
        # Check SmarAct display
        dx_text = controller.window.lbl_comp_dx.text()
        dy_text = controller.window.lbl_comp_dy.text()
        print("Readouts:", dx_text.encode('ascii', 'backslashreplace').decode('ascii'))
        assert "dX =" in dx_text or "ΔX =" in dx_text
        assert "dY =" in dy_text or "ΔY =" in dy_text
        
        # Test Flip X / Y toggle
        old_dx = controller.window.lbl_comp_dx.text()
        old_dy = controller.window.lbl_comp_dy.text()
        controller.window.btn_comp_flip_xy.setChecked(True)
        controller.on_comp_flip_xy_toggled(True)
        new_dx = controller.window.lbl_comp_dx.text()
        new_dy = controller.window.lbl_comp_dy.text()
        print("After Flip X/Y:", new_dx.encode('ascii', 'backslashreplace').decode('ascii'))
        assert "Axes Flipped" in controller.window.btn_comp_flip_xy.text()
        
        # Revert Flip X/Y
        controller.window.btn_comp_flip_xy.setChecked(False)
        controller.on_comp_flip_xy_toggled(False)
        assert controller.window.lbl_comp_dx.text() == old_dx
        
        # Test quick rotation
        controller.on_comp_quick_rotate(45.0)
        assert abs(controller.comp_rot_deg - 45.0) < 1e-4
        assert abs(controller.window.spin_comp_rot.value() - 45.0) < 1e-4
        
        # --- Test 2-Mode Workflow ---
        tested_before = sum(1 for p in controller.comp_particles.values() if p.tested)
        curr_before = controller.comp_current_uid
        next_before = controller.comp_next_uid
        next_p = controller.comp_particles[next_before]
        
        # 1. First Enter press in Mode A (Testing Mode) -> Logs test and enters Mode B (Translation Mode)
        controller.on_comp_hotkey_enter()
        
        assert controller.comp_mode == "trans", f"Expected mode 'trans', got {controller.comp_mode}"
        assert controller.window.comp_mode_stack.currentIndex() == 1, "Expected stack index 1 (Translation Page)"
        tested_after = sum(1 for p in controller.comp_particles.values() if p.tested)
        assert tested_after == tested_before + 1, f"Expected {tested_before+1} tested, got {tested_after}"
        assert controller.comp_particles[curr_before].tested is True
        
        # VERIFY DYNAMIC TARGET SITE VIEW IN TRANSLATION MODE:
        # The site view must display target particle's site, NOT current particle's site
        assert next_p.site_filename in controller.comp_site_filename, (
            f"Expected site view filename to be {next_p.site_filename}, got {controller.comp_site_filename}"
        )
        trans_title_text = controller.window.lbl_comp_site_title.text()
        assert "Target Particle" in trans_title_text
        assert next_before in trans_title_text
        print(f"Mode B confirmed displaying Target particle {next_before} site: {next_p.site_filename}")
        
        # Check session log file creation
        session_file = os.path.join(temp_dir, controller.comp_session_filename)
        assert os.path.exists(session_file), f"Session file {session_file} does not exist"
        with open(session_file, 'r') as f:
            sess_json = json.load(f)
        assert len(sess_json['tests']) == 1
        assert sess_json['tests'][0]['particle_uid'] == curr_before
        print(f"Logged test entry for particle {curr_before} successfully in {controller.comp_session_filename}")
        
        # Test Reset button in Translation Mode
        controller.window.spin_comp_act_dx.setValue(999.0)
        controller.on_comp_reset_trans()
        assert abs(controller.window.spin_comp_act_dx.value() - 999.0) > 1.0, "Reset should restore predicted dx"
        
        # 2. Second Enter press in Mode B (Translation Mode) -> Accepts move, refines alignment, and enters Mode A for next target
        controller.on_comp_hotkey_enter()
        
        assert controller.comp_mode == "test", f"Expected mode 'test', got {controller.comp_mode}"
        assert controller.window.comp_mode_stack.currentIndex() == 0, "Expected stack index 0 (Testing Page)"
        assert controller.comp_current_uid == next_before, "Current particle should now be the target particle"
        assert controller.window.lbl_test_particle_uid.text() == f"Particle {next_before}"
        
        # Verify Site view has reticle items and displays new current particle
        assert len(controller.comp_site_marker_items) > 0, "Site view should have targeting reticle and particle items"
        assert next_p.site_filename in controller.comp_site_filename
        assert "Current Particle" in controller.window.lbl_comp_site_title.text()
        
        # Test Table selection & Set Selected as Current with auto-tour reoptimization
        # Select row 10 in table
        controller.window.comp_particle_table.selectRow(10)
        target_row_uid = controller.window.comp_particle_model.item(10, 1).text()
        controller.on_comp_set_current_clicked()
        assert controller.comp_current_uid == target_row_uid
        assert controller.comp_tour_order[0] == target_row_uid
        assert controller.comp_next_uid == controller.comp_tour_order[1]
        print(f"Selected row 10 (UID {target_row_uid}) as current. Tour recomputed! New next: {controller.comp_next_uid}")
        
        print("Full UI integration passed!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_models_and_planner()
    test_full_ui_integration()
    print("ALL TESTS PASSED SUCCESSFULLY!")
