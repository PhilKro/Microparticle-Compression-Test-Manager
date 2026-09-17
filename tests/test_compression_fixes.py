import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import Sample, ImageRecord, ImageType, Particle
from compression_planner import extract_all_particles, ParticleGlobalInfo

class TestCompressionFixes(unittest.TestCase):
    def test_dynamic_region_dimensions_alignment(self):
        """Verify that passing actual region dimensions eliminates the 324 um offset."""
        px_reg = 633.06e-9
        py_reg = 633.06e-9
        px_site = 7.3243e-9
        py_site = 7.3243e-9

        reg_record = ImageRecord(
            filename="Region1_Closeup.tif",
            classification=ImageType.REGION,
            stage_x=0.004407344,
            stage_y=0.01424656,
            metadata={
                "MAIN": {"PixelSizeX": str(px_reg), "PixelSizeY": str(py_reg), "ViewFieldsCountX": "2", "ImageStripSize": "80"}
            }
        )
        site1_record = ImageRecord(
            filename="Region1_Site1.tif",
            classification=ImageType.SITE,
            stage_x=0.0044675,
            stage_y=0.01396078,
            offset_x_pixels=-276.575,
            offset_y_pixels=999.504,
            alignment_status="Refined (0.85)",
            use_refined_alignment=True,
            metadata={
                "MAIN": {"PixelSizeX": str(px_site), "PixelSizeY": str(py_site), "ViewFieldsCountX": "2", "ImageStripSize": "240"}
            },
            particles=[
                Particle(id=1, pixel_x=523.8, pixel_y=1047.5, top_facet_diameter=1.28e-7)
            ]
        )
        sample = Sample(name="Region1", directory_path="", images=[reg_record, site1_record])

        # Tab 1 visual ROI position (load_region_view)
        cx_tab1 = 1024 * px_reg / 2.0
        cy_tab1 = 1024 * py_reg / 2.0
        site_w = 2048 * px_site
        site_h = 2048 * py_site
        dx = site1_record.stage_x - reg_record.stage_x
        dy = site1_record.stage_y - reg_record.stage_y
        tab1_tl_x = cx_tab1 - dx - site_w / 2.0 + site1_record.offset_x_pixels * px_site
        tab1_tl_y = cy_tab1 + dy - site_h / 2.0 + site1_record.offset_y_pixels * py_site

        # Old way (hardcoded 2048)
        old_parts = extract_all_particles(sample, reg_w=2048, reg_h=2048)
        old_tl_x = old_parts["1_1"].site_top_left_x_m
        old_tl_y = old_parts["1_1"].site_top_left_y_m
        self.assertAlmostEqual(abs(old_tl_x - tab1_tl_x) * 1e6, 324.1, delta=0.5)

        # New way (dynamic 1024)
        new_parts = extract_all_particles(sample, reg_w=1024, reg_h=1024)
        new_tl_x = new_parts["1_1"].site_top_left_x_m
        new_tl_y = new_parts["1_1"].site_top_left_y_m
        self.assertAlmostEqual(new_tl_x, tab1_tl_x, places=8)
        self.assertAlmostEqual(new_tl_y, tab1_tl_y, places=8)
        print("  PASSED: Dynamic region dimensions (1024x1024) exactly match Tab 1 coordinates!")

    def test_unrefined_site_flagging(self):
        """Verify that sites with unrefined or failed alignment are properly flagged."""
        reg = ImageRecord(filename="Region.tif", classification=ImageType.REGION)
        site_refined = ImageRecord(
            filename="Site1.tif", classification=ImageType.SITE, alignment_status="Refined (0.75)", use_refined_alignment=True,
            particles=[Particle(id=1, pixel_x=100, pixel_y=100)]
        )
        site_manual = ImageRecord(
            filename="Site2.tif", classification=ImageType.SITE, alignment_status="Manual", use_refined_alignment=True,
            particles=[Particle(id=1, pixel_x=100, pixel_y=100)]
        )
        site_failed = ImageRecord(
            filename="Site3.tif", classification=ImageType.SITE, alignment_status="Failed (0.12)", use_refined_alignment=True,
            particles=[Particle(id=1, pixel_x=100, pixel_y=100)]
        )
        site_disabled = ImageRecord(
            filename="Site4.tif", classification=ImageType.SITE, alignment_status="Refined (0.90)", use_refined_alignment=False,
            particles=[Particle(id=1, pixel_x=100, pixel_y=100)]
        )
        sample = Sample(name="Test", directory_path="", images=[reg, site_refined, site_manual, site_failed, site_disabled])
        parts = extract_all_particles(sample)

        self.assertTrue(parts["1_1"].site_is_refined)
        self.assertTrue(parts["2_1"].site_is_refined)
        self.assertFalse(parts["3_1"].site_is_refined)
        self.assertFalse(parts["4_1"].site_is_refined)
        print("  PASSED: Unrefined site detection correctly identifies refined vs failed sites!")

    def test_controller_caching_and_unrefined_exclusion(self):
        """Verify that controller uses cache for rotation and excludes unrefined sites from calibration."""
        from unittest.mock import MagicMock, patch
        from PySide6.QtWidgets import QApplication
        from main import AppController

        app = QApplication.instance() or QApplication([])
        ctrl = AppController()

        reg = ImageRecord(filename="Region.tif", classification=ImageType.REGION)
        site1 = ImageRecord(
            filename="Site1.tif", classification=ImageType.SITE, alignment_status="Refined (0.80)", use_refined_alignment=True,
            particles=[Particle(id=1, pixel_x=100, pixel_y=100)]
        )
        site2 = ImageRecord(
            filename="Site2.tif", classification=ImageType.SITE, alignment_status="Failed (0.10)", use_refined_alignment=True,
            particles=[Particle(id=1, pixel_x=200, pixel_y=200)]
        )
        ctrl.sample = Sample(name="TestSample", directory_path="", images=[reg, site1, site2])
        ctrl.comp_particles = extract_all_particles(ctrl.sample, reg_w=1024, reg_h=1024)
        ctrl.comp_current_uid = "1_1"
        ctrl.comp_next_uid = "2_1"

        # 1. Test unrefined warning in SmarAct guidance
        ctrl.update_comp_smaract_guidance()
        self.assertIn("WARNING", ctrl.window.lbl_comp_refine_status.text())
        self.assertIn("Site 2", ctrl.window.lbl_comp_refine_status.text())
        print("  PASSED: SmarAct guidance displays unrefined site warning!")

        # 2. Test unrefined site move exclusion from calibration
        ctrl.window.spin_comp_act_dx.setValue(10.0)
        ctrl.window.spin_comp_act_dy.setValue(20.0)
        initial_history_len = len(ctrl.comp_actual_history)
        ctrl.on_comp_confirm_trans()
        self.assertEqual(len(ctrl.comp_actual_history), initial_history_len)
        print("  PASSED: Unrefined site move correctly excluded from calibration history!")

        # 3. Test in-memory caching during rotation
        ctrl.comp_region_cached_img = np.zeros((100, 100), dtype=np.uint16)
        ctrl.comp_region_cached_filename = "Region.tif"
        ctrl.comp_region_w = 100
        ctrl.comp_region_h = 100
        with patch("main.read_sem_tiff") as mock_read:
            ctrl.on_comp_rot_slider_changed(450) # 45.0 degrees
            mock_read.assert_not_called()
        print("  PASSED: Rotation slider uses in-memory cache without calling read_sem_tiff!")

if __name__ == "__main__":
    unittest.main()
