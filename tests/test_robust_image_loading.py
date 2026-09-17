import sys
import os
import tempfile
from pathlib import Path
import numpy as np
import tifffile

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PySide6.QtWidgets import QApplication
from models import Sample, ImageRecord, ImageType
from main import AppController, read_sem_tiff

app = QApplication.instance() or QApplication(sys.argv)

def test_read_sem_tiff_formats():
    print("Testing read_sem_tiff on various TIFF formats...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Standard 2D grayscale
        f2d = os.path.join(tmp_dir, "gray.tif")
        data_2d = (np.random.rand(512, 1024) * 65535).astype(np.uint16)
        tifffile.imwrite(f2d, data_2d)
        res_2d = read_sem_tiff(f2d)
        assert res_2d.shape == (512, 1024), f"Expected (512, 1024), got {res_2d.shape}"
        assert res_2d.ndim == 2
        print("  Standard 2D grayscale passed.")

        # 2. 3D RGB TIFF (like Tescan Region1_Closeup)
        frgb = os.path.join(tmp_dir, "rgb.tif")
        data_rgb = (np.random.rand(512, 1024, 3) * 255).astype(np.uint8)
        tifffile.imwrite(frgb, data_rgb, photometric='rgb')
        res_rgb = read_sem_tiff(frgb)
        assert res_rgb.shape == (512, 1024), f"Expected (512, 1024), got {res_rgb.shape}"
        assert res_rgb.ndim == 2
        print("  3D RGB TIFF passed.")

        # 3. 3D RGBA TIFF
        frgba = os.path.join(tmp_dir, "rgba.tif")
        data_rgba = (np.random.rand(256, 512, 4) * 255).astype(np.uint8)
        tifffile.imwrite(frgba, data_rgba, photometric='rgb')
        res_rgba = read_sem_tiff(frgba)
        assert res_rgba.shape == (256, 512), f"Expected (256, 512), got {res_rgba.shape}"
        assert res_rgba.ndim == 2
        print("  3D RGBA TIFF passed.")

    print("  PASSED: read_sem_tiff handles all dimensions cleanly.")

def test_detector_signal_pairing():
    print("Testing get_signal_indices with various detector combinations...")
    ctrl = AppController()
    
    # Case 1: Chamber SE + In-Beam SE (the Region 1 case!)
    site1 = ImageRecord(
        filename="Site1.tif",
        metadata={
            "MAIN": {"ViewFieldsCountX": "2"},
            "SEM": {
                "Detector0": "SE",
                "Detector1": "In-Beam SE"
            }
        }
    )
    ctrl.current_site = site1
    ctrl.current_site_signals = [np.zeros((100, 100)), np.zeros((100, 100))]
    se_idx, bse_idx = ctrl.get_signal_indices()
    assert se_idx == 1, f"Expected In-Beam SE to be se_idx (1), got {se_idx}"
    assert bse_idx == 0, f"Expected Chamber SE to be bse_idx (0), got {bse_idx}"
    assert se_idx != bse_idx, "Signals must not collide!"
    print("  SE + In-Beam SE pairing: PASSED (se_idx=1, bse_idx=0).")

    # Case 2: Chamber SE + In-Beam BSE (standard double image)
    site2 = ImageRecord(
        filename="Site2.tif",
        metadata={
            "MAIN": {"ViewFieldsCountX": "2"},
            "SEM": {
                "Detector0": "SE",
                "Detector1": "In-Beam BSE"
            }
        }
    )
    ctrl.current_site = site2
    se_idx, bse_idx = ctrl.get_signal_indices()
    assert se_idx == 0, f"Expected SE to be se_idx (0), got {se_idx}"
    assert bse_idx == 1, f"Expected In-Beam BSE to be bse_idx (1), got {bse_idx}"
    assert se_idx != bse_idx
    print("  SE + In-Beam BSE pairing: PASSED (se_idx=0, bse_idx=1).")

    # Case 3: Single-detector image
    site3 = ImageRecord(
        filename="Site3.tif",
        metadata={
            "MAIN": {"ViewFieldsCountX": "1"},
            "SEM": {"Detector0": "SE"}
        }
    )
    ctrl.current_site = site3
    ctrl.current_site_signals = [np.zeros((100, 100))]
    se_idx, bse_idx = ctrl.get_signal_indices()
    assert se_idx == 0 and bse_idx == 0
    print("  Single-signal fallback: PASSED.")

def test_adaptive_hysteresis_cross_correlation():
    print("Testing adaptive hysteresis with resolution mismatch (1024 vs 2048)...")
    ctrl = AppController()
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctrl.sample_dir = tmp_dir
        
        # Ground truth shift: 10 pixels at 2048 scale (5 pixels at 1024 scale)
        base = np.zeros((2048, 2048), dtype=np.uint16)
        # Add synthetic particle pattern
        for y, x in [(500, 500), (1000, 1200), (1500, 800)]:
            base[y-50:y+50, x-50:x+50] = 50000

        # img2 (Post, 2048x2048 with 240 strip)
        dy_gt, dx_gt = 12, -8
        post_img = np.roll(np.roll(base, dy_gt, axis=0), dx_gt, axis=1)
        post_file_img = np.zeros((2288, 4096), dtype=np.uint16)
        post_file_img[:2048, :2048] = post_img
        tifffile.imwrite(os.path.join(tmp_dir, "post.tif"), post_file_img)

        # img1 (Region, downsampled to 1024x1024 with 80 strip)
        import cv2
        reg_img = cv2.resize(base, (1024, 1024), interpolation=cv2.INTER_AREA)
        reg_file_img = np.zeros((1104, 2048, 3), dtype=np.uint16)
        reg_file_img[:1024, :1024, 0] = reg_img
        reg_file_img[:1024, :1024, 1] = reg_img
        reg_file_img[:1024, :1024, 2] = reg_img
        tifffile.imwrite(os.path.join(tmp_dir, "reg.tif"), reg_file_img, photometric='rgb')

        reg_rec = ImageRecord(
            filename="reg.tif",
            classification=ImageType.REGION,
            metadata={"MAIN": {"ViewFieldsCountX": "2", "ImageStripSize": "80", "PixelSizeX": "6.0e-7", "PixelSizeY": "6.0e-7"}}
        )
        post_rec = ImageRecord(
            filename="post.tif",
            classification=ImageType.REGION_POST,
            metadata={"MAIN": {"ViewFieldsCountX": "2", "ImageStripSize": "240", "PixelSizeX": "3.0e-7", "PixelSizeY": "3.0e-7"}}
        )

        ctrl.sample = Sample(name="Test", directory_path=tmp_dir, images=[reg_rec, post_rec])
        ctrl.estimate_hysteresis()

        assert ctrl.sample.hysteresis_error_x != 0.0 or ctrl.sample.hysteresis_error_y != 0.0
        # Expected shift in meters: dx_gt * 3.0e-7 = -2.4e-6, dy_gt * 3.0e-7 = 3.6e-6
        print(f"  Calculated Hysteresis: X={ctrl.sample.hysteresis_error_x*1e6:.3f} um, Y={ctrl.sample.hysteresis_error_y*1e6:.3f} um")
        assert abs(abs(ctrl.sample.hysteresis_error_x) - abs(dx_gt * 3.0e-7)) < 1.0e-6
        assert abs(abs(ctrl.sample.hysteresis_error_y) - abs(dy_gt * 3.0e-7)) < 1.0e-6
        print("  Adaptive hysteresis cross-correlation PASSED!")

def test_region1_network_dataset():
    p = r"N:\Vol1-Th\Analytik\Personal\Philipp Vol1 Analy Personal\Ruthenium_Dewetting\260914_P3cc07_Imaging\Region1"
    if not os.path.exists(p):
        print(f"  Skipping live Region1 check (network path {p} not accessible).")
        return
    print("Testing live Region1 dataset loading and display...")
    ctrl = AppController()
    ctrl.sample_dir = p
    ctrl.parse_directory(p)
    assert len(ctrl.sample.images) == 29
    
    reg = next(img for img in ctrl.sample.images if img.classification == ImageType.REGION)
    ctrl.load_region_view(reg)
    assert len(ctrl.current_region_signals) == 2
    assert ctrl.current_region_signals[0].shape == (1024, 1024)
    assert ctrl.current_region_signals[1].shape == (1024, 1024)
    assert len(ctrl.region_rois) > 0
    print(f"  Region1_Closeup loaded successfully with 2 signals and {len(ctrl.region_rois)} site ROIs!")

    site1 = next(img for img in ctrl.sample.images if 'Site1.' in img.filename)
    ctrl.load_site_view(site1)
    se_idx, bse_idx = ctrl.get_signal_indices()
    assert se_idx == 1 and bse_idx == 0, f"Expected (1, 0), got ({se_idx}, {bse_idx})"
    print(f"  Site1 detectors mapped: In-Beam SE (index {se_idx}) and Chamber SE (index {bse_idx}).")
    print("  Live Region1 dataset verification PASSED!")

if __name__ == "__main__":
    test_read_sem_tiff_formats()
    test_detector_signal_pairing()
    test_adaptive_hysteresis_cross_correlation()
    test_region1_network_dataset()
    print("\nALL ROBUST IMAGE LOADING TESTS PASSED!")
