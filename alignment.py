import cv2
import numpy as np
import tifffile
from PySide6.QtCore import QRunnable, QObject, Signal, Slot

def align_site_to_region(region_img: np.ndarray, region_px: float, region_py: float,
                         site_img: np.ndarray, site_px: float, site_py: float,
                         expected_cx_phys: float, expected_cy_phys: float) -> tuple[float, float, float]:
    """
    Aligns a site image to a region image using Morphological White Top-Hat filtering
    to isolate bright particles, followed by template matching with subpixel quadratic peak interpolation.
    All physical coordinates are relative to the top-left of the region image (0,0).
    Returns:
        (offset_x_phys, offset_y_phys, confidence) - The physical offset from the expected position.
    """
    if region_img.ndim == 3:
        region_img = region_img[..., 0]
    if site_img.ndim == 3:
        site_img = site_img[..., 0]
        
    region_img = region_img.astype(np.float32)
    site_img = site_img.astype(np.float32)

    # 1. Downsample Site Image to match Region pixel size
    scale_x = site_px / region_px
    scale_y = site_py / region_py
    
    new_w = int(site_img.shape[1] * scale_x)
    new_h = int(site_img.shape[0] * scale_y)
    
    if new_w <= 0 or new_h <= 0:
        return 0.0, 0.0, 0.0
        
    # Top-hat on site image before resizing to isolate small bright particles from background
    kernel_site = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    site_tophat = cv2.morphologyEx(site_img, cv2.MORPH_TOPHAT, kernel_site)
    template = cv2.resize(site_tophat.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 2. Extract Search Area from Region
    center_x_px = int(expected_cx_phys / region_px)
    center_y_px = int(expected_cy_phys / region_py)
    
    # Search area padding (100% of template size)
    pad_x = int(new_w * 1.0)
    pad_y = int(new_h * 1.0)
    
    start_x = max(0, center_x_px - new_w // 2 - pad_x)
    end_x = min(region_img.shape[1], center_x_px + new_w // 2 + pad_x)
    start_y = max(0, center_y_px - new_h // 2 - pad_y)
    end_y = min(region_img.shape[0], center_y_px + new_h // 2 + pad_y)
    
    search_area = region_img[start_y:end_y, start_x:end_x]
    
    if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
        return 0.0, 0.0, 0.0 # Cannot align, template is bigger than search area (out of bounds)
        
    # Top-hat on search area to isolate bright particles in Region
    kernel_reg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    search_tophat = cv2.morphologyEx(search_area, cv2.MORPH_TOPHAT, kernel_reg).astype(np.float32)
    
    # 3. Template Matching
    res = cv2.matchTemplate(search_tophat, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    confidence = float(max_val)
    mx, my = max_loc
    
    # 4. Sub-pixel quadratic interpolation around peak
    if 0 < mx < res.shape[1] - 1 and 0 < my < res.shape[0] - 1:
        denom_x = (2 * (2 * res[my, mx] - res[my, mx + 1] - res[my, mx - 1]))
        denom_y = (2 * (2 * res[my, mx] - res[my + 1, mx] - res[my - 1, mx]))
        dx_sub = float((res[my, mx + 1] - res[my, mx - 1]) / denom_x) if denom_x != 0 else 0.0
        dy_sub = float((res[my + 1, mx] - res[my - 1, mx]) / denom_y) if denom_y != 0 else 0.0
        # Guard against degenerate parabola fits
        dx_sub = float(np.clip(dx_sub, -1.0, 1.0))
        dy_sub = float(np.clip(dy_sub, -1.0, 1.0))
    else:
        dx_sub, dy_sub = 0.0, 0.0
    
    # Convert matched top-left to matched center in region pixel coordinates
    matched_cx_px = start_x + mx + dx_sub + new_w / 2.0
    matched_cy_px = start_y + my + dy_sub + new_h / 2.0
    
    # Calculate pixel shift from expected center
    shift_x_px = matched_cx_px - center_x_px
    shift_y_px = matched_cy_px - center_y_px
    
    # Convert to physical shift
    offset_x_phys = float(shift_x_px * region_px)
    offset_y_phys = float(shift_y_px * region_py)
    
    return offset_x_phys, offset_y_phys, confidence

class AlignmentSignals(QObject):
    finished = Signal(object, float, float, float) # site_record, offset_x_phys, offset_y_phys, confidence
    error = Signal(object, str)
    all_done = Signal()

class AlignmentWorker(QRunnable):
    def __init__(self, site_record, region_img, region_px, region_py, expected_cx_phys, expected_cy_phys, site_path):
        super().__init__()
        self.site_record = site_record
        self.region_img = region_img
        self.region_px = region_px
        self.region_py = region_py
        self.expected_cx_phys = expected_cx_phys
        self.expected_cy_phys = expected_cy_phys
        self.site_path = site_path
        self.signals = AlignmentSignals()

    @Slot()
    def run(self):
        try:
            from main import read_sem_tiff
            site_img = read_sem_tiff(self.site_path)
            
            # Extract main signal if multi-signal
            meta_main_site = self.site_record.metadata.get('MAIN', {})
            n_sig = int(meta_main_site.get('ViewFieldsCountX', 1))
            strip = int(meta_main_site.get('ImageStripSize', 0))
            if strip > 0 and site_img.shape[0] > strip:
                site_img = site_img[:-strip, :]
            
            site_img = site_img[:, :site_img.shape[1] // n_sig]
            
            site_px = float(meta_main_site.get('PixelSizeX', 1.0))
            site_py = float(meta_main_site.get('PixelSizeY', 1.0))
            
            off_x, off_y, conf = align_site_to_region(
                self.region_img, self.region_px, self.region_py,
                site_img, site_px, site_py,
                self.expected_cx_phys, self.expected_cy_phys
            )
            
            self.signals.finished.emit(self.site_record, off_x, off_y, conf)
        except Exception as e:
            self.signals.error.emit(self.site_record, str(e))
