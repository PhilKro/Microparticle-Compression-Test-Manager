import sys
import os
import json
import glob
import configparser
from pathlib import Path
from datetime import datetime
import re
import math
import tifffile
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QMenu, QProgressDialog
from PySide6.QtGui import QStandardItemModel, QStandardItem, QKeySequence, QShortcut
from PySide6.QtCore import Qt, QRectF, QThreadPool, QTimer
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')

from models import Sample, ImageRecord, ImageType, Particle, CompressionTestRecord, CompressionSession
from ui import MainWindow, HelpDialog
from alignment import AlignmentWorker
from compression_planner import (
    ParticleGlobalInfo, extract_all_particles, get_tilt_transform,
    map_point_tilt, map_polygon_tilt, predict_smaract_translation,
    refine_orientation_scale, optimize_particle_route,
    CalibrationResult, robust_refine_orientation_scale
)
from measurement import (
    fit_particle_ransac, get_ellipse_points,
    fit_facet_polygon_gradient_ransac, fit_facet_polygon_dog_ransac,
    fit_facet_ellipse_ransac, get_polygon_points
)
from roi import (
    SHAPE_MODES, HexagonFacetROI, GeneralPolygonROI, EllipseFacetROI,
    create_roi_for_particle, compute_facet_parameters, compute_facet_parameters_ellipse
)
import scipy.ndimage as ndimage
import cv2

def read_sem_tiff(filepath: str) -> np.ndarray:
    """
    Safely reads a SEM TIFF file, converting:
    - Standard 2D grayscale arrays (H, W) -> unchanged
    - 3D RGB / RGBA arrays (H, W, 3) or (H, W, 4) -> 2D luminance grayscale
    - 3D multi-page / series arrays (pages, H, W) -> 2D page 0
    - Higher dimensions or singleton dimensions -> squeezed to 2D
    Returns a 2D numpy array (H, W).
    """
    img = tifffile.imread(filepath)
    if img is None:
        raise ValueError(f"Failed to load image from {filepath}")
    img = np.squeeze(img)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] in (3, 4):
            if np.array_equal(img[..., 0], img[..., 1]):
                return img[..., 0]
            else:
                rgb = img[..., :3].astype(float)
                gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
                return np.round(gray).astype(img.dtype)
        elif img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            return img[0]
        else:
            return img[:, :, 0]
    while img.ndim > 2:
        img = img[0]
    return img

def calculate_scale_size(view_w: float) -> float:
    if view_w <= 0:
        return 1e-6
    target = view_w * 0.18
    exponent = np.floor(np.log10(target))
    fraction = target / (10.0 ** exponent)
    if fraction < 1.5:
        base = 1.0
    elif fraction < 3.5:
        base = 2.0
    elif fraction < 7.5:
        base = 5.0
    else:
        base = 10.0
    return float(base * (10.0 ** exponent))

class AppController:
    def __init__(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = MainWindow()
        self.sample = None
        self.sample_dir = ""
        self.is_dirty = False
        
        self.current_signal_index = 0
        self.current_region_signals = []
        self.current_site_signals = []
        
        self.current_site = None
        self.selected_particle_center_px = None
        self.particle_plot_items = []
        self.particle_center_marker = None
        
        self.roi_to_site = {}
        self.undo_stack = []
        
        self.active_facet_roi = None
        self.active_particle_id = None
        
        # Micro-Compression Testing State (Tab 3)
        self.comp_particles: Dict[str, ParticleGlobalInfo] = {}
        self.comp_current_uid: Optional[str] = None
        self.comp_next_uid: Optional[str] = None
        self.comp_tilt_deg: float = 70.0
        self.comp_rot_deg: float = 0.0
        self.comp_scale: float = 1.0
        self.comp_invert_x: bool = False
        self.comp_invert_y: bool = False
        self.comp_flip_xy: bool = False
        self.comp_actual_history: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.comp_session_filename: str = ""
        self.comp_session_data: Optional[CompressionSession] = None
        self.comp_region_marker_items: List[Any] = []
        self.comp_site_marker_items: List[Any] = []
        self.comp_filter_mode: str = "all"
        self.comp_mode: str = "test"
        self.comp_tour_order: List[str] = []
        self.comp_tour_distance_um: float = 0.0
        self.comp_site_filename: Optional[str] = None
        self._last_tested_uid: Optional[str] = None
        self.comp_initial_rot_deg: float = 0.0
        self.comp_calibration: Optional[CalibrationResult] = None
        self.comp_cumulative_dx: float = 0.0
        self.comp_cumulative_dy: float = 0.0

        self.setup_connections()
        self.window.closeEvent = self.on_close

    def setup_connections(self):
        # File Menu
        self.window.action_open_folder.triggered.connect(lambda: self.load_directory(prompt_if_dirty=True))
        self.window.action_save_session.triggered.connect(self.save_session)
        self.window.action_exit.triggered.connect(self.window.close)
        
        self.window.tree_view.clicked.connect(self.on_tree_item_clicked)
        self.window.tree_view.doubleClicked.connect(self.on_tree_item_double_clicked)
        self.window.tree_view.customContextMenuRequested.connect(self.on_tree_context_menu)
        
        self.window.site_selector.currentIndexChanged.connect(self.on_site_selector_changed)
        if hasattr(self.window, 'btn_next_site'):
            self.window.btn_next_site.clicked.connect(self.on_next_site_clicked)
        
        self.window.tabs.currentChanged.connect(self.on_tab_changed)
        
        # Signal Tabs
        self.window.region_signal_tabs.currentChanged.connect(self.on_region_signal_changed)
        self.window.site_signal_tabs.currentChanged.connect(self.on_site_signal_changed)
        
        # Databar Toggles
        self.window.db_action_wd.toggled.connect(self.toggle_databar_visibility)
        self.window.db_action_hv.toggled.connect(self.toggle_databar_visibility)
        self.window.db_action_mag.toggled.connect(self.toggle_databar_visibility)
        self.window.db_action_date.toggled.connect(self.toggle_databar_visibility)
        self.window.db_action_viewfield.toggled.connect(self.toggle_databar_visibility)
        self.window.db_action_hysteresis.toggled.connect(self.toggle_databar_visibility)
        
        # Undo Shortcut
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self.window)
        self.undo_shortcut.activated.connect(self.undo)
        
        # Auto Align
        self.window.btn_auto_align.clicked.connect(self.auto_align_sites)
        
        # Particle Measurement
        self.window.site_plot.scene().sigMouseClicked.connect(self.on_site_plot_clicked)
        self.window.slider_search_radius.valueChanged.connect(self.on_search_radius_slider_changed)
        self.window.slider_facet_shape.valueChanged.connect(self.on_facet_shape_slider_changed)
        self.window.btn_measure_particle.clicked.connect(self.on_measure_particle_combined)
        self.window.btn_calc_ecd.clicked.connect(self.on_measure_ecd)
        self.window.btn_measure_se.clicked.connect(self.on_measure_se)
        self.window.btn_delete_particle.clicked.connect(self.on_delete_particle)
        self.window.particle_table.selectionModel().selectionChanged.connect(self.on_particle_table_selected)

        # Hotkeys for Measurement and Management
        self.shortcut_measure_space = QShortcut(QKeySequence(Qt.Key_Space), self.window)
        self.shortcut_measure_space.activated.connect(self.on_measure_particle_combined)
        self.shortcut_measure_m = QShortcut(QKeySequence("M"), self.window)
        self.shortcut_measure_m.activated.connect(self.on_measure_particle_combined)
        self.shortcut_delete = QShortcut(QKeySequence.Delete, self.window)
        self.shortcut_delete.activated.connect(self.on_delete_particle)

        # Micro-Compression Tab (Tab 3) Connections
        self.window.spin_comp_tilt.valueChanged.connect(self.on_comp_tilt_changed)
        self.window.slider_comp_rot.valueChanged.connect(self.on_comp_rot_slider_changed)
        self.window.spin_comp_rot.valueChanged.connect(self.on_comp_rot_spin_changed)
        self.window.btn_rot_m90.clicked.connect(lambda: self.on_comp_quick_rotate(-90.0))
        self.window.btn_rot_m10.clicked.connect(lambda: self.on_comp_quick_rotate(-10.0))
        self.window.btn_rot_m1.clicked.connect(lambda: self.on_comp_quick_rotate(-1.0))
        self.window.btn_rot_p1.clicked.connect(lambda: self.on_comp_quick_rotate(1.0))
        self.window.btn_rot_p10.clicked.connect(lambda: self.on_comp_quick_rotate(10.0))
        self.window.btn_rot_p90.clicked.connect(lambda: self.on_comp_quick_rotate(90.0))
        self.window.chk_comp_inv_x.toggled.connect(self.on_comp_polarity_changed)
        self.window.chk_comp_inv_y.toggled.connect(self.on_comp_polarity_changed)
        self.window.btn_comp_flip_xy.toggled.connect(self.on_comp_flip_xy_toggled)
        self.window.spin_comp_scale.valueChanged.connect(self.on_comp_scale_changed)

        self.window.combo_comp_next.currentIndexChanged.connect(self.on_comp_next_combo_changed)
        self.window.btn_comp_auto_next.clicked.connect(self.on_comp_auto_next_clicked)
        self.window.btn_comp_log_test.clicked.connect(self.on_comp_log_test)
        self.window.btn_comp_preview_trans.clicked.connect(lambda: self.set_comp_mode("trans"))
        self.window.btn_comp_back_to_test.clicked.connect(lambda: self.set_comp_mode("test"))
        self.window.btn_comp_reset_trans.clicked.connect(self.on_comp_reset_trans)
        self.window.btn_comp_confirm_trans.clicked.connect(self.on_comp_confirm_trans)

        self.window.comp_particle_table.doubleClicked.connect(self.on_comp_table_double_clicked)
        self.window.btn_comp_set_current.clicked.connect(self.on_comp_set_current_clicked)
        self.window.btn_comp_filter_all.clicked.connect(lambda: self.set_comp_filter("all"))
        self.window.btn_comp_filter_untested.clicked.connect(lambda: self.set_comp_filter("untested"))
        self.window.btn_comp_filter_tested.clicked.connect(lambda: self.set_comp_filter("tested"))

        self.window.btn_comp_autorange_reg.clicked.connect(self.window.comp_region_plot.autoRange)
        self.window.btn_comp_autorange_site.clicked.connect(self.window.comp_site_plot.autoRange)
        self.window.comp_region_plot.scene().sigMouseClicked.connect(self.on_comp_region_plot_clicked)

        # Enter hotkey for compression test logging
        self.shortcut_comp_enter = QShortcut(QKeySequence(Qt.Key_Return), self.window)
        self.shortcut_comp_enter.activated.connect(self.on_comp_hotkey_enter)
        self.shortcut_comp_enter2 = QShortcut(QKeySequence(Qt.Key_Enter), self.window)
        self.shortcut_comp_enter2.activated.connect(self.on_comp_hotkey_enter)

        # Help Menu
        self.window.action_help.triggered.connect(self.show_help_dialog)

    def show_help_dialog(self):
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "help_content.json")
        dlg = HelpDialog(self.window, json_path=json_path)
        dlg.exec()

    def toggle_databar_visibility(self):
        w = self.window
        w.region_lbl_wd.setVisible(w.db_action_wd.isChecked())
        w.region_lbl_hv.setVisible(w.db_action_hv.isChecked())
        w.region_lbl_mag.setVisible(w.db_action_mag.isChecked())
        w.region_lbl_date.setVisible(w.db_action_date.isChecked())
        w.region_lbl_viewfield.setVisible(w.db_action_viewfield.isChecked())
        w.region_lbl_hysteresis.setVisible(w.db_action_hysteresis.isChecked())
        
        w.site_lbl_wd.setVisible(w.db_action_wd.isChecked())
        w.site_lbl_hv.setVisible(w.db_action_hv.isChecked())
        w.site_lbl_mag.setVisible(w.db_action_mag.isChecked())
        w.site_lbl_date.setVisible(w.db_action_date.isChecked())
        w.site_lbl_viewfield.setVisible(w.db_action_viewfield.isChecked())

    def run(self):
        self.window.show()
        self.load_directory()
        return self.app.exec()

    def load_directory(self, prompt_if_dirty=False):
        if prompt_if_dirty and self.is_dirty:
            reply = QMessageBox.question(
                self.window, "Unsaved Changes",
                "You have unsaved changes. Do you want to save them before opening another folder?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            if reply == QMessageBox.Save:
                self.save_session()
            elif reply == QMessageBox.Cancel:
                return False

        dir_path = QFileDialog.getExistingDirectory(self.window, "Select Sample Directory", self.sample_dir or "")
        if not dir_path:
            if not self.sample_dir:
                sys.exit()
            return False
        
        self.sample_dir = dir_path
        self.is_dirty = False
        json_path = os.path.join(self.sample_dir, "sample_data.json")
        
        if os.path.exists(json_path):
            self.load_from_json(json_path)
        else:
            self.parse_directory(self.sample_dir)
            
        self.populate_tree()
        
        # Auto-select and display region image on startup
        for row in range(self.tree_model.rowCount()):
            item = self.tree_model.item(row, 0)
            if item:
                img_idx = item.data()
                if 0 <= img_idx < len(self.sample.images) and self.sample.images[img_idx].classification == ImageType.REGION:
                    idx = self.tree_model.index(row, 0)
                    self.window.tree_view.setCurrentIndex(idx)
                    self.on_tree_item_clicked(idx)
                    QTimer.singleShot(100, self.window.region_plot.autoRange)
                    break
        self.init_compression_tab()
        return True

    def load_from_json(self, json_path: str):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.sample = Sample.model_validate(data)
        except Exception as e:
            QMessageBox.critical(self.window, "Error", f"Failed to load sample_data.json:\n{e}")
            self.sample = Sample(name=os.path.basename(self.sample_dir), directory_path=self.sample_dir)

    def parse_directory(self, dir_path: str):
        self.sample = Sample(name=os.path.basename(dir_path), directory_path=dir_path)
        hdr_files = glob.glob(os.path.join(dir_path, "*.hdr"))
        
        for hdr_path in hdr_files:
            try:
                base_name = os.path.basename(hdr_path).replace("-tif.hdr", "").replace(".hdr", "")
                tif_name = f"{base_name}.tif"
                
                config = configparser.ConfigParser()
                config.optionxform = str 
                config.read(hdr_path)
                
                metadata = {s: dict(config.items(s)) for s in config.sections()}
                
                sx = float(config.get("SEM", "StageX", fallback=0.0))
                sy = float(config.get("SEM", "StageY", fallback=0.0))
                
                date_str = config.get("MAIN", "Date", fallback="")
                time_str = config.get("MAIN", "Time", fallback="")
                acq_time = None
                if date_str and time_str:
                    try:
                        acq_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                
                # Auto-classify
                lower_name = base_name.lower()
                if "location" in lower_name:
                    cls = ImageType.OVERVIEW
                elif "closeup_post" in lower_name:
                    cls = ImageType.REGION_POST
                elif "closeup" in lower_name:
                    cls = ImageType.REGION
                elif "site" in lower_name:
                    cls = ImageType.SITE
                else:
                    cls = ImageType.UNCLASSIFIED
                
                record = ImageRecord(
                    filename=tif_name,
                    classification=cls,
                    acquisition_time=acq_time,
                    stage_x=sx,
                    stage_y=sy,
                    metadata=metadata
                )
                self.sample.images.append(record)
            except Exception as e:
                print(f"Error parsing {hdr_path}: {e}")
                
        # Sort chronologically (handling None gracefully)
        self.sample.images.sort(key=lambda img: img.acquisition_time or datetime.min)
        
        self.estimate_hysteresis()

    def estimate_hysteresis(self):
        region = next((img for img in self.sample.images if img.classification == ImageType.REGION), None)
        region_post = next((img for img in self.sample.images if img.classification == ImageType.REGION_POST), None)
        
        if region and region_post:
            try:
                from skimage.registration import phase_cross_correlation
                tif1 = os.path.join(self.sample_dir, region.filename)
                tif2 = os.path.join(self.sample_dir, region_post.filename)
                
                if os.path.exists(tif1) and os.path.exists(tif2):
                    img1 = read_sem_tiff(tif1)
                    img2 = read_sem_tiff(tif2)
                    
                    meta1 = region.metadata.get('MAIN', {})
                    meta2 = region_post.metadata.get('MAIN', {})
                    
                    n_sig1 = int(meta1.get('ViewFieldsCountX', 1))
                    n_sig2 = int(meta2.get('ViewFieldsCountX', 1))
                    strip1 = int(meta1.get('ImageStripSize', 0))
                    strip2 = int(meta2.get('ImageStripSize', 0))
                    
                    # Strip data bars
                    if strip1 > 0 and img1.shape[0] > strip1:
                        img1 = img1[:-strip1, :]
                    if strip2 > 0 and img2.shape[0] > strip2:
                        img2 = img2[:-strip2, :]
                    
                    w1 = img1.shape[1] // n_sig1
                    w2 = img2.shape[1] // n_sig2
                    
                    # Extract primary detector signal
                    img1 = img1[:, :w1]
                    img2 = img2[:, :w2]
                    
                    px1 = float(meta1.get('PixelSizeX', 0.0))
                    py1 = float(meta1.get('PixelSizeY', 0.0))
                    px2 = float(meta2.get('PixelSizeX', 0.0))
                    py2 = float(meta2.get('PixelSizeY', 0.0))
                    
                    ref_px = px2 if px2 > 0 else px1
                    ref_py = py2 if py2 > 0 else py1
                    
                    # If resolutions differ (e.g. 1024x1024 vs 2048x2048), resample img1 to img2 grid
                    if img1.shape != img2.shape:
                        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
                    shift, error, diffphase = phase_cross_correlation(img1, img2, upsample_factor=10)
                    
                    # shift is [y, x] in img2 pixels
                    self.sample.hysteresis_error_y = shift[0] * ref_py
                    self.sample.hysteresis_error_x = shift[1] * ref_px
                    print(f"Estimated Hysteresis Error: X={self.sample.hysteresis_error_x:.2e} m, Y={self.sample.hysteresis_error_y:.2e} m")
            except ImportError:
                print("scikit-image not installed for hysteresis check.")
            except Exception as e:
                print(f"Error calculating hysteresis: {e}")

    def populate_tree(self):
        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels(["Filename", "Class", "Alignment"])
        
        self.window.site_selector.blockSignals(True)
        self.window.site_selector.clear()
        
        for idx, img in enumerate(self.sample.images):
            item_file = QStandardItem(img.filename)
            item_file.setEditable(False)
            item_file.setData(idx)
            
            item_class = QStandardItem(img.classification.value)
            item_class.setEditable(False)
            
            item_align = QStandardItem(getattr(img, 'alignment_status', 'Pending'))
            item_align.setEditable(False)
            
            self.tree_model.appendRow([item_file, item_class, item_align])
            
            if img.classification == ImageType.SITE:
                self.window.site_selector.addItem(img.filename, idx)
                
        self.window.tree_view.setModel(self.tree_model)
        self.window.tree_view.setColumnWidth(0, 200)
        self.window.site_selector.blockSignals(False)

    def on_tree_context_menu(self, position):
        indexes = self.window.tree_view.selectedIndexes()
        if not indexes:
            return
            
        menu = QMenu()
        for t in ImageType:
            action = menu.addAction(f"Set to {t.value}")
            action.triggered.connect(lambda checked, cls=t: self.change_classification(indexes, cls))
            
        menu.exec_(self.window.tree_view.viewport().mapToGlobal(position))

    def change_classification(self, indexes, new_class: ImageType):
        rows = set(idx.row() for idx in indexes)
        changed = False
        for row in rows:
            idx = self.tree_model.item(row, 0).data()
            if self.sample.images[idx].classification != new_class:
                self.sample.images[idx].classification = new_class
                changed = True
                
        if changed:
            self.populate_tree()
            region = next((img for img in self.sample.images if img.classification == ImageType.REGION), None)
            if region:
                self.load_region_view(region)

    def on_tree_item_clicked(self, index):
        row = index.row()
        item = self.tree_model.item(row, 0)
        img_idx = item.data()
        
        img_record = self.sample.images[img_idx]
        
        if img_record.classification == ImageType.REGION:
            self.load_region_view(img_record)
        elif img_record.classification == ImageType.SITE:
            # Highlight ROI in green
            for roi, (s, text) in self.roi_to_site.items():
                if s == img_record:
                    roi.setPen(pg.mkPen(color='g', width=2))
                    text.setColor('g')
                else:
                    roi.setPen(pg.mkPen(color='r', width=1))
                    text.setColor('r')

    def on_tree_item_double_clicked(self, index):
        row = index.row()
        item = self.tree_model.item(row, 0)
        img_idx = item.data()
        img_record = self.sample.images[img_idx]
        
        if img_record.classification == ImageType.SITE:
            cb_idx = self.window.site_selector.findData(img_idx)
            if cb_idx >= 0:
                self.window.site_selector.setCurrentIndex(cb_idx)
            self.window.tabs.setCurrentIndex(1)
            QTimer.singleShot(50, self.window.site_plot.autoRange)
        elif img_record.classification in (ImageType.REGION, ImageType.OVERVIEW):
            self.load_region_view(img_record)
            self.window.tabs.setCurrentIndex(0)
            QTimer.singleShot(50, self.window.region_plot.autoRange)

    def on_site_selector_changed(self, index):
        if index < 0 or not self.sample: return
        img_idx = self.window.site_selector.itemData(index)
        if img_idx is None or img_idx >= len(self.sample.images): return
        site = self.sample.images[img_idx]
        self.load_site_view(site)

    def on_next_site_clicked(self):
        count = self.window.site_selector.count()
        if count <= 1:
            return
        curr = self.window.site_selector.currentIndex()
        next_idx = (curr + 1) % count
        self.window.site_selector.setCurrentIndex(next_idx)

    def load_region_view(self, region: ImageRecord):
        if not hasattr(self, 'region_rois'):
            self.region_rois = []
            
        for roi, text in self.region_rois:
            try:
                self.window.region_plot.removeItem(roi)
                self.window.region_plot.removeItem(text)
            except: pass
        self.region_rois.clear()
        self.roi_to_site.clear()

        tif_path = os.path.join(self.sample_dir, region.filename)
        if os.path.exists(tif_path):
            try:
                img = read_sem_tiff(tif_path)
                
                meta_main = region.metadata.get('MAIN', {})
                meta_sem = region.metadata.get('SEM', {})
                
                n_signals = int(meta_main.get('ViewFieldsCountX', 1))
                strip_size = int(meta_main.get('ImageStripSize', 0))
                px = float(meta_main.get('PixelSizeX', 1.0))
                py = float(meta_main.get('PixelSizeY', 1.0))
                
                if strip_size > 0 and img.shape[0] > strip_size:
                    img = img[:-strip_size, :]
                
                H, W = img.shape
                w = W // n_signals
                
                self.current_region_signals = [img[:, i*w:(i+1)*w] for i in range(n_signals)]
                
                self.window.region_signal_tabs.blockSignals(True)
                while self.window.region_signal_tabs.count() > 0:
                    self.window.region_signal_tabs.removeTab(0)
                
                for i in range(n_signals):
                    name = meta_sem.get(f'Detector{i}', f'Signal {i+1}')
                    self.window.region_signal_tabs.addTab(name)
                
                if n_signals > 1:
                    self.window.region_signal_tabs.show()
                else:
                    self.window.region_signal_tabs.hide()
                    
                sig_idx = self.current_signal_index if self.current_signal_index < n_signals else 0
                self.window.region_signal_tabs.setCurrentIndex(sig_idx)
                self.window.region_signal_tabs.blockSignals(False)

                self.current_region_record = region

                # 1. Set image FIRST
                self.window.region_image_item.setImage(self.current_region_signals[sig_idx], autoLevels=False)
                # 2. Set physical rect SECOND
                self.window.region_image_item.setRect(QRectF(0, 0, w * px, H * py))
                
                view_w = w * px
                scale_size = calculate_scale_size(view_w)
                self.window.region_scalebar.size = scale_size
                self.window.region_scalebar.text.setText(pg.siFormat(scale_size, suffix='m'))
                self.window.region_scalebar.text.setColor('w')
                self.window.region_scalebar.updateBar()

                self.update_databar(region.metadata, 'region')
                
                # Overlay sites
                if px > 0 and py > 0:
                    cx = w * px / 2.0
                    cy = H * py / 2.0
                    
                    sites = [s for s in self.sample.images if s.classification == ImageType.SITE]
                    for site in sites:
                        match = re.search(r"Site(\d+)", site.filename)
                        site_num = match.group(1) if match else "?"
                        
                        meta_main_site = site.metadata.get('MAIN', {})
                        px_site = float(meta_main_site.get('PixelSizeX', 1.0))
                        py_site = float(meta_main_site.get('PixelSizeY', 1.0))
                        site_w = 2048 * px_site
                        site_h = 2048 * py_site
                        
                        dx = site.stage_x - region.stage_x
                        dy = site.stage_y - region.stage_y
                        
                        roi_x = cx - dx - site_w / 2.0 + site.offset_x_pixels * px_site
                        roi_y = cy + dy - site_h / 2.0 + site.offset_y_pixels * py_site
                        
                        roi = pg.RectROI([roi_x, roi_y], [site_w, site_h], pen=pg.mkPen(color='r', width=1), movable=True)
                        roi.setZValue(10)
                        self.window.region_plot.addItem(roi)
                        
                        text = pg.TextItem(site_num, color='r', anchor=(0, 1))
                        text.setPos(roi_x, roi_y)
                        text.setZValue(10)
                        self.window.region_plot.addItem(text)
                        
                        self.region_rois.append((roi, text))
                        
                        self.roi_to_site[roi] = (site, text)
                        roi.sigRegionChangeFinished.connect(self.on_roi_dragged)

                # 3. autoRange AFTER all items (image + ROIs) are in the plot
                self.window.region_plot.autoRange()
                        
            except Exception as e:
                print(f"Error loading region image: {e}")

    def on_roi_dragged(self, roi):
        site, text = self.roi_to_site.get(roi, (None, None))
        if not site: return
        
        region = next((r for r in self.sample.images if r.classification == ImageType.REGION), None)
        if not region: return
        
        meta_main = region.metadata.get('MAIN', {})
        n_signals = int(meta_main.get('ViewFieldsCountX', 1))
        px = float(meta_main.get('PixelSizeX', 1.0))
        py = float(meta_main.get('PixelSizeY', 1.0))
        W = self.current_region_signals[0].shape[1] if self.current_region_signals else 2048
        H = self.current_region_signals[0].shape[0] if self.current_region_signals else 2048
        
        cx = W * px / 2.0
        cy = H * py / 2.0
        
        dx = site.stage_x - region.stage_x
        dy = site.stage_y - region.stage_y
        
        meta_main_site = site.metadata.get('MAIN', {})
        px_site = float(meta_main_site.get('PixelSizeX', 1.0))
        py_site = float(meta_main_site.get('PixelSizeY', 1.0))
        site_w = 2048 * px_site
        site_h = 2048 * py_site
        
        base_x = cx - dx - site_w / 2.0
        base_y = cy + dy - site_h / 2.0
        
        new_pos = roi.pos()
        new_x, new_y = new_pos.x(), new_pos.y()
        
        # Calculate new pixel offsets
        new_offset_x_pixels = (new_x - base_x) / px_site
        new_offset_y_pixels = (new_y - base_y) / py_site
        
        # Push to undo stack
        self.undo_stack.append({
            'site': site,
            'roi': roi,
            'text': text,
            'old_x': site.offset_x_pixels,
            'old_y': site.offset_y_pixels,
            'old_status': getattr(site, 'alignment_status', 'Pending')
        })
        
        site.offset_x_pixels = new_offset_x_pixels
        site.offset_y_pixels = new_offset_y_pixels
        site.alignment_status = "Manual"
        self.is_dirty = True
        
        self.tree_model.blockSignals(True)
        for row in range(self.tree_model.rowCount()):
            if self.tree_model.item(row, 0).data() == self.sample.images.index(site):
                self.tree_model.item(row, 2).setText("Manual")
                break
        self.tree_model.blockSignals(False)
        
        text.setPos(new_x, new_y)

    def undo(self):
        if not self.undo_stack:
            return
            
        action = self.undo_stack.pop()
        site = action['site']
        roi = action['roi']
        text = action['text']
        
        site.offset_x_pixels = action['old_x']
        site.offset_y_pixels = action['old_y']
        
        if 'old_status' in action:
            site.alignment_status = action['old_status']
            for row in range(self.tree_model.rowCount()):
                if self.tree_model.item(row, 0).data() == self.sample.images.index(site):
                    self.tree_model.item(row, 2).setText(site.alignment_status)
                    break
        
        # Recalculate position
        region = next((r for r in self.sample.images if r.classification == ImageType.REGION), None)
        if region:
            meta_main = region.metadata.get('MAIN', {})
            px = float(meta_main.get('PixelSizeX', 1.0))
            py = float(meta_main.get('PixelSizeY', 1.0))
            W = self.current_region_signals[0].shape[1] if self.current_region_signals else 2048
            H = self.current_region_signals[0].shape[0] if self.current_region_signals else 2048
            
            cx = W * px / 2.0
            cy = H * py / 2.0
            
            dx = site.stage_x - region.stage_x
            dy = site.stage_y - region.stage_y
            
            px_site = float(site.metadata.get('MAIN', {}).get('PixelSizeX', 1.0))
            py_site = float(site.metadata.get('MAIN', {}).get('PixelSizeY', 1.0))
            site_w = 2048 * px_site
            site_h = 2048 * py_site
            
            roi_x = cx - dx - site_w / 2.0 + site.offset_x_pixels * px_site
            roi_y = cy + dy - site_h / 2.0 + site.offset_y_pixels * py_site
            
            roi.blockSignals(True)
            roi.setPos([roi_x, roi_y])
            text.setPos(roi_x, roi_y)
            roi.blockSignals(False)

    def load_site_view(self, site: ImageRecord):
        self.current_site = site
        self.selected_particle_center_px = None
        self.clear_active_facet_roi()
        if self.particle_center_marker is not None:
            try:
                self.window.site_plot.removeItem(self.particle_center_marker)
            except Exception:
                pass
            self.particle_center_marker = None
        self.window.lbl_measure_hint.setText("Click image to select particle center:")

        tif_path = os.path.join(self.sample_dir, site.filename)
        if os.path.exists(tif_path):
            try:
                img = read_sem_tiff(tif_path)
                
                meta_main = site.metadata.get('MAIN', {})
                meta_sem = site.metadata.get('SEM', {})
                
                n_signals = int(meta_main.get('ViewFieldsCountX', 1))
                strip_size = int(meta_main.get('ImageStripSize', 0))
                px = float(meta_main.get('PixelSizeX', 1.0))
                py = float(meta_main.get('PixelSizeY', 1.0))
                
                if strip_size > 0 and img.shape[0] > strip_size:
                    img = img[:-strip_size, :]
                
                H, W = img.shape
                w = W // n_signals
                
                self.current_site_signals = [img[:, i*w:(i+1)*w] for i in range(n_signals)]
                
                self.window.site_signal_tabs.blockSignals(True)
                while self.window.site_signal_tabs.count() > 0:
                    self.window.site_signal_tabs.removeTab(0)
                
                for i in range(n_signals):
                    name = meta_sem.get(f'Detector{i}', f'Signal {i+1}')
                    self.window.site_signal_tabs.addTab(name)
                
                if n_signals > 1:
                    self.window.site_signal_tabs.show()
                else:
                    self.window.site_signal_tabs.hide()
                    
                sig_idx = self.current_signal_index if self.current_signal_index < n_signals else 0
                self.window.site_signal_tabs.setCurrentIndex(sig_idx)
                self.window.site_signal_tabs.blockSignals(False)

                # 1. Set image FIRST
                self.window.site_image_item.setImage(self.current_site_signals[sig_idx], autoLevels=False)
                # 2. Set physical rect SECOND
                self.window.site_image_item.setRect(QRectF(0, 0, w * px, H * py))
                
                view_w = w * px
                scale_size = calculate_scale_size(view_w)
                self.window.site_scalebar.size = scale_size
                self.window.site_scalebar.text.setText(pg.siFormat(scale_size, suffix='m'))
                self.window.site_scalebar.text.setColor('w')
                self.window.site_scalebar.updateBar()

                self.update_databar(site.metadata, 'site')
                
                # Load particle table & overlays
                self.populate_particle_table()
                self.redraw_particle_overlays()
                
                # 3. Auto-range to expand image cleanly to view window
                self.window.site_plot.autoRange()
                
            except Exception as e:
                print(f"Error loading site image: {e}")

    def on_region_signal_changed(self, index):
        if index >= 0 and index < len(self.current_region_signals):
            self.current_signal_index = index
            self.window.region_image_item.setImage(self.current_region_signals[index], autoLevels=False)
            if hasattr(self, 'current_region_record') and self.current_region_record:
                meta = self.current_region_record.metadata.get('MAIN', {})
                px = float(meta.get('PixelSizeX', 1.0))
                py = float(meta.get('PixelSizeY', 1.0))
                H, W = self.current_region_signals[index].shape[:2]
                self.window.region_image_item.setRect(QRectF(0, 0, W * px, H * py))
            
    def on_site_signal_changed(self, index):
        if index >= 0 and index < len(self.current_site_signals):
            self.current_signal_index = index
            self.window.site_image_item.setImage(self.current_site_signals[index], autoLevels=False)
            if self.current_site:
                meta = self.current_site.metadata.get('MAIN', {})
                px = float(meta.get('PixelSizeX', 1.0))
                py = float(meta.get('PixelSizeY', 1.0))
                H, W = self.current_site_signals[index].shape[:2]
                self.window.site_image_item.setRect(QRectF(0, 0, W * px, H * py))

    def update_databar(self, metadata, tab_type):
        meta_sem = metadata.get('SEM', {})
        meta_main = metadata.get('MAIN', {})
        
        wd = meta_sem.get('WD', '-')
        if wd != '-':
            try: wd = f"{float(wd)*1e3:.2f} mm"
            except: pass
            
        hv = meta_sem.get('HV', '-')
        if hv != '-':
            try: hv = f"{float(hv)/1e3:.1f} kV"
            except: pass
            
        mag = meta_main.get('Magnification', '-')
        date = meta_main.get('Date', '-')
        
        viewfield = "-"
        if tab_type == 'region':
            viewfield = f"{self.window.region_image_item.boundingRect().width()*1e6:.1f} µm"
        else:
            viewfield = f"{self.window.site_image_item.boundingRect().width()*1e6:.1f} µm"

        hyst_str = "Hysteresis Error: N/A"
        if self.sample and (self.sample.hysteresis_error_x != 0 or self.sample.hysteresis_error_y != 0):
            hy = self.sample.hysteresis_error_y * 1e9
            hx = self.sample.hysteresis_error_x * 1e9
            hyst_str = f"Hysteresis Error: X={hx:.1f}nm, Y={hy:.1f}nm"

        w = self.window
        if tab_type == 'region':
            w.region_lbl_wd.setText(f"WD: {wd}")
            w.region_lbl_hv.setText(f"HV: {hv}")
            w.region_lbl_mag.setText(f"Mag: {mag}x")
            w.region_lbl_date.setText(f"Date: {date}")
            w.region_lbl_viewfield.setText(f"Viewfield: {viewfield}")
            w.region_lbl_hysteresis.setText(hyst_str)
        else:
            w.site_lbl_wd.setText(f"WD: {wd}")
            w.site_lbl_hv.setText(f"HV: {hv}")
            w.site_lbl_mag.setText(f"Mag: {mag}x")
            w.site_lbl_date.setText(f"Date: {date}")
            w.site_lbl_viewfield.setText(f"Viewfield: {viewfield}")

    def auto_align_sites(self):
        region = next((r for r in self.sample.images if r.classification == ImageType.REGION), None)
        if not region: return
        
        tif_path = os.path.join(self.sample_dir, region.filename)
        if not os.path.exists(tif_path): return
        
        from alignment import align_site_to_region
        import numpy as np
        
        img = read_sem_tiff(tif_path)
        meta_main = region.metadata.get('MAIN', {})
        n_sig = int(meta_main.get('ViewFieldsCountX', 1))
        strip = int(meta_main.get('ImageStripSize', 0))
        
        def get_signal(image, n, strip_sz, sig_idx):
            if strip_sz > 0 and image.shape[0] > strip_sz:
                image = image[:-strip_sz, :]
            w = image.shape[1] // n
            return image[:, sig_idx*w : (sig_idx+1)*w]
                
        img1 = get_signal(img, n_sig, strip, 0)
        img2 = get_signal(img, n_sig, strip, 1) if n_sig > 1 else None
        
        px = float(meta_main.get('PixelSizeX', 1.0))
        py = float(meta_main.get('PixelSizeY', 1.0))
        
        sites = [s for s in self.sample.images if s.classification == ImageType.SITE]
        if not sites: return
        
        progress = QProgressDialog("Aligning Sites...", "Cancel", 0, len(sites), self.window)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        for i, site in enumerate(sites):
            progress.setValue(i)
            if progress.wasCanceled():
                break
                
            dx = site.stage_x - region.stage_x
            dy = site.stage_y - region.stage_y
            cx_phys = (img1.shape[1] * px) / 2.0 - dx
            cy_phys = (img1.shape[0] * py) / 2.0 + dy
            
            try:
                site_path = os.path.join(self.sample_dir, site.filename)
                site_img = read_sem_tiff(site_path)
                meta_main_site = site.metadata.get('MAIN', {})
                n_sig_site = int(meta_main_site.get('ViewFieldsCountX', 1))
                strip_site = int(meta_main_site.get('ImageStripSize', 0))
                
                site_img1 = get_signal(site_img, n_sig_site, strip_site, 0)
                site_img2 = get_signal(site_img, n_sig_site, strip_site, 1) if n_sig_site > 1 else None
                    
                px_site = float(meta_main_site.get('PixelSizeX', 1.0))
                py_site = float(meta_main_site.get('PixelSizeY', 1.0))
                
                # Prioritize BSE (detector 1) for strong atomic number particle contrast, fallback to SE (detector 0)
                primary_reg = img2 if img2 is not None else img1
                primary_site = site_img2 if site_img2 is not None else site_img1
                
                off_x, off_y, conf = align_site_to_region(
                    primary_reg, px, py, primary_site, px_site, py_site, cx_phys, cy_phys
                )
                
                # If BSE had low confidence, try SE as fallback
                if conf < 0.40 and img2 is not None and site_img1 is not None:
                    off_x_se, off_y_se, conf_se = align_site_to_region(
                        img1, px, py, site_img1, px_site, py_site, cx_phys, cy_phys
                    )
                    if conf_se > conf:
                        off_x, off_y, conf = off_x_se, off_y_se, conf_se
                
                site_w_phys = site_img1.shape[1] * px_site
                site_h_phys = site_img1.shape[0] * py_site
                shift_mag = np.sqrt(off_x**2 + off_y**2)
                max_allowed = 2.0 * max(site_w_phys, site_h_phys)
                
                if conf == 0.0:
                    site.alignment_status = "Out of Bounds"
                elif conf >= 0.40 and shift_mag <= max_allowed:
                    site.alignment_status = f"Refined ({conf:.2f})"
                    site.offset_x_pixels = off_x / px_site
                    site.offset_y_pixels = off_y / py_site
                else:
                    site.alignment_status = f"Failed ({conf:.2f})"
                        
            except Exception as e:
                print(f"Error aligning {site.filename}: {e}")
                site.alignment_status = "Error"
                
        progress.setValue(len(sites))
        self.is_dirty = True
        self.populate_tree()
        
        # Redraw region view to instantly show new alignments
        if region:
            self.load_region_view(region)

    def on_tab_changed(self, index):
        if index == 0:
            QTimer.singleShot(30, self.window.region_plot.autoRange)
            QTimer.singleShot(40, self.window.region_scalebar.updateBar)
        elif index == 1:
            QTimer.singleShot(30, self.window.site_plot.autoRange)
            QTimer.singleShot(40, self.window.site_scalebar.updateBar)
        elif index == 2:
            self.init_compression_tab()
            QTimer.singleShot(30, self.window.comp_region_plot.autoRange)
            QTimer.singleShot(30, self.window.comp_site_plot.autoRange)

    # -------------------------------------------------------------------------
    # Micro-Compression Testing Methods (Tab 3)
    # -------------------------------------------------------------------------
    def init_compression_tab(self):
        if not self.sample:
            return
        if not self.comp_session_filename:
            now = datetime.now()
            self.comp_session_filename = now.strftime("compression_test_%y%m%d%H.json")
        self.load_or_create_compression_session()
        self.populate_compression_particles()

    def load_or_create_compression_session(self):
        if not self.sample_dir:
            return
        session_path = os.path.join(self.sample_dir, self.comp_session_filename)
        if os.path.exists(session_path):
            try:
                with open(session_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.comp_session_data = CompressionSession(**data)
                for t in self.comp_session_data.tests:
                    if t.particle_uid in self.comp_particles:
                        self.comp_particles[t.particle_uid].tested = True
                        self.comp_particles[t.particle_uid].test_timestamp = t.timestamp

                if hasattr(self.comp_session_data, 'scale_factor') and self.comp_session_data.scale_factor:
                    self.comp_scale = float(self.comp_session_data.scale_factor)
                    self.window.spin_comp_scale.blockSignals(True)
                    self.window.spin_comp_scale.setValue(self.comp_scale)
                    self.window.spin_comp_scale.blockSignals(False)

                if hasattr(self.comp_session_data, 'rotation_angle_deg') and self.comp_session_data.rotation_angle_deg:
                    self.comp_rot_deg = float(self.comp_session_data.rotation_angle_deg)
                    self.window.spin_comp_rot.blockSignals(True)
                    self.window.spin_comp_rot.setValue(self.comp_rot_deg)
                    self.window.spin_comp_rot.blockSignals(False)
                    self.window.slider_comp_rot.blockSignals(True)
                    self.window.slider_comp_rot.setValue(int(round(self.comp_rot_deg * 10)))
                    self.window.slider_comp_rot.blockSignals(False)

                # Reconstruct cumulative coordinates and history
                self.comp_cumulative_dx = sum(t.smaract_actual_dx_um or 0.0 for t in self.comp_session_data.tests)
                self.comp_cumulative_dy = sum(t.smaract_actual_dy_um or 0.0 for t in self.comp_session_data.tests)
                for t in self.comp_session_data.tests:
                    if t.from_particle_uid and t.from_particle_uid in self.comp_particles and t.particle_uid in self.comp_particles:
                        p_from = self.comp_particles[t.from_particle_uid]
                        p_to = self.comp_particles[t.particle_uid]
                        f_vec = (p_to.flat_x_um - p_from.flat_x_um, p_to.flat_y_um - p_from.flat_y_um)
                        a_vec = (t.smaract_actual_dx_um or 0.0, t.smaract_actual_dy_um or 0.0)
                        if math.hypot(a_vec[0], a_vec[1]) > 1e-4:
                            self.comp_actual_history.append((f_vec, a_vec))

                if self.comp_actual_history:
                    flats = [h[0] for h in self.comp_actual_history]
                    acts = [h[1] for h in self.comp_actual_history]
                    self.comp_calibration = robust_refine_orientation_scale(
                        flats, acts,
                        prior_rot_deg=self.comp_initial_rot_deg or self.comp_rot_deg,
                        prior_scale=1.0,
                        invert_x=self.comp_invert_x,
                        invert_y=self.comp_invert_y,
                        flip_xy=self.comp_flip_xy
                    )
                    self.update_comp_calibration_badge()
            except Exception as e:
                print(f"Error loading existing compression session: {e}")
                self.comp_session_data = CompressionSession(
                    session_id=Path(self.comp_session_filename).stem,
                    sample_name=self.sample.name if self.sample else "",
                    sample_directory=self.sample_dir,
                    created_at=datetime.now().isoformat(),
                    tilt_angle_deg=self.comp_tilt_deg,
                    rotation_angle_deg=self.comp_rot_deg,
                    scale_factor=self.comp_scale
                )
        else:
            self.comp_session_data = CompressionSession(
                session_id=Path(self.comp_session_filename).stem,
                sample_name=self.sample.name if self.sample else "",
                sample_directory=self.sample_dir,
                created_at=datetime.now().isoformat(),
                tilt_angle_deg=self.comp_tilt_deg,
                rotation_angle_deg=self.comp_rot_deg,
                scale_factor=self.comp_scale
            )
            self.window.lbl_comp_session_file.setText(f"{self.comp_session_filename} (created on first test)")

    def save_compression_session(self):
        if not self.comp_session_data or not self.sample_dir:
            return
        # Only create/write the file if at least one test has been logged,
        # or if the file already exists on disk from an earlier session
        session_path = os.path.join(self.sample_dir, self.comp_session_filename) if self.comp_session_filename else ""
        if not self.comp_session_data.tests and (not session_path or not os.path.exists(session_path)):
            return
        try:
            with open(session_path, 'w', encoding='utf-8') as f:
                f.write(self.comp_session_data.model_dump_json(indent=2))
            self.window.lbl_comp_session_file.setText(self.comp_session_filename)
        except Exception as e:
            print(f"Error saving compression session: {e}")

    def reoptimize_comp_tour(self, start_uid: Optional[str] = None):
        """
        Computes the shortest path route (Greedy + 2-Opt) through remaining untested particles,
        automatically setting the immediate next target particle.
        """
        if not self.comp_particles:
            self.comp_tour_order = []
            self.comp_tour_distance_um = 0.0
            return

        if start_uid and start_uid in self.comp_particles:
            self.comp_current_uid = start_uid

        if not self.comp_current_uid or self.comp_current_uid not in self.comp_particles:
            untested = [uid for uid, p in self.comp_particles.items() if not p.tested]
            self.comp_current_uid = untested[0] if untested else list(self.comp_particles.keys())[0]

        untested_uids = [uid for uid, p in self.comp_particles.items() if not p.tested]
        if self.comp_current_uid not in untested_uids and not self.comp_particles[self.comp_current_uid].tested:
            untested_uids.append(self.comp_current_uid)

        tour, total_dist = optimize_particle_route(self.comp_current_uid, untested_uids, self.comp_particles)
        self.comp_tour_order = tour
        self.comp_tour_distance_um = total_dist

        # Automatically assign the next target particle in the optimal tour
        if len(tour) > 1:
            self.comp_next_uid = tour[1]
        else:
            self.comp_next_uid = None

        self.window.combo_comp_next.blockSignals(True)
        if self.comp_next_uid:
            idx = self.window.combo_comp_next.findData(self.comp_next_uid)
            if idx >= 0:
                self.window.combo_comp_next.setCurrentIndex(idx)
        else:
            self.window.combo_comp_next.setCurrentIndex(-1)
        self.window.combo_comp_next.blockSignals(False)

    def populate_compression_particles(self):
        self.comp_particles = extract_all_particles(self.sample)
        if self.comp_session_data:
            for t in self.comp_session_data.tests:
                if t.particle_uid in self.comp_particles:
                    self.comp_particles[t.particle_uid].tested = True
                    self.comp_particles[t.particle_uid].test_timestamp = t.timestamp

        # Default selections if needed
        if not self.comp_current_uid or self.comp_current_uid not in self.comp_particles:
            untested = [uid for uid, p in self.comp_particles.items() if not p.tested]
            self.comp_current_uid = untested[0] if untested else (list(self.comp_particles.keys())[0] if self.comp_particles else None)

        # Optimize tour starting from current particle
        self.reoptimize_comp_tour(start_uid=self.comp_current_uid)

        # Populate target combo
        self.window.combo_comp_next.blockSignals(True)
        self.window.combo_comp_next.clear()
        for uid, p in self.comp_particles.items():
            label = f"{uid} (Site {p.site_number})"
            self.window.combo_comp_next.addItem(label, uid)

        if self.comp_next_uid:
            idx = self.window.combo_comp_next.findData(self.comp_next_uid)
            if idx >= 0:
                self.window.combo_comp_next.setCurrentIndex(idx)

        self.window.combo_comp_next.blockSignals(False)

        self.populate_compression_table()
        self.update_comp_plots()
        self.update_comp_smaract_guidance()

    def populate_compression_table(self):
        m = self.window.comp_particle_model
        m.removeRows(0, m.rowCount())

        tested_count = sum(1 for p in self.comp_particles.values() if p.tested)
        total_count = len(self.comp_particles)
        remaining_count = total_count - tested_count
        self.window.lbl_comp_queue_summary.setText(
            f"Tour: {remaining_count} remaining | Est. path: {self.comp_tour_distance_um:.1f} µm"
        )
        self.window.lbl_comp_tested_count.setText(f"Tested: {tested_count} / {total_count}")

        # Build ordered list of UIDs to display based on optimized tour
        display_uids: List[Tuple[str, str, str]] = []

        prev_pos = None
        for i, uid in enumerate(self.comp_tour_order):
            if uid not in self.comp_particles:
                continue
            p = self.comp_particles[uid]
            if p.tested:
                continue
            curr_pos = (p.flat_x_um, p.flat_y_um)
            if i == 0:
                step_str = "▶ Current"
                hop_str = "-"
            elif i == 1:
                step_str = "➔ Next"
                hop = math.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]) if prev_pos else 0.0
                hop_str = f"{hop:.1f}"
            else:
                step_str = f"#{i}"
                hop = math.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]) if prev_pos else 0.0
                hop_str = f"{hop:.1f}"
            prev_pos = curr_pos
            display_uids.append((uid, step_str, hop_str))

        # Any untested particles not in tour (fallback)
        in_tour = set(self.comp_tour_order)
        for uid, p in self.comp_particles.items():
            if not p.tested and uid not in in_tour:
                display_uids.append((uid, "Queued", "-"))

        # Tested particles
        tested_uids = [
            (uid, "✓ Done", "-") for uid, p in self.comp_particles.items() if p.tested
        ]

        if self.comp_filter_mode == "untested":
            items_to_show = display_uids
        elif self.comp_filter_mode == "tested":
            items_to_show = tested_uids
        else:
            items_to_show = display_uids + tested_uids

        for uid, step_str, hop_str in items_to_show:
            p = self.comp_particles[uid]
            status_str = "✓ Tested" if p.tested else "○ Queued"
            facet_str = f"{p.particle.top_facet_diameter * 1e9:.1f}" if p.particle.top_facet_diameter else "-"
            shape_str = p.particle.shape_type.capitalize()

            row_items = [
                QStandardItem(step_str),
                QStandardItem(uid),
                QStandardItem(hop_str),
                QStandardItem(shape_str),
                QStandardItem(facet_str),
                QStandardItem(status_str),
            ]

            # Clean semantic highlighting
            if uid == self.comp_current_uid and self.comp_mode == "test":
                for it in row_items:
                    it.setBackground(pg.mkColor(254, 243, 199)) # warm yellow/amber
            elif uid == self.comp_next_uid:
                for it in row_items:
                    it.setBackground(pg.mkColor(224, 242, 254)) # light blue
            elif p.tested:
                for it in row_items:
                    it.setBackground(pg.mkColor(240, 253, 244)) # light green

            for it in row_items:
                it.setData(uid, Qt.UserRole)
                it.setEditable(False)
            m.appendRow(row_items)

        self.window.lbl_comp_tested_count.setText(f"Tested: {tested_count} / {total_count}")

    def update_comp_plots(self):
        self.update_comp_region_plot()
        self.update_comp_site_plot()

    def update_comp_region_plot(self):
        if not self.sample:
            return
        region = next((r for r in self.sample.images if r.classification == ImageType.REGION), None)
        if not region:
            return

        meta_reg = region.metadata.get('MAIN', {})
        meta_sem = region.metadata.get('SEM', {})
        px_reg = float(meta_reg.get('PixelSizeX', 1.0))
        py_reg = float(meta_reg.get('PixelSizeY', 1.0))
        n_sig = int(meta_reg.get('ViewFieldsCountX', 1))
        strip = int(meta_reg.get('ImageStripSize', 0))

        # Always extract the SE detector channel (never BSE)
        tif_path = os.path.join(self.sample_dir, region.filename)
        if os.path.exists(tif_path):
            img = read_sem_tiff(tif_path)
            if strip > 0 and img.shape[0] > strip:
                img = img[:-strip, :]
            w = img.shape[1] // n_sig
            se_idx = 0
            for i in range(n_sig):
                det = meta_sem.get(f'Detector{i}', '')
                if 'SE' in det.upper() and 'BSE' not in det.upper():
                    se_idx = i
                    break
            se_img = img[:, se_idx*w:(se_idx+1)*w]
            H, W = se_img.shape[:2]
            self.window.comp_region_image_item.setImage(se_img, autoLevels=False)

            # Center coordinates in physical meters
            cx = W * px_reg / 2.0
            cy = H * py_reg / 2.0

            # Scale and tilt transform: scales pixels to meters, rotates and foreshortens around (cx, cy)
            tr = get_tilt_transform(cx, cy, self.comp_tilt_deg, self.comp_rot_deg, px_reg, py_reg)
            self.window.comp_region_image_item.setTransform(tr)
        else:
            cx = 2048 * px_reg / 2.0
            cy = 2048 * py_reg / 2.0

        for item in self.comp_region_marker_items:
            try: self.window.comp_region_plot.removeItem(item)
            except Exception: pass
        self.comp_region_marker_items.clear()

        curr_tx, curr_ty = None, None
        next_tx, next_ty = None, None

        for uid, p in self.comp_particles.items():
            tx, ty = map_point_tilt(p.flat_x_m, p.flat_y_m, cx, cy, self.comp_tilt_deg, self.comp_rot_deg)

            is_curr = (uid == self.comp_current_uid)
            is_next = (uid == self.comp_next_uid)

            if is_curr:
                curr_tx, curr_ty = tx, ty
                symbol = 'star'
                size = 18
                brush = pg.mkBrush(255, 140, 0, 240)
                pen = pg.mkPen('w', width=2)
                text_color = '#FF8C00'
            elif is_next:
                next_tx, next_ty = tx, ty
                symbol = 't1'
                size = 14
                brush = pg.mkBrush(0, 180, 255, 230)
                pen = pg.mkPen('w', width=2)
                text_color = '#00B4FF'
            elif p.tested:
                symbol = 'o'
                size = 8
                brush = pg.mkBrush(16, 124, 65, 200)
                pen = pg.mkPen('w', width=1)
                text_color = '#107C41'
            else:
                symbol = 'o'
                size = 7
                brush = pg.mkBrush(220, 220, 220, 180)
                pen = pg.mkPen('k', width=1)
                text_color = '#FFFFFF'

            scatter = pg.ScatterPlotItem(
                x=[tx], y=[ty],
                size=size, symbol=symbol,
                brush=brush, pen=pen
            )
            scatter.setZValue(20 if (is_curr or is_next) else 10)
            self.window.comp_region_plot.addItem(scatter)
            self.comp_region_marker_items.append(scatter)

            text_item = pg.TextItem(uid, color=text_color, anchor=(0.5, 1.2))
            text_item.setPos(tx, ty)
            font = text_item.textItem.font()
            font.setBold(is_curr or is_next)
            font.setPointSize(10 if (is_curr or is_next) else 8)
            text_item.setFont(font)
            text_item.setZValue(21 if (is_curr or is_next) else 11)
            self.window.comp_region_plot.addItem(text_item)
            self.comp_region_marker_items.append(text_item)

        if curr_tx is not None and next_tx is not None and (curr_tx != next_tx or curr_ty != next_ty):
            line = pg.PlotCurveItem(
                x=[curr_tx, next_tx],
                y=[curr_ty, next_ty],
                pen=pg.mkPen(color='#FF8C00', width=2, style=Qt.DashLine)
            )
            line.setZValue(15)
            self.window.comp_region_plot.addItem(line)
            self.comp_region_marker_items.append(line)

        # Tour path polyline connecting upcoming tour sequence
        if self.comp_tour_order and len(self.comp_tour_order) >= 2:
            pts_x, pts_y = [], []
            for uid in self.comp_tour_order[:8]:
                if uid in self.comp_particles:
                    p = self.comp_particles[uid]
                    tx, ty = map_point_tilt(p.flat_x_m, p.flat_y_m, cx, cy, self.comp_tilt_deg, self.comp_rot_deg)
                    pts_x.append(tx)
                    pts_y.append(ty)
            if len(pts_x) >= 2:
                tour_line = pg.PlotCurveItem(
                    x=pts_x, y=pts_y,
                    pen=pg.mkPen(color='#38BDF8', width=1.5, style=Qt.DashLine)
                )
                tour_line.setZValue(14)
                self.window.comp_region_plot.addItem(tour_line)
                self.comp_region_marker_items.append(tour_line)

    def update_comp_site_plot(self):
        for item in self.comp_site_marker_items:
            try: self.window.comp_site_plot.removeItem(item)
            except Exception: pass
        self.comp_site_marker_items.clear()

        # In Translation Mode, show the TARGET particle site. In Testing Mode, show CURRENT particle site.
        is_trans_mode = (self.comp_mode == "trans")
        active_uid = self.comp_next_uid if (is_trans_mode and self.comp_next_uid) else self.comp_current_uid

        if not active_uid or active_uid not in self.comp_particles:
            return

        p_info = self.comp_particles[active_uid]
        self.comp_site_filename = p_info.site_filename
        site_record = next((s for s in self.sample.images if s.filename == p_info.site_filename), None)
        if not site_record:
            return

        tif_path = os.path.join(self.sample_dir, site_record.filename)
        if not os.path.exists(tif_path):
            return

        img = read_sem_tiff(tif_path)
        meta_main = site_record.metadata.get('MAIN', {})
        meta_sem = site_record.metadata.get('SEM', {})
        n_sig = int(meta_main.get('ViewFieldsCountX', 1))
        strip = int(meta_main.get('ImageStripSize', 0))
        px = float(meta_main.get('PixelSizeX', 1.0))
        py = float(meta_main.get('PixelSizeY', 1.0))

        if strip > 0 and img.shape[0] > strip:
            img = img[:-strip, :]
        H, W = img.shape[:2]
        w = W // n_sig

        # SE Signal only (Detector0, NOT BSE)
        se_idx = 0
        for i in range(n_sig):
            det = meta_sem.get(f'Detector{i}', '')
            if 'SE' in det.upper() and 'BSE' not in det.upper():
                se_idx = i
                break
        site_signal = img[:, se_idx*w:(se_idx+1)*w]

        self.window.comp_site_image_item.setImage(site_signal, autoLevels=False)

        # Center tilt and rotation around active particle
        cx_site = p_info.particle.pixel_x * px
        cy_site = p_info.particle.pixel_y * py

        tr_site = get_tilt_transform(cx_site, cy_site, self.comp_tilt_deg, self.comp_rot_deg, px, py)
        self.window.comp_site_image_item.setTransform(tr_site)

        # Draw ALL particles belonging to this site
        for site_p in site_record.particles:
            sp_uid = f"{p_info.site_number}_{site_p.id}"
            is_active_target = (site_p.id == p_info.particle.id)
            sp_x_m = site_p.pixel_x * px
            sp_y_m = site_p.pixel_y * py
            tx, ty = map_point_tilt(sp_x_m, sp_y_m, cx_site, cy_site, self.comp_tilt_deg, self.comp_rot_deg)

            # Draw facet polygon if available
            polygon = site_p.facet_polygon
            if polygon and len(polygon) >= 3:
                poly_phys = [[pt[0] * px, pt[1] * py] for pt in polygon]
                poly_phys.append(poly_phys[0])
                mapped_poly = map_polygon_tilt(poly_phys, cx_site, cy_site, self.comp_tilt_deg, self.comp_rot_deg)
                xs = [pt[0] for pt in mapped_poly]
                ys = [pt[1] for pt in mapped_poly]

                if is_active_target:
                    poly_color = '#0078D7' if is_trans_mode else '#FFD700'
                    poly_pen = pg.mkPen(color=poly_color, width=2)
                else:
                    poly_pen = pg.mkPen(color='#00E5FF', width=1)

                poly_item = pg.PlotCurveItem(x=xs, y=ys, pen=poly_pen)
                poly_item.setZValue(15 if is_active_target else 12)
                self.window.comp_site_plot.addItem(poly_item)
                self.comp_site_marker_items.append(poly_item)

            if is_active_target:
                # PROMINENT TARGETING INDICATOR: Reticle with concentric circles and crosshair
                r_outer = 45 * px
                r_inner = 18 * px
                n_angles = 36
                angles = np.linspace(0, 2*math.pi, n_angles)

                reticle_primary = '#0078D7' if is_trans_mode else '#FF3366'
                reticle_accent = '#38BDF8' if is_trans_mode else '#FFD700'

                # Outer dashed circle
                ox = [tx + r_outer * math.cos(a) for a in angles]
                oy = [ty + r_outer * math.sin(a) for a in angles]
                c_outer = pg.PlotCurveItem(x=ox, y=oy, pen=pg.mkPen(color=reticle_primary, width=2, style=Qt.DashLine))
                c_outer.setZValue(25)
                self.window.comp_site_plot.addItem(c_outer)
                self.comp_site_marker_items.append(c_outer)

                # Inner circle
                ix = [tx + r_inner * math.cos(a) for a in angles]
                iy = [ty + r_inner * math.sin(a) for a in angles]
                c_inner = pg.PlotCurveItem(x=ix, y=iy, pen=pg.mkPen(color=reticle_accent, width=2))
                c_inner.setZValue(25)
                self.window.comp_site_plot.addItem(c_inner)
                self.comp_site_marker_items.append(c_inner)

                # Crosshair lines
                ch_arm = 65 * px
                nan = np.nan
                crosshair = pg.PlotCurveItem(
                    x=np.array([tx - ch_arm, tx - r_inner, nan, tx + r_inner, tx + ch_arm, nan, tx, tx, nan, tx, tx], dtype=float),
                    y=np.array([ty, ty, nan, ty, ty, nan, ty - ch_arm, ty - r_inner, nan, ty + r_inner, ty + ch_arm], dtype=float),
                    pen=pg.mkPen(color=reticle_primary, width=2),
                    connect='finite'
                )
                crosshair.setZValue(25)
                self.window.comp_site_plot.addItem(crosshair)
                self.comp_site_marker_items.append(crosshair)

                badge_text = f"★ TARGET: Particle {sp_uid}" if is_trans_mode else f"★ TESTING: Particle {sp_uid}"
                badge_color = '#38BDF8' if is_trans_mode else '#FFD700'
                badge = pg.TextItem(badge_text, color=badge_color, anchor=(0.5, 1.2))
                badge.setPos(tx, ty - ch_arm)
                f = badge.textItem.font()
                f.setBold(True)
                f.setPointSize(11)
                badge.setFont(f)
                badge.setZValue(30)
                self.window.comp_site_plot.addItem(badge)
                self.comp_site_marker_items.append(badge)
            else:
                # Neighboring particle on same site: subtle marker and ID
                lbl = pg.TextItem(f"#{sp_uid}", color='#00E5FF', anchor=(0.5, 1.2))
                lbl.setPos(tx, ty)
                f = lbl.textItem.font()
                f.setPointSize(9)
                lbl.setFont(f)
                lbl.setZValue(14)
                self.window.comp_site_plot.addItem(lbl)
                self.comp_site_marker_items.append(lbl)

                sc = pg.ScatterPlotItem(x=[tx], y=[ty], size=6, brush=pg.mkBrush(0, 229, 255, 180), pen=pg.mkPen('w', width=1))
                sc.setZValue(13)
                self.window.comp_site_plot.addItem(sc)
                self.comp_site_marker_items.append(sc)

        if is_trans_mode:
            p_curr = self.comp_particles.get(self.comp_current_uid)
            dx_str, dy_str = "+0.0", "+0.0"
            if p_curr:
                dx, dy, _ = predict_smaract_translation(
                    (p_curr.flat_x_um, p_curr.flat_y_um),
                    (p_info.flat_x_um, p_info.flat_y_um),
                    rot_deg=self.comp_rot_deg,
                    scale=self.comp_scale,
                    invert_x=self.comp_invert_x,
                    invert_y=self.comp_invert_y,
                    flip_xy=self.comp_flip_xy
                )
                dx_sign = "+" if dx >= 0 else ""
                dy_sign = "+" if dy >= 0 else ""
                dx_str = f"{dx_sign}{dx:.1f}"
                dy_str = f"{dy_sign}{dy:.1f}"

            self.window.lbl_comp_site_title.setText(
                f"Tilted Site View (SE Signal) - Target Particle {p_info.uid} on Site {p_info.site_number} (Move: ΔX={dx_str} µm, ΔY={dy_str} µm)"
            )
            self.window.lbl_comp_site_title.setStyleSheet("font-weight: bold; color: #0078D7; font-size: 12px;")
        else:
            self.window.lbl_comp_site_title.setText(
                f"Tilted Site View (SE Signal) - Current Particle {p_info.uid} on Site {p_info.site_number}"
            )
            self.window.lbl_comp_site_title.setStyleSheet("font-weight: bold; color: #107C41; font-size: 12px;")

    def update_comp_smaract_guidance(self):
        # 1. Update Mode A (Testing Mode) UI
        if self.comp_current_uid and self.comp_current_uid in self.comp_particles:
            p_curr = self.comp_particles[self.comp_current_uid]
            self.window.lbl_test_particle_uid.setText(f"Particle {p_curr.uid}")
            ecd_str = f"{p_curr.particle.ecd*1e9:.1f} nm" if p_curr.particle.ecd else "-"
            facet_str = f"{p_curr.particle.top_facet_diameter*1e9:.1f} nm" if p_curr.particle.top_facet_diameter else "-"
            shape_str = p_curr.particle.shape_type.capitalize()
            status_txt = " (✓ Already Tested)" if p_curr.tested else " (○ Untested)"
            self.window.lbl_test_particle_details.setText(
                f"Site {p_curr.site_number} | Shape: {shape_str} | ECD: {ecd_str} | Facet: {facet_str}{status_txt}"
            )
        else:
            self.window.lbl_test_particle_uid.setText("No Particle Selected")
            self.window.lbl_test_particle_details.setText("Select a particle from the queue or click on the overview")

        # 2. Update Mode B (Translation Mode) UI
        if not self.comp_current_uid or not self.comp_next_uid:
            self.window.lbl_trans_target_uid.setText("Target: None")
            self.window.lbl_comp_dx.setText("ΔX = +0.00 µm")
            self.window.lbl_comp_dy.setText("ΔY = +0.00 µm")
            self.window.lbl_comp_distance.setText("Dist: 0.0 µm")
            return

        p_curr = self.comp_particles.get(self.comp_current_uid)
        p_next = self.comp_particles.get(self.comp_next_uid)
        if not p_curr or not p_next:
            return

        self.window.lbl_trans_target_uid.setText(f"Target: Particle {p_next.uid} (Site {p_next.site_number})")

        dx, dy, dist = predict_smaract_translation(
            (p_curr.flat_x_um, p_curr.flat_y_um),
            (p_next.flat_x_um, p_next.flat_y_um),
            rot_deg=self.comp_rot_deg,
            scale=self.comp_scale,
            invert_x=self.comp_invert_x,
            invert_y=self.comp_invert_y,
            flip_xy=self.comp_flip_xy
        )

        dx_sign = "+" if dx >= 0 else ""
        dy_sign = "+" if dy >= 0 else ""
        self.window.lbl_comp_dx.setText(f"ΔX = {dx_sign}{dx:.2f} µm")
        self.window.lbl_comp_dy.setText(f"ΔY = {dy_sign}{dy:.2f} µm")
        self.window.lbl_comp_distance.setText(f"Dist: {dist:.2f} µm ({self.comp_scale:.3f}× scale)")

        self.window.spin_comp_act_dx.blockSignals(True)
        self.window.spin_comp_act_dy.blockSignals(True)
        self.window.spin_comp_act_dx.setValue(dx)
        self.window.spin_comp_act_dy.setValue(dy)
        self.window.spin_comp_act_dx.blockSignals(False)
        self.window.spin_comp_act_dy.blockSignals(False)

    def set_comp_mode(self, mode: str):
        self.comp_mode = mode
        if mode == "test":
            self.window.comp_mode_stack.setCurrentIndex(0)
        else:
            self.window.comp_mode_stack.setCurrentIndex(1)
        self.update_comp_smaract_guidance()
        self.update_comp_site_plot()
        self.populate_compression_table()

    def on_comp_tilt_changed(self, val):
        self.comp_tilt_deg = val
        self.update_comp_plots()

    def on_comp_rot_slider_changed(self, val):
        deg = val / 10.0
        self.comp_rot_deg = deg
        if len(self.comp_actual_history) == 0:
            self.comp_initial_rot_deg = deg
        self.window.spin_comp_rot.blockSignals(True)
        self.window.spin_comp_rot.setValue(deg)
        self.window.spin_comp_rot.blockSignals(False)
        self.update_comp_plots()
        self.update_comp_smaract_guidance()

    def on_comp_rot_spin_changed(self, val):
        self.comp_rot_deg = val
        if len(self.comp_actual_history) == 0:
            self.comp_initial_rot_deg = val
        self.window.slider_comp_rot.blockSignals(True)
        self.window.slider_comp_rot.setValue(int(round(val * 10)))
        self.window.slider_comp_rot.blockSignals(False)
        self.update_comp_plots()
        self.update_comp_smaract_guidance()

    def on_comp_scale_changed(self, val):
        self.comp_scale = float(val)
        self.update_comp_smaract_guidance()
        self.update_comp_site_plot()

    def update_comp_calibration_badge(self):
        if not self.comp_calibration or self.comp_calibration.total_count == 0:
            self.window.lbl_comp_calibration_badge.setText("⚡ Prior (Uncalibrated)")
            self.window.lbl_comp_calibration_badge.setStyleSheet(
                "QLabel { font-size: 11px; font-weight: bold; color: #475569; background-color: #f1f5f9; "
                "border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 6px; }"
            )
            self.window.lbl_comp_calibration_badge.setToolTip("Prior only (coarse visual alignment). Enter actual stage translations to calibrate.")
            return

        cal = self.comp_calibration
        in_c = cal.inlier_count
        tot_c = cal.total_count
        conf = cal.confidence_pct

        if in_c < 2:
            badge_text = f"⚡ Initial ({in_c}/{tot_c}): φ={cal.rot_deg:.1f}°±{cal.rot_std_deg:.1f}° | s={cal.scale:.3f} | {conf:.0f}% Conf"
            color_style = "color: #b45309; background-color: #fef3c7; border: 1px solid #fde68a;"
        elif cal.rot_std_deg <= 0.35 and cal.scale_std <= 0.015:
            badge_text = f"⚡ Converged ({in_c}/{tot_c}): φ={cal.rot_deg:.2f}°±{cal.rot_std_deg:.2f}° | s={cal.scale:.3f}±{cal.scale_std:.3f} | {conf:.0f}% Conf"
            color_style = "color: #15803d; background-color: #dcfce7; border: 1px solid #86efac;"
        else:
            badge_text = f"⚡ Calibrating ({in_c}/{tot_c}): φ={cal.rot_deg:.2f}°±{cal.rot_std_deg:.2f}° | s={cal.scale:.3f}±{cal.scale_std:.3f} | {conf:.0f}% Conf"
            color_style = "color: #0369a1; background-color: #e0f2fe; border: 1px solid #7dd3fc;"

        self.window.lbl_comp_calibration_badge.setText(badge_text)
        self.window.lbl_comp_calibration_badge.setStyleSheet(
            f"QLabel {{ font-size: 11px; font-weight: bold; {color_style} border-radius: 4px; padding: 2px 6px; }}"
        )
        tooltip = (
            f"<b>Calibration Metrics (RB-PSE)</b><br>"
            f"• Orientation (φ): {cal.rot_deg:.3f}° ± {cal.rot_std_deg:.3f}°<br>"
            f"• Scale Factor (s): {cal.scale:.4f} ± {cal.scale_std:.4f}<br>"
            f"• Confidence: {cal.confidence_pct:.1f}%<br>"
            f"• Consensus Inliers: {in_c} of {tot_c} moves<br>"
            f"• Mean Residual: {cal.mean_residual_um:.3f} µm<br>"
            f"• Anisotropic Scales: Sx = {cal.scale_x:.4f}, Sy = {cal.scale_y:.4f}<br>"
            f"<i>Outliers (blunders/drift) are automatically excluded.</i>"
        )
        self.window.lbl_comp_calibration_badge.setToolTip(tooltip)

    def on_comp_quick_rotate(self, delta_deg):
        new_deg = (self.comp_rot_deg + delta_deg) % 360.0
        self.window.spin_comp_rot.setValue(new_deg)

    def on_comp_polarity_changed(self):
        self.comp_invert_x = self.window.chk_comp_inv_x.isChecked()
        self.comp_invert_y = self.window.chk_comp_inv_y.isChecked()
        self.update_comp_smaract_guidance()

    def on_comp_flip_xy_toggled(self, checked):
        self.comp_flip_xy = checked
        if checked:
            self.window.btn_comp_flip_xy.setText("Axes Flipped (Y, X)")
            self.window.btn_comp_flip_xy.setStyleSheet("QPushButton { background-color: #D83B01; color: white; font-weight: bold; }")
        else:
            self.window.btn_comp_flip_xy.setText("Flip X / Y")
            self.window.btn_comp_flip_xy.setStyleSheet("")
        self.update_comp_smaract_guidance()

    def on_comp_next_combo_changed(self, index):
        uid = self.window.combo_comp_next.itemData(index)
        if uid and uid != self.comp_next_uid:
            self.comp_next_uid = uid
            self.populate_compression_table()
            self.update_comp_plots()
            self.update_comp_smaract_guidance()

    def on_comp_auto_next_clicked(self):
        if not self.comp_current_uid or not self.comp_particles:
            return
        curr_pos = (self.comp_particles[self.comp_current_uid].flat_x_um, self.comp_particles[self.comp_current_uid].flat_y_um)
        untested = [
            (math.hypot(p.flat_x_um - curr_pos[0], p.flat_y_um - curr_pos[1]), uid)
            for uid, p in self.comp_particles.items()
            if not p.tested and uid != self.comp_current_uid
        ]
        if untested:
            untested.sort()
            best_uid = untested[0][1]
            idx = self.window.combo_comp_next.findData(best_uid)
            if idx >= 0:
                self.window.combo_comp_next.setCurrentIndex(idx)

    def on_comp_reset_trans(self):
        if not self.comp_current_uid or not self.comp_next_uid:
            return
        p_curr = self.comp_particles.get(self.comp_current_uid)
        p_next = self.comp_particles.get(self.comp_next_uid)
        if not p_curr or not p_next:
            return
        dx, dy, dist = predict_smaract_translation(
            (p_curr.flat_x_um, p_curr.flat_y_um),
            (p_next.flat_x_um, p_next.flat_y_um),
            rot_deg=self.comp_rot_deg,
            scale=self.comp_scale,
            invert_x=self.comp_invert_x,
            invert_y=self.comp_invert_y,
            flip_xy=self.comp_flip_xy
        )
        self.window.spin_comp_act_dx.blockSignals(True)
        self.window.spin_comp_act_dy.blockSignals(True)
        self.window.spin_comp_act_dx.setValue(dx)
        self.window.spin_comp_act_dy.setValue(dy)
        self.window.spin_comp_act_dx.blockSignals(False)
        self.window.spin_comp_act_dy.blockSignals(False)

    def on_comp_confirm_trans(self):
        if not self.comp_next_uid or self.comp_next_uid not in self.comp_particles:
            self.set_comp_mode("test")
            return

        p_curr = self.comp_particles.get(self.comp_current_uid)
        p_next = self.comp_particles.get(self.comp_next_uid)
        if not p_curr or not p_next:
            self.set_comp_mode("test")
            return

        act_dx = self.window.spin_comp_act_dx.value()
        act_dy = self.window.spin_comp_act_dy.value()

        # Automatic Robust Bayesian alignment & scale refinement (RB-PSE)
        flat_vec = (p_next.flat_x_um - p_curr.flat_x_um, p_next.flat_y_um - p_curr.flat_y_um)
        actual_vec = (act_dx, act_dy)
        if math.hypot(act_dx, act_dy) > 1e-4:
            self.comp_actual_history.append((flat_vec, actual_vec))
            flats = [h[0] for h in self.comp_actual_history]
            acts = [h[1] for h in self.comp_actual_history]
            cal = robust_refine_orientation_scale(
                flats, acts,
                prior_rot_deg=self.comp_initial_rot_deg or self.comp_rot_deg,
                prior_scale=1.0,
                invert_x=self.comp_invert_x,
                invert_y=self.comp_invert_y,
                flip_xy=self.comp_flip_xy
            )
            self.comp_calibration = cal
            self.comp_rot_deg = cal.rot_deg
            self.comp_scale = cal.scale

            self.window.spin_comp_rot.blockSignals(True)
            self.window.spin_comp_rot.setValue(cal.rot_deg)
            self.window.spin_comp_rot.blockSignals(False)

            self.window.slider_comp_rot.blockSignals(True)
            self.window.slider_comp_rot.setValue(int(round(cal.rot_deg * 10)))
            self.window.slider_comp_rot.blockSignals(False)

            self.window.spin_comp_scale.blockSignals(True)
            self.window.spin_comp_scale.setValue(cal.scale)
            self.window.spin_comp_scale.blockSignals(False)

            self.update_comp_calibration_badge()
            self.window.lbl_comp_refine_status.setText(
                f"Auto-refined ({cal.inlier_count}/{cal.total_count} inliers): φ={cal.rot_deg:.2f}°±{cal.rot_std_deg:.2f}°, Scale={cal.scale:.3f}±{cal.scale_std:.3f}, Residual={cal.mean_residual_um:.2f} µm"
            )

        # Track accumulated stage travel from origin
        self.comp_cumulative_dx += act_dx
        self.comp_cumulative_dy += act_dy

        if self.comp_session_data and self.comp_session_data.tests:
            self.comp_session_data.last_updated_at = datetime.now().isoformat()
            self.comp_session_data.rotation_angle_deg = self.comp_rot_deg
            self.comp_session_data.scale_factor = self.comp_scale
            self.comp_session_data.stage_settings = {
                "invert_x": self.comp_invert_x,
                "invert_y": self.comp_invert_y,
                "flip_xy": self.comp_flip_xy
            }
            if self.comp_calibration:
                self.comp_session_data.calibration_summary = {
                    "rot_deg": round(self.comp_calibration.rot_deg, 3),
                    "rot_std_deg": round(self.comp_calibration.rot_std_deg, 3),
                    "scale": round(self.comp_calibration.scale, 4),
                    "scale_std": round(self.comp_calibration.scale_std, 4),
                    "confidence_pct": self.comp_calibration.confidence_pct,
                    "inlier_count": self.comp_calibration.inlier_count,
                    "total_moves": self.comp_calibration.total_count,
                    "mean_residual_um": round(self.comp_calibration.mean_residual_um, 3),
                    "scale_x": round(self.comp_calibration.scale_x, 4),
                    "scale_y": round(self.comp_calibration.scale_y, 4),
                    "status": self.comp_calibration.status_text
                }
            self.save_compression_session()

        # Advance target particle to be current
        self.comp_current_uid = self.comp_next_uid

        # Re-optimize remaining route starting from new current particle
        self.reoptimize_comp_tour(start_uid=self.comp_current_uid)

        # Transition back to Testing Mode
        self.set_comp_mode("test")
        self.populate_compression_table()
        self.update_comp_plots()
        self.update_comp_smaract_guidance()

    def on_comp_hotkey_enter(self):
        if self.window.tabs.currentIndex() == 2:
            if self.comp_mode == "test":
                self.on_comp_log_test()
            else:
                self.on_comp_confirm_trans()

    def on_comp_log_test(self):
        if not self.comp_current_uid or self.comp_current_uid not in self.comp_particles:
            return

        p_curr = self.comp_particles[self.comp_current_uid]
        timestamp = datetime.now().isoformat()
        p_curr.tested = True
        p_curr.test_timestamp = timestamp

        if not self.comp_session_data:
            self.load_or_create_compression_session()

        prev_uid = getattr(self, '_last_tested_uid', None)

        pred_dx, pred_dy, _ = predict_smaract_translation(
            (self.comp_particles[prev_uid].flat_x_um, self.comp_particles[prev_uid].flat_y_um) if (prev_uid and prev_uid in self.comp_particles) else (p_curr.flat_x_um, p_curr.flat_y_um),
            (p_curr.flat_x_um, p_curr.flat_y_um),
            rot_deg=self.comp_rot_deg,
            scale=self.comp_scale,
            invert_x=self.comp_invert_x,
            invert_y=self.comp_invert_y,
            flip_xy=self.comp_flip_xy
        )

        act_dx = self.window.spin_comp_act_dx.value()
        act_dy = self.window.spin_comp_act_dy.value()
        act_dist = math.hypot(act_dx, act_dy) if prev_uid else 0.0

        cal_snapshot = {
            "rot_deg": round(self.comp_calibration.rot_deg, 3),
            "rot_std_deg": round(self.comp_calibration.rot_std_deg, 3),
            "scale": round(self.comp_calibration.scale, 4),
            "scale_std": round(self.comp_calibration.scale_std, 4),
            "confidence_pct": self.comp_calibration.confidence_pct,
            "inlier_count": self.comp_calibration.inlier_count,
            "total_moves": self.comp_calibration.total_count,
            "mean_residual_um": round(self.comp_calibration.mean_residual_um, 3),
            "scale_x": round(self.comp_calibration.scale_x, 4),
            "scale_y": round(self.comp_calibration.scale_y, 4),
            "status": self.comp_calibration.status_text
        } if self.comp_calibration else {
            "rot_deg": round(self.comp_rot_deg, 3),
            "scale": round(self.comp_scale, 4),
            "status": "Prior (Visual Coarse Alignment)"
        }

        record = CompressionTestRecord(
            test_index=len(self.comp_session_data.tests) + 1,
            particle_uid=p_curr.uid,
            site_filename=p_curr.site_filename,
            site_number=p_curr.site_number,
            particle_id=p_curr.particle_id,
            timestamp=timestamp,
            flat_x_um=p_curr.flat_x_um,
            flat_y_um=p_curr.flat_y_um,
            particle_properties={
                "shape_type": p_curr.particle.shape_type,
                "ecd_m": p_curr.particle.ecd,
                "ecd_nm": round(p_curr.particle.ecd * 1e9, 2) if p_curr.particle.ecd else None,
                "top_facet_diameter_m": p_curr.particle.top_facet_diameter,
                "top_facet_diameter_nm": round(p_curr.particle.top_facet_diameter * 1e9, 2) if p_curr.particle.top_facet_diameter else None,
                "top_facet_area_m2": p_curr.particle.top_facet_area,
                "top_facet_area_um2": round(p_curr.particle.top_facet_area * 1e12, 4) if p_curr.particle.top_facet_area else None,
                "is_manually_fitted": p_curr.particle.is_manually_fitted,
                "pixel_x": p_curr.particle.pixel_x,
                "pixel_y": p_curr.particle.pixel_y,
                "facet_parameters": p_curr.particle.facet_parameters
            },
            from_particle_uid=prev_uid,
            smaract_predicted_dx_um=pred_dx if prev_uid else 0.0,
            smaract_predicted_dy_um=pred_dy if prev_uid else 0.0,
            smaract_actual_dx_um=act_dx if prev_uid else 0.0,
            smaract_actual_dy_um=act_dy if prev_uid else 0.0,
            smaract_actual_distance_um=act_dist,
            stage_tilt_deg=self.comp_tilt_deg,
            stage_rotation_deg=self.comp_rot_deg,
            stage_scale_factor=self.comp_scale,
            stage_axis_polarity={
                "invert_x": self.comp_invert_x,
                "invert_y": self.comp_invert_y,
                "flip_xy": self.comp_flip_xy
            },
            cumulative_stage_dx_um=self.comp_cumulative_dx,
            cumulative_stage_dy_um=self.comp_cumulative_dy,
            calibration_snapshot=cal_snapshot
        )
        self.comp_session_data.tests.append(record)
        self.comp_session_data.last_updated_at = timestamp
        self.comp_session_data.rotation_angle_deg = self.comp_rot_deg
        self.comp_session_data.scale_factor = self.comp_scale
        self.comp_session_data.total_particles_tested = len(self.comp_session_data.tests)
        self.comp_session_data.total_travel_distance_um = sum(
            t.smaract_actual_distance_um or 0.0 for t in self.comp_session_data.tests
        )
        self.comp_session_data.stage_settings = {
            "invert_x": self.comp_invert_x,
            "invert_y": self.comp_invert_y,
            "flip_xy": self.comp_flip_xy
        }
        if self.comp_calibration:
            self.comp_session_data.calibration_summary = cal_snapshot
        self.save_compression_session()

        self._last_tested_uid = p_curr.uid

        # Ensure next target is selected
        if not self.comp_next_uid or self.comp_next_uid == self.comp_current_uid or (self.comp_next_uid in self.comp_particles and self.comp_particles[self.comp_next_uid].tested):
            curr_pos = (p_curr.flat_x_um, p_curr.flat_y_um)
            untested = [
                (math.hypot(p.flat_x_um - curr_pos[0], p.flat_y_um - curr_pos[1]), uid)
                for uid, p in self.comp_particles.items()
                if not p.tested and uid != self.comp_current_uid
            ]
            if untested:
                untested.sort()
                self.comp_next_uid = untested[0][1]
            else:
                self.comp_next_uid = None

        if self.comp_next_uid:
            idx = self.window.combo_comp_next.findData(self.comp_next_uid)
            if idx >= 0:
                self.window.combo_comp_next.blockSignals(True)
                self.window.combo_comp_next.setCurrentIndex(idx)
                self.window.combo_comp_next.blockSignals(False)

        # Transition to Translation Mode
        self.set_comp_mode("trans")
        self.populate_compression_table()
        self.update_comp_plots()
        self.update_comp_smaract_guidance()

    def on_comp_table_double_clicked(self, index):
        item = self.window.comp_particle_model.item(index.row(), 0)
        if item:
            uid = item.data(Qt.UserRole)
            if uid:
                self.comp_current_uid = uid
                self.reoptimize_comp_tour(start_uid=uid)
                self.populate_compression_table()
                self.update_comp_plots()
                self.update_comp_smaract_guidance()

    def on_comp_set_current_clicked(self):
        sel = self.window.comp_particle_table.selectionModel().selectedRows()
        if sel:
            item = self.window.comp_particle_model.item(sel[0].row(), 0)
            if item:
                uid = item.data(Qt.UserRole)
                if uid:
                    self.comp_current_uid = uid
                    self.reoptimize_comp_tour(start_uid=uid)
                    self.populate_compression_table()
                    self.update_comp_plots()
                    self.update_comp_smaract_guidance()

    def set_comp_filter(self, mode):
        self.comp_filter_mode = mode
        self.window.btn_comp_filter_all.blockSignals(True)
        self.window.btn_comp_filter_untested.blockSignals(True)
        self.window.btn_comp_filter_tested.blockSignals(True)
        self.window.btn_comp_filter_all.setChecked(mode == "all")
        self.window.btn_comp_filter_untested.setChecked(mode == "untested")
        self.window.btn_comp_filter_tested.setChecked(mode == "tested")
        self.window.btn_comp_filter_all.blockSignals(False)
        self.window.btn_comp_filter_untested.blockSignals(False)
        self.window.btn_comp_filter_tested.blockSignals(False)
        self.populate_compression_table()

    def on_comp_region_plot_clicked(self, event):
        if self.window.tabs.currentIndex() != 2:
            return
        if event.button() != Qt.LeftButton:
            return
        pos = self.window.comp_region_plot.vb.mapSceneToView(event.scenePos())
        click_x, click_y = pos.x(), pos.y()

        region = next((r for r in self.sample.images if r.classification == ImageType.REGION), None)
        if not region: return
        meta_reg = region.metadata.get('MAIN', {})
        px_reg = float(meta_reg.get('PixelSizeX', 1.0))
        py_reg = float(meta_reg.get('PixelSizeY', 1.0))
        cx = 2048 * px_reg / 2.0
        cy = 2048 * py_reg / 2.0

        best_uid = None
        min_dist = float('inf')
        for uid, p in self.comp_particles.items():
            tx, ty = map_point_tilt(p.flat_x_m, p.flat_y_m, cx, cy, self.comp_tilt_deg, self.comp_rot_deg)
            d = math.hypot(tx - click_x, ty - click_y)
            if d < min_dist:
                min_dist = d
                best_uid = uid

        if best_uid:
            modifiers = QApplication.keyboardModifiers()
            if modifiers & Qt.ShiftModifier:
                self.comp_current_uid = best_uid
                self.reoptimize_comp_tour(start_uid=best_uid)
            else:
                self.comp_next_uid = best_uid
                if self.comp_next_uid:
                    idx_n = self.window.combo_comp_next.findData(self.comp_next_uid)
                    if idx_n >= 0:
                        self.window.combo_comp_next.blockSignals(True)
                        self.window.combo_comp_next.setCurrentIndex(idx_n)
                        self.window.combo_comp_next.blockSignals(False)

            self.populate_compression_table()
            self.update_comp_plots()
            self.update_comp_smaract_guidance()


    def save_session(self):
        if self.sample:
            json_path = os.path.join(self.sample_dir, "sample_data.json")
            try:
                with open(json_path, 'w') as f:
                    f.write(self.sample.model_dump_json(indent=2))
                self.is_dirty = False
                print(f"Session saved to {json_path}")
                QMessageBox.information(self.window, "Saved", f"Session successfully saved to:\n{json_path}")
            except Exception as e:
                print(f"Error saving session: {e}")
                QMessageBox.critical(self.window, "Error", f"Failed to save session:\n{e}")

    def on_close(self, event):
        if self.comp_session_data and self.comp_session_data.tests:
            try:
                self.save_compression_session()
            except Exception:
                pass

        if self.is_dirty:
            reply = QMessageBox.question(
                self.window, "Unsaved Changes",
                "You have unsaved changes. Do you want to save them before exiting?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            if reply == QMessageBox.Save:
                self.save_session()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    # --- Particle Measurement Workflow ---

    def on_site_plot_clicked(self, event):
        if not self.current_site:
            return
            
        pos = event.scenePos()
        if not self.window.site_plot.sceneBoundingRect().contains(pos):
            return
            
        # Ignore clicks directly on the active facet ROI or any of its scale/rotate handles
        if self.active_facet_roi is not None:
            if self.active_facet_roi.sceneBoundingRect().contains(pos):
                return
            for h in self.active_facet_roi.handles:
                h_item = h.get('item')
                if h_item and h_item.sceneBoundingRect().contains(pos):
                    return
            
        mouse_point = self.window.site_plot.vb.mapSceneToView(pos)
        x_m = mouse_point.x()
        y_m = mouse_point.y()
        
        meta = self.current_site.metadata.get('MAIN', {})
        px = float(meta.get('PixelSizeX', 1.0))
        py = float(meta.get('PixelSizeY', 1.0))
        
        px_x = x_m / px
        px_y = y_m / py
        
        if hasattr(self, 'current_site_signals') and self.current_site_signals:
            h, w = self.current_site_signals[0].shape[:2]
            if 0 <= px_x < w and 0 <= px_y < h:
                self.selected_particle_center_px = (float(px_x), float(px_y))
                self.draw_particle_center_marker(x_m, y_m)
                
                # Check if this click is on or near an existing measured particle:
                matched_particle = None
                for idx, p in enumerate(self.current_site.particles):
                    dist = np.sqrt((p.pixel_x - px_x)**2 + (p.pixel_y - px_y)**2)
                    if dist < 60.0:
                        matched_particle = (idx, p)
                        break
                
                if matched_particle:
                    idx, p = matched_particle
                    self.window.particle_table.selectRow(idx)
                    ecd_str = f"{p.ecd*1e9:.1f} nm" if p.ecd else "N/A"
                    se_str = f"{p.top_facet_diameter*1e9:.1f} nm" if p.top_facet_diameter else "N/A"
                    self.window.lbl_measure_hint.setText(f"Particle #{p.id} selected | ECD: {ecd_str} | Top: {se_str}")
                    self.update_active_facet_roi(p)
                else:
                    self.clear_active_facet_roi()
                    self.window.particle_table.clearSelection()
                    self.redraw_particle_overlays()
                    self.window.lbl_measure_hint.setText(f"Target particle: ({px_x:.0f}, {px_y:.0f}) px. Ready to measure.")

    def draw_particle_center_marker(self, x_m, y_m):
        if self.particle_center_marker is not None:
            try:
                self.window.site_plot.removeItem(self.particle_center_marker)
            except Exception:
                pass
        self.particle_center_marker = pg.ScatterPlotItem(
            [x_m], [y_m], size=14, pen=pg.mkPen('w', width=2), brush=pg.mkBrush(255, 0, 0, 180), symbol='+'
        )
        self.window.site_plot.addItem(self.particle_center_marker)

    def find_or_create_particle(self, cx: float, cy: float) -> Particle:
        if not self.current_site:
            return None
        for p in self.current_site.particles:
            dist = np.sqrt((p.pixel_x - cx)**2 + (p.pixel_y - cy)**2)
            if dist < 80.0:
                p.pixel_x = cx
                p.pixel_y = cy
                return p
        new_id = len(self.current_site.particles) + 1
        p = Particle(id=new_id, pixel_x=cx, pixel_y=cy)
        self.current_site.particles.append(p)
        return p

    def synthesize_default_facet(
        self,
        particle: Particle,
        shape_key: Optional[str] = None,
        r_outer: Optional[float] = None,
        is_placeholder: bool = True
    ):
        """
        Creates a candidate / default facet geometry (polygon or ellipse) centered
        on the particle. If is_placeholder is True, particle.top_facet_diameter and
        particle.top_facet_area remain None so the table shows '-' until the user
        interacts with the shape.
        """
        meta = self.current_site.metadata.get('MAIN', {}) if self.current_site else {}
        px = float(meta.get('PixelSizeX', 1.0))
        py = float(meta.get('PixelSizeY', 1.0))

        if shape_key is None:
            shape_idx = self.window.slider_facet_shape.value()
            shape_mode = SHAPE_MODES[shape_idx] if 0 <= shape_idx < len(SHAPE_MODES) else SHAPE_MODES[0]
            shape_key = getattr(particle, 'shape_type', None) or shape_mode["key"]

        # Determine facet radius in pixels
        if particle.top_facet_diameter is not None and particle.top_facet_diameter > 0:
            r_facet_px = 0.5 * (particle.top_facet_diameter / px)
        elif r_outer is not None and r_outer > 0:
            r_facet_px = 0.5 * r_outer
        elif particle.ellipse_bse and len(particle.ellipse_bse) >= 4:
            r_facet_px = 0.5 * (0.5 * (particle.ellipse_bse[2] + particle.ellipse_bse[3]))
        elif particle.ecd is not None and particle.ecd > 0:
            r_facet_px = 0.5 * ((particle.ecd / px) * 0.5)
        else:
            r_facet_px = 25.0

        cx = float(particle.pixel_x)
        cy = float(particle.pixel_y)

        particle.shape_type = shape_key

        if shape_key == "ellipse":
            particle.ellipse_se = [cx, cy, r_facet_px, r_facet_px, 0.0]
            particle.facet_polygon = None
            params = compute_facet_parameters_ellipse(
                particle.ellipse_se, px, py, ellipse_bse=particle.ellipse_bse, is_manual=True
            )
        else:
            n_sides_map = {
                "triangle": 3,
                "quadrilateral": 4,
                "pentagon": 5,
                "hexagon": 6,
                "polygon_6": 6,
                "polygon_8": 8
            }
            n_sides = n_sides_map.get(shape_key, 6)
            angles = [k * 2.0 * np.pi / n_sides for k in range(n_sides)]
            corners_px = [[float(cx + r_facet_px * np.cos(a)), float(cy + r_facet_px * np.sin(a))] for a in angles]
            particle.facet_polygon = corners_px
            particle.ellipse_se = None
            params = compute_facet_parameters(
                corners_px, px, py, shape_type=shape_key, ellipse_bse=particle.ellipse_bse, is_manual=True
            )

        particle.facet_parameters = params

        if is_placeholder:
            particle.top_facet_diameter = None
            particle.top_facet_area = None
            particle.is_manually_fitted = False
        else:
            particle.top_facet_diameter = params['diameter_m']
            particle.top_facet_area = params['area_m2']
            particle.is_manually_fitted = True

    def on_search_radius_slider_changed(self, value):
        radius_um = value / 10.0
        self.window.lbl_search_radius_val.setText(f"{radius_um:.1f} µm")

    def on_facet_shape_slider_changed(self, value):
        if 0 <= value < len(SHAPE_MODES):
            mode = SHAPE_MODES[value]
            self.window.lbl_facet_shape_val.setText(mode["name"])
            if self.active_particle_id is not None and self.current_site:
                p = self.get_particle_by_id(self.active_particle_id)
                if p and p.shape_type != mode["key"]:
                    is_placeholder = (p.top_facet_diameter is None)
                    self.synthesize_default_facet(p, shape_key=mode["key"], is_placeholder=is_placeholder)
                    self.clear_active_facet_roi()
                    self.update_active_facet_roi(p)
                    self.populate_particle_table()

    def get_signal_indices(self) -> Tuple[Optional[int], Optional[int]]:
        meta_sem = self.current_site.metadata.get('SEM', {}) if self.current_site else {}
        se_idx = None
        bse_idx = None
        n = len(self.current_site_signals) if hasattr(self, 'current_site_signals') else 0
        
        # Step 1: Detect explicit detectors
        for i in range(n):
            det_name = meta_sem.get(f'Detector{i}', '').upper()
            if 'BSE' in det_name:
                bse_idx = i
            elif 'IN-BEAM SE' in det_name or 'INBEAM SE' in det_name:
                se_idx = i
            elif 'SE' in det_name and se_idx is None:
                se_idx = i

        # Step 2: Handle cases with 2 or more signals
        if n >= 2:
            if se_idx is not None and bse_idx is None:
                # If only SE / In-Beam SE was specified, use the other signal for BSE/boundary
                bse_idx = 1 if se_idx == 0 else 0
            elif bse_idx is not None and se_idx is None:
                se_idx = 1 if bse_idx == 0 else 0
            elif se_idx is None and bse_idx is None:
                se_idx = 0
                bse_idx = 1
            elif se_idx == bse_idx:
                # Disambiguate if both mapped to the same index
                if se_idx == 1:
                    bse_idx = 0
                else:
                    bse_idx = 1
        else:
            # Single-image fallback
            if se_idx is None and n > 0:
                se_idx = 0
            if bse_idx is None and n > 0:
                bse_idx = 0

        return se_idx, bse_idx

    def on_measure_particle_combined(self):
        """
        1-Click Combined Particle Measurement:
        1. Checks for double image (SE + BSE).
        2. Fits outer particle ellipse on BSE via RANSAC constrained by the search radius slider.
        3. Adjusts/snaps center coordinates to the true geometric center from outer RANSAC.
        4. Fits inner 6-gon top facet on SE via Difference-of-Gaussians (DoG) RANSAC constrained by outer radius.
        """
        if self.window.tabs.currentIndex() != 1:
            return

        if not self.current_site or not hasattr(self, 'current_site_signals') or not self.current_site_signals:
            QMessageBox.information(self.window, "Info", "Please select a site first.")
            return

        se_idx, bse_idx = self.get_signal_indices()
        if bse_idx is None or bse_idx >= len(self.current_site_signals):
            QMessageBox.warning(
                self.window,
                "Double Image Required",
                "Particle and facet recognition requires a double image (SE + BSE signals).\n\n"
                "No BSE image was detected for this site."
            )
            return

        if se_idx is None or se_idx >= len(self.current_site_signals):
            QMessageBox.warning(
                self.window,
                "Double Image Required",
                "Particle and facet recognition requires a double image (SE + BSE signals).\n\n"
                "No SE image was detected for this site."
            )
            return

        meta = self.current_site.metadata.get('MAIN', {})
        px = float(meta.get('PixelSizeX', 1.0))
        py = float(meta.get('PixelSizeY', 1.0))

        bse_img = self.current_site_signals[bse_idx]
        se_img = self.current_site_signals[se_idx]

        if self.selected_particle_center_px is None:
            smoothed = ndimage.gaussian_filter(bse_img.astype(float), 10.0)
            cy, cx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
            self.selected_particle_center_px = (float(cx), float(cy))
            self.draw_particle_center_marker(float(cx) * px, float(cy) * py)

        cx_init, cy_init = self.selected_particle_center_px

        # Read search radius from slider (default 2.5 µm)
        radius_um = self.window.slider_search_radius.value() / 10.0
        search_radius_px = int((radius_um * 1e-6) / px)

        # Step 1: Fit outer particle boundary on BSE
        res_bse = fit_particle_ransac(bse_img, cx_init, cy_init, px, search_radius=search_radius_px, sigma=2.0, residual_threshold=4.0)
        if not res_bse:
            QMessageBox.warning(
                self.window,
                "Measurement Failed",
                f"Could not fit outer particle boundary on BSE signal within {radius_um:.1f} µm search radius.\n\n"
                "Try clicking closer to the particle or increasing the search radius slider."
            )
            return

        # Step 2: Adjust center point based on outer RANSAC fit
        cx_fit = float(res_bse['cx'])
        cy_fit = float(res_bse['cy'])
        r_outer = float(np.sqrt(res_bse['a'] * res_bse['b']))

        self.selected_particle_center_px = (cx_fit, cy_fit)
        self.draw_particle_center_marker(cx_fit * px, cy_fit * py)

        # Step 3: Fit inner top facet on SE constrained by outer radius
        shape_idx = self.window.slider_facet_shape.value()
        shape_mode = SHAPE_MODES[shape_idx] if 0 <= shape_idx < len(SHAPE_MODES) else SHAPE_MODES[0]
        shape_key = shape_mode["key"]

        if shape_key == "ellipse":
            res_facet = fit_facet_ellipse_ransac(se_img, cx_fit, cy_fit, px, r_outer=r_outer)
            facet_is_ellipse = True
        else:
            n_sides = shape_mode["n_sides"]
            res_facet = fit_facet_polygon_gradient_ransac(se_img, cx_fit, cy_fit, px, r_outer=r_outer, n_sides=n_sides)
            if not res_facet:
                res_facet = fit_facet_polygon_dog_ransac(se_img, cx_fit, cy_fit, px, r_outer=r_outer, n_sides=n_sides)
            facet_is_ellipse = False

        # Step 4: Record particle
        particle = self.find_or_create_particle(cx_fit, cy_fit)
        particle.shape_type = shape_key
        particle.ecd = res_bse['diameter_m']
        particle.ellipse_bse = res_bse['ellipse_params']
        if res_facet:
            particle.top_facet_diameter = res_facet['diameter_m']
            particle.top_facet_area = res_facet['area_m2']
            particle.is_manually_fitted = False
            if facet_is_ellipse:
                particle.ellipse_se = res_facet['ellipse_params']
                particle.facet_polygon = None
                particle.facet_parameters = compute_facet_parameters_ellipse(
                    res_facet['ellipse_params'], px, py, ellipse_bse=particle.ellipse_bse, is_manual=False
                )
                drag_hint = "(Drag axes to scale, rotate handle to rotate)"
            else:
                particle.facet_polygon = res_facet['corners']
                particle.ellipse_se = None
                particle.facet_parameters = compute_facet_parameters(
                    res_facet['corners'], px, py, shape_type=shape_key, ellipse_bse=particle.ellipse_bse, is_manual=False
                )
                if shape_key == "hexagon":
                    drag_hint = "(Drag corner: scale/rotate, Shift+Drag: extend face)"
                else:
                    drag_hint = "(Drag any corner freely; other corners stay fixed)"
            msg = f"Particle #{particle.id}: ECD = {res_bse['diameter_m']*1e9:.1f} nm | Facet ({shape_mode['name']}) = {res_facet['diameter_m']*1e9:.1f} nm {drag_hint}"
        else:
            self.synthesize_default_facet(particle, shape_key=shape_key, r_outer=r_outer, is_placeholder=True)
            if shape_key == "ellipse":
                drag_hint = "(Drag axes to scale, rotate handle to rotate)"
            elif shape_key == "hexagon":
                drag_hint = "(Drag corner: scale/rotate, Shift+Drag: extend face)"
            else:
                drag_hint = "(Drag any corner freely; other corners stay fixed)"
            msg = f"Particle #{particle.id}: ECD = {res_bse['diameter_m']*1e9:.1f} nm | Facet inconclusive for {shape_mode['name']} -> Red placeholder placed {drag_hint}"

        self.is_dirty = True
        self.populate_particle_table()
        try:
            p_idx = self.current_site.particles.index(particle)
            self.window.particle_table.selectRow(p_idx)
        except ValueError:
            pass
        self.update_active_facet_roi(particle)
        self.window.lbl_measure_hint.setText(msg)

    def on_measure_ecd(self):
        if not self.current_site or not hasattr(self, 'current_site_signals') or not self.current_site_signals:
            QMessageBox.information(self.window, "Info", "Please select a site first.")
            return

        se_idx, bse_idx = self.get_signal_indices()
        if bse_idx is None or bse_idx >= len(self.current_site_signals):
            QMessageBox.warning(
                self.window,
                "Double Image Required",
                "Particle and facet recognition requires a double image (SE + BSE signals).\n\n"
                "No BSE image was detected for this site."
            )
            return

        meta = self.current_site.metadata.get('MAIN', {})
        px = float(meta.get('PixelSizeX', 1.0))
        py = float(meta.get('PixelSizeY', 1.0))

        if self.window.site_signal_tabs.currentIndex() != bse_idx:
            self.window.site_signal_tabs.setCurrentIndex(bse_idx)

        bse_img = self.current_site_signals[bse_idx]

        if self.selected_particle_center_px is None:
            smoothed = ndimage.gaussian_filter(bse_img.astype(float), 10.0)
            cy, cx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
            self.selected_particle_center_px = (float(cx), float(cy))
            self.draw_particle_center_marker(float(cx) * px, float(cy) * py)

        cx, cy = self.selected_particle_center_px

        radius_um = self.window.slider_search_radius.value() / 10.0
        search_radius_px = int((radius_um * 1e-6) / px)

        res = fit_particle_ransac(bse_img, cx, cy, px, search_radius=search_radius_px, sigma=2.0, residual_threshold=4.0)
        if not res:
            QMessageBox.warning(
                self.window,
                "Measurement Failed",
                f"Could not fit outer particle boundary on BSE signal within {radius_um:.1f} µm search radius."
            )
            return

        # Adjust center
        cx_fit, cy_fit = float(res['cx']), float(res['cy'])
        self.selected_particle_center_px = (cx_fit, cy_fit)
        self.draw_particle_center_marker(cx_fit * px, cy_fit * py)

        particle = self.find_or_create_particle(cx_fit, cy_fit)
        particle.ecd = res['diameter_m']
        particle.ellipse_bse = res['ellipse_params']

        self.is_dirty = True
        self.populate_particle_table()
        self.redraw_particle_overlays()
        self.window.lbl_measure_hint.setText(f"Particle #{particle.id}: ECD = {res['diameter_m']*1e9:.1f} nm")

    def on_measure_se(self):
        if not self.current_site or not hasattr(self, 'current_site_signals') or not self.current_site_signals:
            QMessageBox.information(self.window, "Info", "Please select a site first.")
            return

        se_idx, bse_idx = self.get_signal_indices()
        if bse_idx is None or bse_idx >= len(self.current_site_signals):
            QMessageBox.warning(
                self.window,
                "Double Image Required",
                "Particle and facet recognition requires a double image (SE + BSE signals).\n\n"
                "No BSE image was detected for this site."
            )
            return

        if se_idx is None or se_idx >= len(self.current_site_signals):
            QMessageBox.warning(self.window, "Error", "No SE image detected for this site.")
            return

        meta = self.current_site.metadata.get('MAIN', {})
        px = float(meta.get('PixelSizeX', 1.0))
        py = float(meta.get('PixelSizeY', 1.0))

        if self.window.site_signal_tabs.currentIndex() != se_idx:
            self.window.site_signal_tabs.setCurrentIndex(se_idx)

        se_img = self.current_site_signals[se_idx]
        bse_img = self.current_site_signals[bse_idx]

        if self.selected_particle_center_px is None:
            smoothed = ndimage.gaussian_filter(se_img.astype(float), 10.0)
            cy, cx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
            self.selected_particle_center_px = (float(cx), float(cy))
            self.draw_particle_center_marker(float(cx) * px, float(cy) * py)

        cx, cy = self.selected_particle_center_px

        radius_um = self.window.slider_search_radius.value() / 10.0
        search_radius_px = int((radius_um * 1e-6) / px)

        # Check existing outer ellipse or fit from BSE
        particle = self.find_or_create_particle(cx, cy)
        r_outer = None
        if particle and particle.ellipse_bse:
            r_outer = float(np.sqrt(particle.ellipse_bse[2] * particle.ellipse_bse[3]))
            cx_fit, cy_fit = particle.pixel_x, particle.pixel_y
        else:
            bse_res = fit_particle_ransac(bse_img, cx, cy, px, search_radius=search_radius_px, sigma=2.0, residual_threshold=4.0)
            if bse_res:
                cx_fit, cy_fit = float(bse_res['cx']), float(bse_res['cy'])
                r_outer = float(np.sqrt(bse_res['a'] * bse_res['b']))
                particle.pixel_x = cx_fit
                particle.pixel_y = cy_fit
                particle.ecd = bse_res['diameter_m']
                particle.ellipse_bse = bse_res['ellipse_params']
                self.selected_particle_center_px = (cx_fit, cy_fit)
                self.draw_particle_center_marker(cx_fit * px, cy_fit * py)
            else:
                cx_fit, cy_fit = cx, cy

        shape_idx = self.window.slider_facet_shape.value()
        shape_mode = SHAPE_MODES[shape_idx] if 0 <= shape_idx < len(SHAPE_MODES) else SHAPE_MODES[0]
        shape_key = shape_mode["key"]

        if shape_key == "ellipse":
            res = fit_facet_ellipse_ransac(se_img, cx_fit, cy_fit, px, r_outer=r_outer)
            facet_is_ellipse = True
        else:
            n_sides = shape_mode["n_sides"]
            res = fit_facet_polygon_gradient_ransac(se_img, cx_fit, cy_fit, px, r_outer=r_outer, n_sides=n_sides)
            if not res:
                res = fit_facet_polygon_dog_ransac(se_img, cx_fit, cy_fit, px, r_outer=r_outer, n_sides=n_sides)
            facet_is_ellipse = False

        if not res:
            self.synthesize_default_facet(particle, shape_key=shape_key, r_outer=r_outer, is_placeholder=True)
            if shape_key == "ellipse":
                drag_hint = "(Drag axes to scale, rotate handle to rotate)"
            elif shape_key == "hexagon":
                drag_hint = "(Drag corner: scale/rotate, Shift+Drag: extend face)"
            else:
                drag_hint = "(Drag any corner freely; other corners stay fixed)"
            msg = f"Particle #{particle.id}: Auto-fit inconclusive for {shape_mode['name']}. Red placeholder placed — drag handles to set size {drag_hint}"
        else:
            particle.shape_type = shape_key
            particle.top_facet_diameter = res['diameter_m']
            particle.top_facet_area = res['area_m2']
            particle.is_manually_fitted = False
            if facet_is_ellipse:
                particle.ellipse_se = res['ellipse_params']
                particle.facet_polygon = None
                particle.facet_parameters = compute_facet_parameters_ellipse(
                    res['ellipse_params'], px, py, ellipse_bse=particle.ellipse_bse, is_manual=False
                )
                drag_hint = "(Drag axes to scale, rotate handle to rotate)"
            else:
                particle.facet_polygon = res['corners']
                particle.ellipse_se = None
                particle.facet_parameters = compute_facet_parameters(
                    res['corners'], px, py, shape_type=shape_key, ellipse_bse=particle.ellipse_bse, is_manual=False
                )
                if shape_key == "hexagon":
                    drag_hint = "(Drag corner: scale/rotate, Shift+Drag: extend face)"
                else:
                    drag_hint = "(Drag any corner freely; other corners stay fixed)"
            msg = f"Particle #{particle.id}: Facet ({shape_mode['name']}) = {res['diameter_m']*1e9:.1f} nm {drag_hint}"

        self.is_dirty = True
        self.populate_particle_table()
        try:
            p_idx = self.current_site.particles.index(particle)
            self.window.particle_table.selectRow(p_idx)
        except ValueError:
            pass
        self.update_active_facet_roi(particle)
        self.window.lbl_measure_hint.setText(f"Particle #{particle.id}: Facet ({shape_mode['name']}) = {res['diameter_m']*1e9:.1f} nm {drag_hint}")

    def populate_particle_table(self):
        self.window.particle_model.removeRows(0, self.window.particle_model.rowCount())
        if not self.current_site:
            return
            
        meta = self.current_site.metadata.get('MAIN', {})
        px = float(meta.get('PixelSizeX', 1.0))
        py = float(meta.get('PixelSizeY', 1.0))
        
        for p in self.current_site.particles:
            row = [
                QStandardItem(str(p.id)),
                QStandardItem(f"{p.pixel_x * px * 1e6:.2f}"),
                QStandardItem(f"{p.pixel_y * py * 1e6:.2f}"),
                QStandardItem(f"{p.ecd * 1e9:.1f}" if p.ecd is not None else "-"),
                QStandardItem(f"{p.top_facet_diameter * 1e9:.1f}" if p.top_facet_diameter is not None else "-")
            ]
            for it in row:
                it.setEditable(False)
            self.window.particle_model.appendRow(row)

    def redraw_particle_overlays(self):
        for item in self.particle_plot_items:
            try:
                self.window.site_plot.removeItem(item)
            except Exception:
                pass
        self.particle_plot_items.clear()
        
        if not self.current_site:
            return
            
        meta = self.current_site.metadata.get('MAIN', {})
        px = float(meta.get('PixelSizeX', 1.0))
        py = float(meta.get('PixelSizeY', 1.0))
        
        for p in self.current_site.particles:
            # BSE Ellipse (Cyan)
            if p.ellipse_bse:
                cx, cy, a, b, theta = p.ellipse_bse
                xp, yp = get_ellipse_points(cx, cy, a, b, theta, px, py)
                curve = pg.PlotCurveItem(xp, yp, pen=pg.mkPen(color='#00E5FF', width=2, style=Qt.SolidLine))
                self.window.site_plot.addItem(curve)
                self.particle_plot_items.append(curve)
                
            # SE Facet Polygon (Gold) or Ellipse
            if self.active_particle_id == p.id and self.active_facet_roi is not None:
                # Active particle is currently represented by the interactive ROI
                pass
            elif p.facet_polygon:
                xp, yp = get_polygon_points(p.facet_polygon, px, py)
                pen_color = '#FF1744' if p.top_facet_diameter is None else '#FFD600'
                pen_style = Qt.DashLine if p.top_facet_diameter is None else Qt.SolidLine
                curve = pg.PlotCurveItem(xp, yp, pen=pg.mkPen(color=pen_color, width=2, style=pen_style))
                self.window.site_plot.addItem(curve)
                self.particle_plot_items.append(curve)
            elif p.ellipse_se:
                cx, cy, a, b, theta = p.ellipse_se
                xp, yp = get_ellipse_points(cx, cy, a, b, theta, px, py)
                pen_color = '#FF1744' if p.top_facet_diameter is None else '#FFD600'
                curve = pg.PlotCurveItem(xp, yp, pen=pg.mkPen(color=pen_color, width=2, style=Qt.DashLine))
                self.window.site_plot.addItem(curve)
                self.particle_plot_items.append(curve)
                
            # Label at center
            lbl = pg.TextItem(text=f"#{p.id}", color='#FFFFFF', anchor=(0.5, 0.5))
            lbl.setPos(p.pixel_x * px, p.pixel_y * py)
            self.window.site_plot.addItem(lbl)
            self.particle_plot_items.append(lbl)

    def on_delete_particle(self):
        if not self.current_site:
            return
        self.clear_active_facet_roi()
        selected = self.window.particle_table.selectionModel().selectedRows()
        if not selected:
            return
        rows_to_delete = sorted([s.row() for s in selected], reverse=True)
        for r in rows_to_delete:
            if 0 <= r < len(self.current_site.particles):
                self.current_site.particles.pop(r)
        for i, p in enumerate(self.current_site.particles):
            p.id = i + 1
        self.selected_particle_center_px = None
        if self.particle_center_marker is not None:
            try:
                self.window.site_plot.removeItem(self.particle_center_marker)
            except Exception:
                pass
            self.particle_center_marker = None
        self.is_dirty = True
        self.populate_particle_table()
        self.redraw_particle_overlays()

    def on_particle_table_selected(self, selected, deselected):
        indexes = selected.indexes()
        if not indexes or not self.current_site:
            self.clear_active_facet_roi()
            self.redraw_particle_overlays()
            return
        row = indexes[0].row()
        if 0 <= row < len(self.current_site.particles):
            p = self.current_site.particles[row]
            meta = self.current_site.metadata.get('MAIN', {})
            px = float(meta.get('PixelSizeX', 1.0))
            py = float(meta.get('PixelSizeY', 1.0))
            self.selected_particle_center_px = (p.pixel_x, p.pixel_y)
            self.draw_particle_center_marker(p.pixel_x * px, p.pixel_y * py)
            ecd_str = f"{p.ecd*1e9:.1f} nm" if p.ecd else "N/A"
            if p.top_facet_diameter is not None:
                se_str = f"{p.top_facet_diameter*1e9:.1f} nm"
                self.window.lbl_measure_hint.setText(f"Particle #{p.id} selected | ECD: {ecd_str} | Top: {se_str}")
            else:
                self.window.lbl_measure_hint.setText(f"Particle #{p.id} selected | ECD: {ecd_str} | Top: - (Red placeholder: drag to set manual size)")
            self.update_active_facet_roi(p)

    def get_particle_by_id(self, p_id: int) -> Optional[Particle]:
        if not self.current_site:
            return None
        for p in self.current_site.particles:
            if p.id == p_id:
                return p
        return None

    def clear_active_facet_roi(self):
        if self.active_facet_roi is not None:
            try:
                self.window.site_plot.removeItem(self.active_facet_roi)
            except Exception:
                pass
            self.active_facet_roi = None
        self.active_particle_id = None

    def update_active_facet_roi(self, particle: Optional[Particle]):
        if not self.current_site or particle is None:
            self.clear_active_facet_roi()
            self.redraw_particle_overlays()
            return

        if not particle.facet_polygon and not particle.ellipse_se:
            self.synthesize_default_facet(particle, is_placeholder=True)
            self.populate_particle_table()
            
        if self.active_particle_id == particle.id and self.active_facet_roi is not None:
            return
            
        self.clear_active_facet_roi()

        # Synchronize facet shape slider to match particle's shape
        p_shape = getattr(particle, 'shape_type', 'hexagon') or 'hexagon'
        for idx, mode in enumerate(SHAPE_MODES):
            if mode["key"] == p_shape:
                self.window.slider_facet_shape.blockSignals(True)
                self.window.slider_facet_shape.setValue(idx)
                self.window.lbl_facet_shape_val.setText(mode["name"])
                self.window.slider_facet_shape.blockSignals(False)
                break
        
        meta = self.current_site.metadata.get('MAIN', {})
        px = float(meta.get('PixelSizeX', 1.0))
        py = float(meta.get('PixelSizeY', 1.0))
        
        roi = create_roi_for_particle(particle, px, py)
        if roi is None:
            return
            
        self.active_facet_roi = roi
        self.active_particle_id = particle.id
        
        roi.sigRegionChanged.connect(self.on_facet_roi_changed)
        roi.sigRegionChangeFinished.connect(self.on_facet_roi_change_finished)
        
        self.window.site_plot.addItem(roi)
        self.redraw_particle_overlays()

    def on_facet_roi_changed(self, roi):
        if not self.current_site or self.active_particle_id is None:
            return
        p = self.get_particle_by_id(self.active_particle_id)
        if not p:
            return

        # Turn yellow as soon as user starts interacting
        if getattr(roi, 'is_placeholder', False):
            roi.set_placeholder(False)

        d_m, a_m2 = roi.get_equivalent_diameter_and_area()
        p_shape = getattr(p, "shape_type", "hexagon")
        if p_shape == "hexagon":
            edge_idx = getattr(roi, 'edge_drag_idx', None)
            if edge_idx is not None:
                hint = f"Shift-dragging Edge #{edge_idx}: sliding endpoints on adjacent edges (120° locked)"
            else:
                hint = "Drag corner: scale/rotate | Shift+Drag corner: extend | Shift+Drag edge: lengthen/shorten | Drag body: move"
        elif p_shape == "ellipse":
            hint = "Drag axis handle to scale major/minor | Drag rotation handle to rotate | Drag body to move"
        else:
            hint = "Drag any corner freely (other corners stay fixed) | Drag body to move"
        self.window.lbl_measure_hint.setText(
            f"Manual Facet Particle #{p.id} ({p_shape}): Facet = {d_m*1e9:.1f} nm | Area = {a_m2*1e12:.4f} µm² ({hint})"
        )

    def on_facet_roi_change_finished(self, roi):
        if not self.current_site or self.active_particle_id is None:
            return
        p = self.get_particle_by_id(self.active_particle_id)
        if not p:
            return

        if getattr(roi, 'is_placeholder', False):
            roi.set_placeholder(False)
            
        meta = self.current_site.metadata.get('MAIN', {})
        px = float(meta.get('PixelSizeX', 1.0))
        py = float(meta.get('PixelSizeY', 1.0))
        
        if isinstance(roi, EllipseFacetROI):
            cx_m, cy_m, a_m, b_m, theta = roi.get_ellipse_params_m()
            d_m, a_m2 = roi.get_equivalent_diameter_and_area()
            cx_px = cx_m / px
            cy_px = cy_m / py
            a_px = a_m / px
            b_px = b_m / py
            p.shape_type = "ellipse"
            p.ellipse_se = [cx_px, cy_px, a_px, b_px, theta]
            p.facet_polygon = None
            p.top_facet_diameter = d_m
            p.top_facet_area = a_m2
            p.pixel_x = cx_px
            p.pixel_y = cy_px
            p.is_manually_fitted = True
            p.facet_parameters = compute_facet_parameters_ellipse(
                p.ellipse_se, px, py, ellipse_bse=p.ellipse_bse, is_manual=True
            )
            self.selected_particle_center_px = (cx_px, cy_px)
            self.draw_particle_center_marker(cx_m, cy_m)
        else:
            corners_m = roi.get_corners() if hasattr(roi, 'get_corners') else roi.get_hexagon_corners()
            cx_m, cy_m = roi.get_center() if hasattr(roi, 'get_center') else roi.get_hexagon_center()
            d_m, a_m2 = roi.get_equivalent_diameter_and_area()
            
            corners_px = [[x / px, y / py] for x, y in corners_m]
            cx_px = cx_m / px
            cy_px = cy_m / py
            
            p.facet_polygon = corners_px
            p.ellipse_se = None
            p.top_facet_diameter = d_m
            p.top_facet_area = a_m2
            p.pixel_x = cx_px
            p.pixel_y = cy_px
            p.is_manually_fitted = True
            p.facet_parameters = compute_facet_parameters(
                corners_px, px, py, shape_type=p.shape_type, ellipse_bse=p.ellipse_bse, is_manual=True
            )
            
            self.selected_particle_center_px = (cx_px, cy_px)
            self.draw_particle_center_marker(cx_m, cy_m)
        
        self.is_dirty = True
        self.populate_particle_table()
        
        try:
            row_idx = self.current_site.particles.index(p)
            self.window.particle_table.selectRow(row_idx)
        except ValueError:
            pass
            
        ecd_str = f"{p.ecd*1e9:.1f} nm" if p.ecd else "N/A"
        self.window.lbl_measure_hint.setText(
            f"Particle #{p.id} ({p.shape_type}) ground truth saved: Top Facet = {d_m*1e9:.1f} nm | Area = {a_m2*1e12:.4f} µm² | ECD = {ecd_str}"
        )

if __name__ == "__main__":
    app_controller = AppController()
    sys.exit(app_controller.run())
