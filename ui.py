from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QSplitter, QTreeView, QPushButton, 
    QTableView, QLabel, QMenuBar, QMenu, QTabBar, QComboBox,
    QHeaderView, QSlider, QDialog, QTextBrowser,
    QDoubleSpinBox, QCheckBox, QGroupBox, QFrame,
    QStackedWidget, QScrollArea
)
import json
import os
from PySide6.QtGui import QAction, QStandardItemModel
from PySide6.QtCore import Qt
import pyqtgraph as pg

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM Particle Manager")
        self.resize(1200, 800)

        self.setup_menu()

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Setup Tabs
        self.setup_overview_tab()
        self.setup_analysis_tab()
        self.setup_compression_tab()

    def setup_menu(self):
        menu_bar = self.menuBar()
        
        # File Menu
        self.file_menu = menu_bar.addMenu("File")
        self.action_open_folder = QAction("Open Folder...", self)
        self.action_open_folder.setShortcut("Ctrl+O")
        self.action_save_session = QAction("Save Session", self)
        self.action_save_session.setShortcut("Ctrl+S")
        self.action_exit = QAction("Exit", self)
        self.action_exit.setShortcut("Ctrl+Q")
        
        self.file_menu.addAction(self.action_open_folder)
        self.file_menu.addAction(self.action_save_session)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.action_exit)

        # Settings Menu
        settings_menu = menu_bar.addMenu("Settings")
        self.databar_menu = settings_menu.addMenu("Databar Fields")

        # Databar field actions
        self.db_action_wd = QAction("Working Distance (WD)", self, checkable=True)
        self.db_action_hv = QAction("SEM HV", self, checkable=True)
        self.db_action_mag = QAction("Magnification", self, checkable=True)
        self.db_action_date = QAction("Date", self, checkable=True)
        self.db_action_viewfield = QAction("Viewfield", self, checkable=True)
        self.db_action_hysteresis = QAction("Hysteresis", self, checkable=True)
        
        # Default checked
        self.db_action_wd.setChecked(True)
        self.db_action_hv.setChecked(True)
        self.db_action_mag.setChecked(True)
        self.db_action_date.setChecked(True)
        self.db_action_viewfield.setChecked(True)
        self.db_action_hysteresis.setChecked(True)

        self.databar_menu.addAction(self.db_action_wd)
        self.databar_menu.addAction(self.db_action_hv)
        self.databar_menu.addAction(self.db_action_mag)
        self.databar_menu.addAction(self.db_action_date)
        self.databar_menu.addAction(self.db_action_viewfield)
        self.databar_menu.addAction(self.db_action_hysteresis)

        # Help Menu
        self.help_menu = menu_bar.addMenu("Help")
        self.action_help = QAction("Controls & Shortcuts...", self)
        self.action_help.setShortcut("F1")
        self.help_menu.addAction(self.action_help)

    def setup_overview_tab(self):
        self.overview_tab = QWidget()
        layout = QHBoxLayout(self.overview_tab)

        splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel: Tree View and Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tree_view = QTreeView()
        self.tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_view.setSelectionMode(QTreeView.ExtendedSelection)
        left_layout.addWidget(self.tree_view)
        
        self.btn_auto_align = QPushButton("Refine Site Alignment")
        self.btn_auto_align.setStyleSheet(
            "QPushButton { background-color: #0078D7; color: white; padding: 10px; border-radius: 4px; font-weight: bold; font-size: 14px; margin-top: 5px; }"
            "QPushButton:hover { background-color: #005A9E; }"
            "QPushButton:disabled { background-color: #CCCCCC; color: #666666; }"
        )
        left_layout.addWidget(self.btn_auto_align)
        
        splitter.addWidget(left_panel)

        # Right Panel: Image View with signal tabs and databar
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.region_signal_tabs = QTabBar()
        self.region_signal_tabs.hide() # Hide initially if only 1 signal
        right_layout.addWidget(self.region_signal_tabs)

        self.region_graphics_layout = pg.GraphicsLayoutWidget()
        self.region_plot = self.region_graphics_layout.addPlot()
        self.region_plot.hideAxis('bottom')
        self.region_plot.hideAxis('left')
        self.region_plot.setAspectLocked(True)
        self.region_plot.invertY(True)
        self.region_image_item = pg.ImageItem()
        self.region_plot.addItem(self.region_image_item)
        
        self.region_scalebar = pg.ScaleBar(size=10, suffix='m', width=6, brush='w', pen='w')
        self.region_scalebar.setParentItem(self.region_plot.vb)
        self.region_scalebar.anchor((1, 1), (1, 1), offset=(-20, -20))
        
        right_layout.addWidget(self.region_graphics_layout)
        
        # Databar
        self.region_databar_widget = QWidget()
        self.region_databar_layout = QHBoxLayout(self.region_databar_widget)
        self.region_databar_layout.setContentsMargins(5, 5, 5, 5)
        self.region_lbl_wd = QLabel("WD: -")
        self.region_lbl_hv = QLabel("HV: -")
        self.region_lbl_mag = QLabel("Mag: -")
        self.region_lbl_date = QLabel("Date: -")
        self.region_lbl_viewfield = QLabel("Viewfield: -")
        self.region_lbl_hysteresis = QLabel("Hysteresis Error: N/A")
        
        self.region_databar_layout.addWidget(self.region_lbl_wd)
        self.region_databar_layout.addWidget(self.region_lbl_hv)
        self.region_databar_layout.addWidget(self.region_lbl_mag)
        self.region_databar_layout.addWidget(self.region_lbl_date)
        self.region_databar_layout.addWidget(self.region_lbl_viewfield)
        self.region_databar_layout.addWidget(self.region_lbl_hysteresis)
        self.region_databar_layout.addStretch()
        
        right_layout.addWidget(self.region_databar_widget)
        splitter.addWidget(right_panel)

        # Adjust initial splitter sizes
        splitter.setSizes([300, 900])
        layout.addWidget(splitter)
        self.tabs.addTab(self.overview_tab, "Sample & Region Overview")

    def setup_analysis_tab(self):
        self.analysis_tab = QWidget()
        layout = QHBoxLayout(self.analysis_tab)

        splitter = QSplitter(Qt.Horizontal)

        # Left Panel: Site Image View with signal tabs and databar
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.site_signal_tabs = QTabBar()
        self.site_signal_tabs.hide()
        left_layout.addWidget(self.site_signal_tabs)

        self.site_graphics_layout = pg.GraphicsLayoutWidget()
        self.site_plot = self.site_graphics_layout.addPlot()
        self.site_plot.hideAxis('bottom')
        self.site_plot.hideAxis('left')
        self.site_plot.setAspectLocked(True)
        self.site_plot.invertY(True)
        self.site_image_item = pg.ImageItem()
        self.site_plot.addItem(self.site_image_item)
        
        self.site_scalebar = pg.ScaleBar(size=10, suffix='m', width=6, brush='w', pen='w')
        self.site_scalebar.setParentItem(self.site_plot.vb)
        self.site_scalebar.anchor((1, 1), (1, 1), offset=(-20, -20))

        left_layout.addWidget(self.site_graphics_layout)
        
        # Databar
        self.site_databar_widget = QWidget()
        self.site_databar_layout = QHBoxLayout(self.site_databar_widget)
        self.site_databar_layout.setContentsMargins(5, 5, 5, 5)
        self.site_lbl_wd = QLabel("WD: -")
        self.site_lbl_hv = QLabel("HV: -")
        self.site_lbl_mag = QLabel("Mag: -")
        self.site_lbl_date = QLabel("Date: -")
        self.site_lbl_viewfield = QLabel("Viewfield: -")
        self.site_databar_layout.addWidget(self.site_lbl_wd)
        self.site_databar_layout.addWidget(self.site_lbl_hv)
        self.site_databar_layout.addWidget(self.site_lbl_mag)
        self.site_databar_layout.addWidget(self.site_lbl_date)
        self.site_databar_layout.addWidget(self.site_lbl_viewfield)
        self.site_databar_layout.addStretch()
        
        left_layout.addWidget(self.site_databar_widget)
        splitter.addWidget(left_panel)

        # Right Panel: Controls and Table
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)

        self.site_selector = QComboBox()
        controls_layout.addWidget(QLabel("Select Site:"))
        controls_layout.addWidget(self.site_selector)
        
        self.btn_next_site = QPushButton("Next Site ➔")
        self.btn_next_site.setToolTip("Advance to the next site in the list")
        controls_layout.addWidget(self.btn_next_site)
        controls_layout.addSpacing(15)

        self.lbl_measure_hint = QLabel("Click image to select particle center:")
        controls_layout.addWidget(self.lbl_measure_hint)

        # Search radius slider
        slider_box = QVBoxLayout()
        slider_box.setSpacing(2)
        slider_header = QHBoxLayout()
        self.lbl_search_radius_title = QLabel("Outer Search Radius:")
        self.lbl_search_radius_val = QLabel("2.5 µm")
        self.lbl_search_radius_val.setStyleSheet("font-weight: bold; color: #0078D7;")
        slider_header.addWidget(self.lbl_search_radius_title)
        slider_header.addWidget(self.lbl_search_radius_val)
        slider_header.addStretch()
        slider_box.addLayout(slider_header)

        self.slider_search_radius = QSlider(Qt.Horizontal)
        self.slider_search_radius.setRange(5, 100) # 0.5 to 10.0 µm in 0.1 µm increments
        self.slider_search_radius.setValue(25)     # 2.5 µm default
        self.slider_search_radius.setTickPosition(QSlider.TicksBelow)
        self.slider_search_radius.setTickInterval(10)
        slider_box.addWidget(self.slider_search_radius)
        controls_layout.addLayout(slider_box)

        # Facet Shape Slider
        shape_slider_box = QVBoxLayout()
        shape_slider_box.setSpacing(2)
        shape_slider_header = QHBoxLayout()
        self.lbl_facet_shape_title = QLabel("Facet Shape:")
        self.lbl_facet_shape_val = QLabel("Hexagon (Cryst 6-gon)")
        self.lbl_facet_shape_val.setStyleSheet("font-weight: bold; color: #0078D7;")
        shape_slider_header.addWidget(self.lbl_facet_shape_title)
        shape_slider_header.addWidget(self.lbl_facet_shape_val)
        shape_slider_header.addStretch()
        shape_slider_box.addLayout(shape_slider_header)

        self.slider_facet_shape = QSlider(Qt.Horizontal)
        self.slider_facet_shape.setRange(0, 6)
        self.slider_facet_shape.setValue(0)
        self.slider_facet_shape.setTickPosition(QSlider.TicksBelow)
        self.slider_facet_shape.setTickInterval(1)
        shape_slider_box.addWidget(self.slider_facet_shape)
        controls_layout.addLayout(shape_slider_box)

        # Primary Combined Measurement Action
        self.btn_measure_particle = QPushButton("Measure Particle (BSE + SE)  [Space]")
        self.btn_measure_particle.setToolTip("Hotkey: Spacebar or M")
        self.btn_measure_particle.setStyleSheet(
            "QPushButton { background-color: #0078D7; color: white; padding: 9px; border-radius: 4px; font-weight: bold; font-size: 13px; margin-top: 4px; }"
            "QPushButton:hover { background-color: #005A9E; }"
            "QPushButton:pressed { background-color: #004578; }"
        )
        controls_layout.addWidget(self.btn_measure_particle)

        # Secondary / manual individual measurements
        btn_layout = QHBoxLayout()
        self.btn_calc_ecd = QPushButton("Outer ECD only (BSE)")
        self.btn_measure_se = QPushButton("Facet only (SE)")
        btn_layout.addWidget(self.btn_calc_ecd)
        btn_layout.addWidget(self.btn_measure_se)
        controls_layout.addLayout(btn_layout)

        controls_layout.addSpacing(10)
        controls_layout.addWidget(QLabel("Measured Particles:"))
        
        self.particle_table = QTableView()
        self.particle_model = QStandardItemModel(0, 5)
        self.particle_model.setHorizontalHeaderLabels(["ID", "X (µm)", "Y (µm)", "ECD (nm)", "Top Facet (nm)"])
        self.particle_table.setModel(self.particle_model)
        self.particle_table.setSelectionBehavior(QTableView.SelectRows)
        self.particle_table.setSelectionMode(QTableView.ExtendedSelection)
        self.particle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        controls_layout.addWidget(self.particle_table)

        self.btn_delete_particle = QPushButton("Delete Selected Particle")
        controls_layout.addWidget(self.btn_delete_particle)

        splitter.addWidget(self.controls_widget)
        
        # Adjust initial splitter sizes
        splitter.setSizes([800, 400])

        layout.addWidget(splitter)
        self.tabs.addTab(self.analysis_tab, "Site Analysis")

    def setup_compression_tab(self):
        self.compression_tab = QWidget()
        layout = QHBoxLayout(self.compression_tab)

        main_splitter = QSplitter(Qt.Horizontal)

        # -------------------------------------------------------------
        # Left Panel: Dual Visual Confirmation Plots
        # (Top: Tilted Region Overview, Bottom: Tilted Site View)
        # -------------------------------------------------------------
        views_splitter = QSplitter(Qt.Vertical)

        # Top: Tilted Region Overview
        region_container = QWidget()
        region_layout = QVBoxLayout(region_container)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.setSpacing(2)

        reg_header = QHBoxLayout()
        lbl_reg_title = QLabel("Tilted Region Overview (SE Signal - 70° Tilt Foreshortened)")
        lbl_reg_title.setStyleSheet("font-weight: bold; color: #222; font-size: 12px;")
        reg_header.addWidget(lbl_reg_title)
        reg_header.addStretch()
        self.btn_comp_autorange_reg = QPushButton("Reset View")
        self.btn_comp_autorange_reg.setMaximumWidth(85)
        reg_header.addWidget(self.btn_comp_autorange_reg)
        region_layout.addLayout(reg_header)

        self.comp_region_graphics = pg.GraphicsLayoutWidget()
        self.comp_region_plot = self.comp_region_graphics.addPlot()
        self.comp_region_plot.hideAxis('bottom')
        self.comp_region_plot.hideAxis('left')
        self.comp_region_plot.setAspectLocked(True)
        self.comp_region_plot.invertY(True)
        self.comp_region_image_item = pg.ImageItem()
        self.comp_region_plot.addItem(self.comp_region_image_item)
        region_layout.addWidget(self.comp_region_graphics)

        views_splitter.addWidget(region_container)

        # Bottom: Tilted Site View
        site_container = QWidget()
        site_layout = QVBoxLayout(site_container)
        site_layout.setContentsMargins(0, 0, 0, 0)
        site_layout.setSpacing(2)

        site_header = QHBoxLayout()
        self.lbl_comp_site_title = QLabel("Tilted Site View (SE Signal) - Current Particle Confirmation")
        self.lbl_comp_site_title.setStyleSheet("font-weight: bold; color: #0078D7; font-size: 12px;")
        site_header.addWidget(self.lbl_comp_site_title)
        site_header.addStretch()
        self.btn_comp_autorange_site = QPushButton("Reset View")
        self.btn_comp_autorange_site.setMaximumWidth(85)
        site_header.addWidget(self.btn_comp_autorange_site)
        site_layout.addLayout(site_header)

        self.comp_site_graphics = pg.GraphicsLayoutWidget()
        self.comp_site_plot = self.comp_site_graphics.addPlot()
        self.comp_site_plot.hideAxis('bottom')
        self.comp_site_plot.hideAxis('left')
        self.comp_site_plot.setAspectLocked(True)
        self.comp_site_plot.invertY(True)
        self.comp_site_image_item = pg.ImageItem()
        self.comp_site_plot.addItem(self.comp_site_image_item)
        site_layout.addWidget(self.comp_site_graphics)

        views_splitter.addWidget(site_container)
        views_splitter.setSizes([480, 420])

        main_splitter.addWidget(views_splitter)

        # -------------------------------------------------------------
        # Right Panel: Scrollable Controls, Queue, and Mode Switcher
        # -------------------------------------------------------------
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        ctrl_layout = QVBoxLayout(scroll_content)
        ctrl_layout.setContentsMargins(6, 4, 6, 4)
        ctrl_layout.setSpacing(8)

        # 1. Orientation & Tilt Group (Always visible at top)
        align_group = QGroupBox("1. Tilt & Rotation Alignment")
        align_layout = QVBoxLayout(align_group)
        align_layout.setSpacing(5)

        # Tilt Spinbox
        tilt_row = QHBoxLayout()
        tilt_row.addWidget(QLabel("Sample Tilt Angle:"))
        self.spin_comp_tilt = QDoubleSpinBox()
        self.spin_comp_tilt.setRange(50.0, 85.0)
        self.spin_comp_tilt.setValue(70.0)
        self.spin_comp_tilt.setSingleStep(0.5)
        self.spin_comp_tilt.setSuffix(" °")
        tilt_row.addWidget(self.spin_comp_tilt)
        align_layout.addLayout(tilt_row)

        # Rotation Controls
        rot_header = QHBoxLayout()
        rot_header.addWidget(QLabel("Rotation Angle (φ):"))
        self.spin_comp_rot = QDoubleSpinBox()
        self.spin_comp_rot.setRange(0.0, 360.0)
        self.spin_comp_rot.setValue(0.0)
        self.spin_comp_rot.setSingleStep(0.5)
        self.spin_comp_rot.setSuffix(" °")
        rot_header.addWidget(self.spin_comp_rot)
        align_layout.addLayout(rot_header)

        self.slider_comp_rot = QSlider(Qt.Horizontal)
        self.slider_comp_rot.setRange(0, 3600) # 0.1 deg increments
        self.slider_comp_rot.setValue(0)
        align_layout.addWidget(self.slider_comp_rot)

        # Quick step buttons
        btn_rot_row = QHBoxLayout()
        self.btn_rot_m90 = QPushButton("-90°")
        self.btn_rot_m10 = QPushButton("-10°")
        self.btn_rot_m1 = QPushButton("-1°")
        self.btn_rot_p1 = QPushButton("+1°")
        self.btn_rot_p10 = QPushButton("+10°")
        self.btn_rot_p90 = QPushButton("+90°")
        for b in [self.btn_rot_m90, self.btn_rot_m10, self.btn_rot_m1, self.btn_rot_p1, self.btn_rot_p10, self.btn_rot_p90]:
            btn_rot_row.addWidget(b)
        align_layout.addLayout(btn_rot_row)

        # Stage Polarities: Invert X, Invert Y, Flip X/Y
        stage_pol_row = QHBoxLayout()
        self.chk_comp_inv_x = QCheckBox("Invert X")
        self.chk_comp_inv_y = QCheckBox("Invert Y")
        self.btn_comp_flip_xy = QPushButton("Flip X / Y")
        self.btn_comp_flip_xy.setCheckable(True)
        self.btn_comp_flip_xy.setToolTip("Swap X and Y axes if motor coordinate directions are interchanged")
        stage_pol_row.addWidget(self.chk_comp_inv_x)
        stage_pol_row.addWidget(self.chk_comp_inv_y)
        stage_pol_row.addWidget(self.btn_comp_flip_xy)
        align_layout.addLayout(stage_pol_row)

        # Scale Factor & Calibration Status
        scale_cal_row = QHBoxLayout()
        scale_cal_row.addWidget(QLabel("Stage Scale (s):"))
        self.spin_comp_scale = QDoubleSpinBox()
        self.spin_comp_scale.setRange(0.500, 2.000)
        self.spin_comp_scale.setValue(1.000)
        self.spin_comp_scale.setSingleStep(0.005)
        self.spin_comp_scale.setDecimals(3)
        self.spin_comp_scale.setSuffix(" ×")
        self.spin_comp_scale.setToolTip("Piezo stage travel scale factor (actual distance / nominal flat distance). Automatically updates from logged moves or can be manually adjusted.")
        scale_cal_row.addWidget(self.spin_comp_scale)

        self.lbl_comp_calibration_badge = QLabel("⚡ Prior (Uncalibrated)")
        self.lbl_comp_calibration_badge.setStyleSheet(
            "QLabel { font-size: 11px; font-weight: bold; color: #475569; background-color: #f1f5f9; "
            "border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 6px; }"
        )
        self.lbl_comp_calibration_badge.setToolTip("Calibration status, orientation uncertainty, and scale confidence.")
        scale_cal_row.addWidget(self.lbl_comp_calibration_badge, 1)
        align_layout.addLayout(scale_cal_row)

        ctrl_layout.addWidget(align_group)

        # 2. Particle Queue Status (Always visible in middle)
        table_group = QGroupBox("2. Particle Queue Status")
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(6, 6, 6, 6)
        table_layout.setSpacing(5)

        # Header bar with Tour metrics & compact filter pills
        tbl_header_row = QHBoxLayout()
        self.lbl_comp_queue_summary = QLabel("Tour: -- remaining | Est. path: -- µm")
        self.lbl_comp_queue_summary.setStyleSheet(
            "QLabel { font-size: 11px; font-weight: bold; color: #0078D7; background-color: #eff6ff; "
            "border: 1px solid #bfdbfe; border-radius: 4px; padding: 3px 8px; }"
        )
        tbl_header_row.addWidget(self.lbl_comp_queue_summary, 1)

        filter_btn_style = (
            "QPushButton { font-size: 11px; padding: 3px 8px; border-radius: 4px; border: 1px solid #d1d5db; background-color: #ffffff; }"
            "QPushButton:hover { background-color: #f3f4f6; }"
            "QPushButton:checked { background-color: #0078D7; color: white; font-weight: bold; border-color: #005A9E; }"
        )
        self.btn_comp_filter_all = QPushButton("All")
        self.btn_comp_filter_all.setCheckable(True)
        self.btn_comp_filter_all.setChecked(True)
        self.btn_comp_filter_all.setStyleSheet(filter_btn_style)

        self.btn_comp_filter_untested = QPushButton("Untested")
        self.btn_comp_filter_untested.setCheckable(True)
        self.btn_comp_filter_untested.setStyleSheet(filter_btn_style)

        self.btn_comp_filter_tested = QPushButton("Tested")
        self.btn_comp_filter_tested.setCheckable(True)
        self.btn_comp_filter_tested.setStyleSheet(filter_btn_style)

        tbl_header_row.addWidget(self.btn_comp_filter_all)
        tbl_header_row.addWidget(self.btn_comp_filter_untested)
        tbl_header_row.addWidget(self.btn_comp_filter_tested)
        table_layout.addLayout(tbl_header_row)

        self.comp_particle_table = QTableView()
        self.comp_particle_model = QStandardItemModel(0, 6)
        self.comp_particle_model.setHorizontalHeaderLabels(
            ["Step", "UID", "Hop (µm)", "Shape", "Facet (nm)", "Status"]
        )
        self.comp_particle_table.setModel(self.comp_particle_model)
        self.comp_particle_table.setSelectionBehavior(QTableView.SelectRows)
        self.comp_particle_table.setSelectionMode(QTableView.SingleSelection)
        self.comp_particle_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.comp_particle_table.horizontalHeader().setStretchLastSection(True)
        self.comp_particle_table.setStyleSheet(
            "QTableView { gridline-color: #e5e7eb; selection-background-color: #e0f2fe; selection-color: #0369a1; alternate-background-color: #f9fafb; }"
            "QHeaderView::section { background-color: #f8fafc; font-weight: bold; border: 1px solid #e2e8f0; padding: 4px; font-size: 11px; }"
        )
        self.comp_particle_table.setAlternatingRowColors(True)
        self.comp_particle_table.setMaximumHeight(210)
        table_layout.addWidget(self.comp_particle_table)

        # Single primary button for re-ordering tour
        self.btn_comp_set_current = QPushButton("Set Selected as Current && Optimize Tour ➔")
        self.btn_comp_set_current.setToolTip("Set selected particle as starting point and automatically compute the shortest tour through all untested particles")
        self.btn_comp_set_current.setStyleSheet(
            "QPushButton { background-color: #f8fafc; color: #1e293b; border: 1px solid #cbd5e1; "
            "border-radius: 5px; padding: 6px 12px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background-color: #0078D7; color: white; border-color: #005A9E; }"
            "QPushButton:pressed { background-color: #004578; }"
        )
        table_layout.addWidget(self.btn_comp_set_current)

        ctrl_layout.addWidget(table_group)

        # 3. Dynamic Workflow Mode Window (Changes between Testing and Translation)
        self.mode_group = QGroupBox("3. Active Workflow Step")
        mode_layout = QVBoxLayout(self.mode_group)
        mode_layout.setContentsMargins(4, 6, 4, 6)
        mode_layout.setSpacing(6)

        self.comp_mode_stack = QStackedWidget()

        # -------------------------------------------------------------
        # PAGE 0: Particle Testing Mode
        # -------------------------------------------------------------
        self.comp_page_test = QWidget()
        page_test_layout = QVBoxLayout(self.comp_page_test)
        page_test_layout.setContentsMargins(2, 2, 2, 2)
        page_test_layout.setSpacing(6)

        card_test = QFrame()
        card_test.setStyleSheet(
            "QFrame { background-color: #f0fdf4; border: 2px solid #107C41; border-radius: 6px; padding: 10px; }"
        )
        card_test_layout = QVBoxLayout(card_test)
        card_test_layout.setSpacing(4)

        header_test = QHBoxLayout()
        lbl_mode_a = QLabel("CURRENT PARTICLE TO TEST:")
        lbl_mode_a.setStyleSheet("font-size: 11px; font-weight: bold; color: #0E6B38;")
        header_test.addWidget(lbl_mode_a)
        header_test.addStretch()
        self.btn_comp_preview_trans = QPushButton("Preview Move ➔")
        self.btn_comp_preview_trans.setStyleSheet("QPushButton { font-size: 11px; padding: 2px 8px; }")
        header_test.addWidget(self.btn_comp_preview_trans)
        card_test_layout.addLayout(header_test)

        self.lbl_test_particle_uid = QLabel("Particle --")
        self.lbl_test_particle_uid.setStyleSheet("font-size: 22px; font-weight: bold; color: #107C41;")
        card_test_layout.addWidget(self.lbl_test_particle_uid)

        self.lbl_test_particle_details = QLabel("Site: - | Shape: - | ECD: - | Facet: -")
        self.lbl_test_particle_details.setStyleSheet("font-size: 12px; color: #333;")
        card_test_layout.addWidget(self.lbl_test_particle_details)
        page_test_layout.addWidget(card_test)

        # Big Log Test Button
        self.btn_comp_log_test = QPushButton("Log Test [Enter]  ➔  Proceed to Stage Move")
        self.btn_comp_log_test.setToolTip("Record test timestamp for current particle and transition to stage translation mode")
        self.btn_comp_log_test.setStyleSheet(
            "QPushButton { background-color: #107C41; color: white; padding: 12px; border-radius: 6px; font-weight: bold; font-size: 14px; }"
            "QPushButton:hover { background-color: #0E6B38; }"
            "QPushButton:pressed { background-color: #0B542C; }"
        )
        page_test_layout.addWidget(self.btn_comp_log_test)

        sess_row = QHBoxLayout()
        sess_row.addWidget(QLabel("Session Log:"))
        self.lbl_comp_session_file = QLabel("compression_test_YYMMDDHH.json")
        self.lbl_comp_session_file.setStyleSheet("font-weight: bold; color: #444;")
        sess_row.addWidget(self.lbl_comp_session_file, 1)
        self.lbl_comp_tested_count = QLabel("Tested: 0 / 0")
        self.lbl_comp_tested_count.setStyleSheet("font-weight: bold; color: #107C41;")
        sess_row.addWidget(self.lbl_comp_tested_count)
        page_test_layout.addLayout(sess_row)

        self.comp_mode_stack.addWidget(self.comp_page_test)

        # -------------------------------------------------------------
        # PAGE 1: Stage Translation Mode
        # -------------------------------------------------------------
        self.comp_page_trans = QWidget()
        page_trans_layout = QVBoxLayout(self.comp_page_trans)
        page_trans_layout.setContentsMargins(2, 2, 2, 2)
        page_trans_layout.setSpacing(6)

        card_trans_header = QFrame()
        card_trans_header.setStyleSheet(
            "QFrame { background-color: #eff6ff; border: 2px solid #0078D7; border-radius: 6px; padding: 8px; }"
        )
        card_trans_layout = QVBoxLayout(card_trans_header)
        card_trans_layout.setSpacing(4)

        header_trans = QHBoxLayout()
        lbl_mode_b = QLabel("STAGE TRANSLATION GUIDANCE:")
        lbl_mode_b.setStyleSheet("font-size: 11px; font-weight: bold; color: #005A9E;")
        header_trans.addWidget(lbl_mode_b)
        header_trans.addStretch()
        self.btn_comp_back_to_test = QPushButton("⬅ Back to Current Particle")
        self.btn_comp_back_to_test.setStyleSheet("QPushButton { font-size: 11px; padding: 2px 8px; }")
        header_trans.addWidget(self.btn_comp_back_to_test)
        card_trans_layout.addLayout(header_trans)

        target_row = QHBoxLayout()
        self.lbl_trans_target_uid = QLabel("Target: Particle --")
        self.lbl_trans_target_uid.setStyleSheet("font-size: 18px; font-weight: bold; color: #0078D7;")
        target_row.addWidget(self.lbl_trans_target_uid, 1)
        self.combo_comp_next = QComboBox()
        target_row.addWidget(self.combo_comp_next, 1)
        self.btn_comp_auto_next = QPushButton("Auto Next")
        self.btn_comp_auto_next.setToolTip("Select nearest untested particle")
        target_row.addWidget(self.btn_comp_auto_next)
        card_trans_layout.addLayout(target_row)
        page_trans_layout.addWidget(card_trans_header)

        # Big SmarAct Readout Box
        self.card_smaract = QFrame()
        self.card_smaract.setStyleSheet(
            "QFrame { background-color: #f8fafc; border: 1px solid #0078D7; border-radius: 6px; padding: 8px; }"
        )
        card_layout = QVBoxLayout(self.card_smaract)
        card_layout.setSpacing(2)

        readout_header = QHBoxLayout()
        readout_title = QLabel("SmarAct Move Needed:")
        readout_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #555;")
        readout_header.addWidget(readout_title)
        readout_header.addStretch()
        self.lbl_comp_distance = QLabel("Dist: 0.0 µm")
        self.lbl_comp_distance.setStyleSheet("font-size: 11px; color: #555; font-weight: bold;")
        readout_header.addWidget(self.lbl_comp_distance)
        card_layout.addLayout(readout_header)

        coords_row = QHBoxLayout()
        self.lbl_comp_dx = QLabel("ΔX = +0.00 µm")
        self.lbl_comp_dx.setStyleSheet("font-size: 22px; font-weight: bold; color: #0078D7;")
        self.lbl_comp_dy = QLabel("ΔY = +0.00 µm")
        self.lbl_comp_dy.setStyleSheet("font-size: 22px; font-weight: bold; color: #107C41;")
        coords_row.addWidget(self.lbl_comp_dx)
        coords_row.addWidget(self.lbl_comp_dy)
        card_layout.addLayout(coords_row)
        page_trans_layout.addWidget(self.card_smaract)

        # Actual Translation Inputs with Reset Button
        actual_box = QHBoxLayout()
        actual_box.addWidget(QLabel("Actual ΔX:"))
        self.spin_comp_act_dx = QDoubleSpinBox()
        self.spin_comp_act_dx.setRange(-9999.0, 9999.0)
        self.spin_comp_act_dx.setDecimals(2)
        self.spin_comp_act_dx.setSuffix(" µm")
        actual_box.addWidget(self.spin_comp_act_dx)

        actual_box.addWidget(QLabel("Actual ΔY:"))
        self.spin_comp_act_dy = QDoubleSpinBox()
        self.spin_comp_act_dy.setRange(-9999.0, 9999.0)
        self.spin_comp_act_dy.setDecimals(2)
        self.spin_comp_act_dy.setSuffix(" µm")
        actual_box.addWidget(self.spin_comp_act_dy)

        self.btn_comp_reset_trans = QPushButton("Reset")
        self.btn_comp_reset_trans.setToolTip("Reset actual translations back to predicted values")
        actual_box.addWidget(self.btn_comp_reset_trans)
        page_trans_layout.addLayout(actual_box)

        # Confirm Move Action Button
        self.btn_comp_confirm_trans = QPushButton("Confirm Move [Enter]  ➔  Ready to Test")
        self.btn_comp_confirm_trans.setToolTip("Accept stage move, refine orientation φ automatically, and enter test mode for target particle")
        self.btn_comp_confirm_trans.setStyleSheet(
            "QPushButton { background-color: #0078D7; color: white; padding: 12px; border-radius: 6px; font-weight: bold; font-size: 14px; }"
            "QPushButton:hover { background-color: #005A9E; }"
            "QPushButton:pressed { background-color: #004578; }"
        )
        page_trans_layout.addWidget(self.btn_comp_confirm_trans)

        self.lbl_comp_refine_status = QLabel("Actual translation will refine orientation automatically on Enter.")
        self.lbl_comp_refine_status.setStyleSheet("font-size: 11px; color: #666; font-style: italic;")
        page_trans_layout.addWidget(self.lbl_comp_refine_status)

        self.comp_mode_stack.addWidget(self.comp_page_trans)
        mode_layout.addWidget(self.comp_mode_stack)
        ctrl_layout.addWidget(self.mode_group)

        scroll_area.setWidget(scroll_content)
        main_splitter.addWidget(scroll_area)
        main_splitter.setSizes([750, 450])

        layout.addWidget(main_splitter)
        self.tabs.addTab(self.compression_tab, "Micro-Compression")


class HelpDialog(QDialog):
    def __init__(self, parent=None, json_path=None):
        super().__init__(parent)
        self.setWindowTitle("SEM Particle Manager - Controls & Shortcuts")
        self.resize(780, 620)
        
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        layout.addWidget(self.browser)
        
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet("QPushButton { padding: 6px 18px; font-weight: bold; }")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)
        
        if json_path is None:
            json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "help_content.json")
            
        self.load_content(json_path)

    def load_content(self, json_path: str):
        if not os.path.exists(json_path):
            self.browser.setHtml(f"<h3>Help file not found:</h3><p>{json_path}</p>")
            return
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.browser.setHtml(f"<h3>Error loading help:</h3><p>{e}</p>")
            return

        html = ["""
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; line-height: 1.5; color: #222; }
            h2 { color: #0078D7; margin-top: 0; margin-bottom: 4px; }
            .summary { font-size: 13px; color: #555; margin-bottom: 16px; }
            h3 { color: #111; margin-top: 18px; margin-bottom: 6px; padding-bottom: 3px; border-bottom: 1px solid #ddd; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 14px; }
            th { background-color: #f2f4f7; text-align: left; padding: 6px 10px; font-size: 12px; font-weight: bold; border-bottom: 2px solid #ccc; }
            td { padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top; font-size: 12px; }
            kbd { background: #f4f4f4; border: 1px solid #ccc; border-radius: 3px; padding: 2px 5px; font-family: monospace; font-size: 11px; font-weight: bold; color: #0078D7; }
            .action { font-weight: 600; color: #222; }
        </style>
        """]
        
        title = data.get("app_title", "SEM Particle Manager")
        summary = data.get("summary", "")
        html.append(f"<h2>{title}</h2>")
        if summary:
            html.append(f"<p class='summary'>{summary}</p>")
            
        for sec in data.get("sections", []):
            sec_title = sec.get("title", "")
            html.append(f"<h3>{sec_title}</h3>")
            items = sec.get("items", [])
            if items:
                html.append("<table>")
                html.append("<tr><th style='width: 28%;'>Action</th><th style='width: 34%;'>Shortcut / Control</th><th style='width: 38%;'>Description</th></tr>")
                for it in items:
                    act = it.get("action", "")
                    sc = it.get("shortcut", "")
                    desc = it.get("description", "")
                    html.append(f"<tr><td class='action'>{act}</td><td><kbd>{sc}</kbd></td><td>{desc}</td></tr>")
                html.append("</table>")
                
        self.browser.setHtml("".join(html))
