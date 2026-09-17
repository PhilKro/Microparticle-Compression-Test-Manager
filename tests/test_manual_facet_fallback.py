import sys
import os
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from models import Particle, ImageRecord, ImageType
from roi import (
    SHAPE_MODES, HexagonFacetROI, GeneralPolygonROI, EllipseFacetROI,
    create_roi_for_particle
)
from main import AppController

app = QApplication.instance() or QApplication(sys.argv)

def test_placeholder_creation_and_table_display():
    print("Testing placeholder candidate synthesis...")
    ctrl = AppController()
    
    # Setup dummy site
    site = ImageRecord(
        filename="dummy_site.tif",
        classification=ImageType.SITE,
        metadata={"MAIN": {"PixelSizeX": "2.5e-9", "PixelSizeY": "2.5e-9"}}
    )
    ctrl.current_site = site
    
    p = Particle(id=1, pixel_x=200.0, pixel_y=200.0, ecd=500e-9)
    site.particles.append(p)
    
    # Synthesize placeholder
    ctrl.synthesize_default_facet(p, shape_key="hexagon", is_placeholder=True)
    
    assert p.facet_polygon is not None, "Facet polygon should be created as candidate"
    assert len(p.facet_polygon) == 6, "Hexagon should have 6 corners"
    assert p.top_facet_diameter is None, "Placeholder top_facet_diameter must remain None"
    assert p.top_facet_area is None, "Placeholder top_facet_area must remain None"
    assert p.is_manually_fitted is False, "Placeholder is not yet manually fitted"
    
    # Check table displays "-"
    ctrl.populate_particle_table()
    table_item = ctrl.window.particle_model.item(0, 4)
    assert table_item.text() == "-", f"Expected table to display '-', got '{table_item.text()}'"
    print("  PASSED: Placeholder created with None diameter and '-' in table.")

def test_roi_placeholder_visuals_and_interaction():
    print("Testing ROI placeholder visuals (red) and transition to registered (yellow)...")
    ctrl = AppController()
    site = ImageRecord(
        filename="dummy_site.tif",
        classification=ImageType.SITE,
        metadata={"MAIN": {"PixelSizeX": "2.5e-9", "PixelSizeY": "2.5e-9"}}
    )
    ctrl.current_site = site
    px, py = 2.5e-9, 2.5e-9
    
    p = Particle(id=1, pixel_x=150.0, pixel_y=150.0, ecd=400e-9)
    site.particles.append(p)
    ctrl.synthesize_default_facet(p, shape_key="hexagon", is_placeholder=True)
    
    # Create ROI
    roi = create_roi_for_particle(p, px, py)
    assert roi is not None
    assert getattr(roi, 'is_placeholder', False) is True, "ROI should be initialized as placeholder"
    assert roi.pen.color().name().upper() == "#FF1744", f"Expected red pen, got {roi.pen.color().name()}"
    assert roi.handlePen.color().name().upper() == "#FF1744", f"Expected red handlePen, got {roi.handlePen.color().name()}"
    print("  PASSED: Initial ROI is RED placeholder.")
    
    # Attach to controller
    ctrl.active_facet_roi = roi
    ctrl.active_particle_id = p.id
    
    # Simulate user interaction (drag started)
    ctrl.on_facet_roi_changed(roi)
    assert getattr(roi, 'is_placeholder', False) is False, "ROI should transition to non-placeholder on interaction"
    assert roi.pen.color().name().upper() == "#FFD600", f"Expected yellow pen after drag, got {roi.pen.color().name()}"
    assert roi.handlePen.color().name().upper() == "#FFD600", f"Expected yellow handlePen after drag, got {roi.handlePen.color().name()}"
    
    # Simulate user interaction finished (mouse released)
    ctrl.on_facet_roi_change_finished(roi)
    assert p.top_facet_diameter is not None and p.top_facet_diameter > 0, "Registered top_facet_diameter should be positive float"
    assert p.top_facet_area is not None and p.top_facet_area > 0, "Registered top_facet_area should be positive float"
    assert p.is_manually_fitted is True, "Particle should be marked as manually fitted"
    
    # Check table now displays numeric size
    table_item = ctrl.window.particle_model.item(0, 4)
    assert table_item.text() != "-", "Table should no longer show '-'"
    val = float(table_item.text())
    assert val > 0, f"Expected positive nm facet size, got {val}"
    print(f"  PASSED: Interaction registered manual size {val:.1f} nm and turned yellow.")

def test_universal_fallbacks_all_7_modes():
    print("Testing universal ROI fallback creation across all 7 shape modes...")
    px, py = 2.0e-9, 2.0e-9
    for mode in SHAPE_MODES:
        key = mode["key"]
        p = Particle(id=1, pixel_x=100.0, pixel_y=100.0, shape_type=key)
        # Completely empty facet data
        roi = create_roi_for_particle(p, px, py)
        assert roi is not None, f"Fallback failed for mode {key}"
        assert getattr(roi, 'is_placeholder', False) is True, f"Mode {key} should default to placeholder"
        d, a = roi.get_equivalent_diameter_and_area()
        assert d > 0 and a > 0, f"Valid non-zero fallback geometry required for {key}"
        print(f"  Mode {key}: fallback ROI successfully created (d={d*1e9:.1f} nm).")
    print("  PASSED: Universal fallbacks working for all 7 modes.")

def test_shape_slider_change_active_particle():
    print("Testing slider shape switching for active particle...")
    ctrl = AppController()
    site = ImageRecord(
        filename="dummy_site.tif",
        classification=ImageType.SITE,
        metadata={"MAIN": {"PixelSizeX": "2.5e-9", "PixelSizeY": "2.5e-9"}}
    )
    ctrl.current_site = site
    p = Particle(id=1, pixel_x=120.0, pixel_y=120.0, ecd=500e-9)
    site.particles.append(p)
    ctrl.synthesize_default_facet(p, shape_key="hexagon", is_placeholder=True)
    ctrl.update_active_facet_roi(p)
    
    assert ctrl.active_particle_id == p.id
    assert p.shape_type == "hexagon"
    
    # Change slider to triangle (index 1)
    ctrl.on_facet_shape_slider_changed(1)
    assert p.shape_type == "triangle"
    assert len(p.facet_polygon) == 3
    assert p.top_facet_diameter is None, "Placeholder status preserved across shape morph"
    assert ctrl.active_facet_roi.is_placeholder is True
    
    # Change slider to ellipse (index 6)
    ctrl.on_facet_shape_slider_changed(6)
    assert p.shape_type == "ellipse"
    assert p.ellipse_se is not None
    assert p.facet_polygon is None
    assert isinstance(ctrl.active_facet_roi, EllipseFacetROI)
    print("  PASSED: Slider dynamically reshapes candidate ROI while preserving placeholder status.")

if __name__ == "__main__":
    test_placeholder_creation_and_table_display()
    test_roi_placeholder_visuals_and_interaction()
    test_universal_fallbacks_all_7_modes()
    test_shape_slider_change_active_particle()
    print("\nALL MANUAL FACET FALLBACK TESTS PASSED!")
