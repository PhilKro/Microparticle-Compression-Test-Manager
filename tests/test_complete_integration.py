import sys
import os
import json
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF
from models import Sample, Particle
from roi import (
    SHAPE_MODES, HexagonFacetROI, GeneralPolygonROI, EllipseFacetROI,
    create_roi_for_particle, compute_facet_parameters, compute_facet_parameters_ellipse
)
from measurement import (
    fit_facet_polygon_gradient_ransac, fit_facet_polygon_dog_ransac,
    fit_facet_ellipse_ransac
)

app = QApplication.instance() or QApplication(sys.argv)

def test_pydantic_validation():
    print("Testing Pydantic validation on sample test data...")
    sample_json = ROOT_DIR / "tests" / "fixtures" / "sample_test_data.json"
    if not sample_json.exists():
        sample_json = ROOT_DIR / "Region2" / "sample_data.json"
    with open(sample_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    sample = Sample.model_validate(data)
    print(f"  Successfully loaded sample: {sample.name} with {len(sample.images)} images.")
    particle_count = sum(len(img.particles) for img in sample.images)
    print(f"  Total valid particles: {particle_count}")
    assert particle_count >= 13, f"Expected at least 13 particles, got {particle_count}"
    for img in sample.images:
        for p in img.particles:
            assert p.shape_type in ["hexagon", "triangle", "quadrilateral", "pentagon", "polygon_6", "polygon_8", "ellipse"]
            assert (p.facet_polygon is not None) or (p.ellipse_se is not None)
    print("  Pydantic validation PASSED.")

def test_roi_creation_all_shapes():
    print("Testing ROI creation and interaction for all 7 shape modes...")
    px, py = 2.5e-9, 2.5e-9
    
    # 0: Hexagon Cryst
    p_hex = Particle(
        id=1, pixel_x=100, pixel_y=100, shape_type="hexagon",
        facet_polygon=[[100 + 30*np.cos(i*np.pi/3), 100 + 30*np.sin(i*np.pi/3)] for i in range(6)]
    )
    roi_hex = create_roi_for_particle(p_hex, px, py)
    assert isinstance(roi_hex, HexagonFacetROI), f"Expected HexagonFacetROI, got {type(roi_hex)}"
    corners_before = roi_hex.get_corners()
    assert len(corners_before) == 6
    print("  HexagonFacetROI created successfully.")

    # 1: Triangle
    p_tri = Particle(
        id=2, pixel_x=100, pixel_y=100, shape_type="triangle",
        facet_polygon=[[100, 70], [130, 120], [70, 120]]
    )
    roi_tri = create_roi_for_particle(p_tri, px, py)
    assert isinstance(roi_tri, GeneralPolygonROI)
    assert len(roi_tri.get_corners()) == 3
    print("  Triangle (GeneralPolygonROI) created successfully.")

    # 2: Quadrilateral
    p_quad = Particle(
        id=3, pixel_x=100, pixel_y=100, shape_type="quadrilateral",
        facet_polygon=[[80, 80], [120, 80], [120, 120], [80, 120]]
    )
    roi_quad = create_roi_for_particle(p_quad, px, py)
    assert isinstance(roi_quad, GeneralPolygonROI)
    assert len(roi_quad.get_corners()) == 4
    print("  Quadrilateral (GeneralPolygonROI) created successfully.")

    # 3: Pentagon
    p_penta = Particle(
        id=4, pixel_x=100, pixel_y=100, shape_type="pentagon",
        facet_polygon=[[100 + 30*np.cos(i*2*np.pi/5), 100 + 30*np.sin(i*2*np.pi/5)] for i in range(5)]
    )
    roi_penta = create_roi_for_particle(p_penta, px, py)
    assert isinstance(roi_penta, GeneralPolygonROI)
    assert len(roi_penta.get_corners()) == 5
    print("  Pentagon (GeneralPolygonROI) created successfully.")

    # 4: General 6-gon
    p_poly6 = Particle(
        id=5, pixel_x=100, pixel_y=100, shape_type="polygon_6",
        facet_polygon=[[100 + 30*np.cos(i*np.pi/3), 100 + 30*np.sin(i*np.pi/3)] for i in range(6)]
    )
    roi_poly6 = create_roi_for_particle(p_poly6, px, py)
    assert isinstance(roi_poly6, GeneralPolygonROI)
    assert len(roi_poly6.get_corners()) == 6
    print("  General 6-gon (GeneralPolygonROI) created successfully.")

    # 5: General 8-gon
    p_poly8 = Particle(
        id=6, pixel_x=100, pixel_y=100, shape_type="polygon_8",
        facet_polygon=[[100 + 30*np.cos(i*np.pi/4), 100 + 30*np.sin(i*np.pi/4)] for i in range(8)]
    )
    roi_poly8 = create_roi_for_particle(p_poly8, px, py)
    assert isinstance(roi_poly8, GeneralPolygonROI)
    assert len(roi_poly8.get_corners()) == 8
    print("  General 8-gon (GeneralPolygonROI) created successfully.")

    # 6: Ellipse
    p_ell = Particle(
        id=7, pixel_x=100, pixel_y=100, shape_type="ellipse",
        ellipse_se=[100.0, 100.0, 30.0, 20.0, float(np.radians(15.0))]
    )
    roi_ell = create_roi_for_particle(p_ell, px, py)
    assert isinstance(roi_ell, EllipseFacetROI)
    cx_m, cy_m, a_m, b_m, theta_rad = roi_ell.get_ellipse_params_m()
    assert np.isclose(cx_m, 100*px)
    assert np.isclose(cy_m, 100*py)
    assert np.isclose(a_m, 30*px)
    assert np.isclose(b_m, 20*py)
    assert np.isclose(np.degrees(theta_rad), 15.0)
    print("  EllipseFacetROI created and parameters verified.")

def test_general_polygon_independent_corner_drag():
    print("Testing GeneralPolygonROI independent corner drag...")
    px, py = 2.5e-9, 2.5e-9
    p_poly = Particle(
        id=1, pixel_x=100, pixel_y=100, shape_type="polygon_6",
        facet_polygon=[[100 + 30*np.cos(i*np.pi/3), 100 + 30*np.sin(i*np.pi/3)] for i in range(6)]
    )
    roi = create_roi_for_particle(p_poly, px, py)
    orig_corners = [QPointF(h['item'].pos()) for h in roi.handles]
    
    # Drag corner 0 by (+10 um, -5 um)
    h0 = roi.handles[0]['item']
    h0.movePoint(orig_corners[0] + QPointF(10e-6, -5e-6))
    
    new_corners = [QPointF(h['item'].pos()) for h in roi.handles]
    
    # Corner 0 must have moved
    assert new_corners[0] != orig_corners[0]
    assert np.isclose(new_corners[0].x(), orig_corners[0].x() + 10e-6)
    assert np.isclose(new_corners[0].y(), orig_corners[0].y() - 5e-6)
    
    # Corners 1 to 5 must NOT have moved at all!
    for i in range(1, 6):
        assert np.isclose(new_corners[i].x(), orig_corners[i].x()), f"Corner {i} moved in X!"
        assert np.isclose(new_corners[i].y(), orig_corners[i].y()), f"Corner {i} moved in Y!"
        
    print("  PASSED: Only corner 0 moved. All other 5 corners remained strictly stationary!")

def test_ellipse_parameter_calculation():
    print("Testing Ellipse parameter calculation...")
    px, py = 2.5e-9, 2.5e-9
    ellipse_se = [100.0, 100.0, 40.0, 25.0, 30.0]
    ellipse_bse = [102.0, 99.0, 80.0, 75.0, 0.0]
    params = compute_facet_parameters_ellipse(ellipse_se, px, py, ellipse_bse=ellipse_bse, is_manual=True)
    
    assert params["shape_type"] == "ellipse"
    assert params["semi_major_axis_nm"] == 40.0 * 2.5
    assert params["semi_minor_axis_nm"] == 25.0 * 2.5
    assert np.isclose(params["aspect_ratio"], 40.0 / 25.0)
    assert params["is_manually_fitted"] == True
    assert "facet_to_particle_center_offset_nm" in params
    
    # Ensure JSON serializable
    json_str = json.dumps(params)
    assert len(json_str) > 50
    print("  PASSED: Ellipse metrics calculated and JSON serializable.")

def test_next_site_button():
    print("Testing Next Site button navigation...")
    from main import AppController
    ctrl = AppController()
    # Populate site_selector with dummy sites
    ctrl.window.site_selector.clear()
    ctrl.window.site_selector.addItem("Site 1", 0)
    ctrl.window.site_selector.addItem("Site 2", 1)
    ctrl.window.site_selector.addItem("Site 3", 2)
    assert ctrl.window.site_selector.count() == 3
    ctrl.window.site_selector.setCurrentIndex(0)
    assert ctrl.window.site_selector.currentIndex() == 0
    
    # Click Next Site
    ctrl.on_next_site_clicked()
    assert ctrl.window.site_selector.currentIndex() == 1
    ctrl.on_next_site_clicked()
    assert ctrl.window.site_selector.currentIndex() == 2
    # Wrap around to 0
    ctrl.on_next_site_clicked()
    assert ctrl.window.site_selector.currentIndex() == 0
    print("  PASSED: Next Site cycles sequentially and wraps around.")

if __name__ == "__main__":
    test_pydantic_validation()
    test_roi_creation_all_shapes()
    test_general_polygon_independent_corner_drag()
    test_ellipse_parameter_calculation()
    test_next_site_button()
    print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
