import sys
import os
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from roi import HexagonFacetROI, compute_facet_parameters

def internal_angles(pts):
    n_pts = len(pts)
    angs = []
    for i in range(n_pts):
        v1 = pts[(i-1)%n_pts] - pts[i]
        v2 = pts[(i+1)%n_pts] - pts[i]
        cos_th = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angs.append(float(np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0)))))
    return angs

def test_edge_shift_drag():
    print("==================================================")
    print("TESTING HEXAGON EDGE SHIFT-DRAG IMPLEMENTATION")
    print("==================================================")

    R = 100.0
    angles = [k * np.pi / 3.0 for k in range(6)]
    init_corners = [[R * np.cos(a), R * np.sin(a)] for a in angles]
    roi = HexagonFacetROI(corners=init_corners)

    assert len(roi.handles) == 6, f"Expected 6 handles, got {len(roi.handles)}"
    
    # 1. Test mathematical perform_edge_shift_drag on all 6 edges
    corners_arr = np.array(init_corners)
    for edge_idx in range(6):
        k = edge_idx
        k_next = (k + 1) % 6
        pA = corners_arr[k]
        pB = corners_arr[k_next]
        mid = 0.5 * (pA + pB)
        outward_n = mid / np.linalg.norm(mid)
        
        orig_len = float(np.linalg.norm(pB - pA))

        # Test outward drag (+25 px) -> lengthens/shortens edge while keeping 120 deg
        out_P = roi.perform_edge_shift_drag(edge_idx, 25.0 * outward_n, corners_arr)
        # Test inward drag (-25 px)
        in_P = roi.perform_edge_shift_drag(edge_idx, -25.0 * outward_n, corners_arr)

        # Check internal angles
        out_angs = internal_angles(out_P)
        in_angs = internal_angles(in_P)
        for a in out_angs:
            assert np.isclose(a, 120.0, atol=1e-3), f"Edge {edge_idx} outward angle {a} != 120°"
        for a in in_angs:
            assert np.isclose(a, 120.0, atol=1e-3), f"Edge {edge_idx} inward angle {a} != 120°"

        # Check that ONLY corners k and k_next moved!
        for i in range(6):
            if i not in (k, k_next):
                assert np.allclose(out_P[i], corners_arr[i]), f"Corner {i} moved during outward drag of edge {k}!"
                assert np.allclose(in_P[i], corners_arr[i]), f"Corner {i} moved during inward drag of edge {k}!"

        # Check that endpoints slide along adjacent edge rays
        pA_prev = corners_arr[(k - 1) % 6]
        pB_next = corners_arr[(k + 2) % 6]
        rayA = (corners_arr[k] - pA_prev) / np.linalg.norm(corners_arr[k] - pA_prev)
        rayB = (corners_arr[k_next] - pB_next) / np.linalg.norm(corners_arr[k_next] - pB_next)

        deltaA = out_P[k] - corners_arr[k]
        deltaB = out_P[k_next] - corners_arr[k_next]
        assert np.isclose(abs(np.dot(deltaA / np.linalg.norm(deltaA), rayA)), 1.0), "Corner A did not slide along adjacent ray!"
        assert np.isclose(abs(np.dot(deltaB / np.linalg.norm(deltaB), rayB)), 1.0), "Corner B did not slide along adjacent ray!"

        print(f"  Edge #{edge_idx}: Inward and Outward shift-drags verified! 120.0° locked.")

    # 2. Test Qt MouseDragEvent integration
    class MockDragEvent:
        def __init__(self, pos, start_pos, is_start=False, is_finish=False, is_shift=True):
            self._pos = QtCore.QPointF(pos[0], pos[1])
            self._start = QtCore.QPointF(start_pos[0], start_pos[1])
            self._is_start = is_start
            self._is_finish = is_finish
            self._modifiers = QtCore.Qt.KeyboardModifier.ShiftModifier if is_shift else QtCore.Qt.KeyboardModifier.NoModifier
            self._accepted = False
        def pos(self): return self._pos
        def buttonDownPos(self): return self._start
        def isStart(self): return self._is_start
        def isFinish(self): return self._is_finish
        def modifiers(self): return self._modifiers
        def accept(self): self._accepted = True

    # Click on edge 0 (midpoint of corner 0 and 1)
    mid0 = 0.5 * (corners_arr[0] + corners_arr[1])
    n0 = mid0 / np.linalg.norm(mid0)

    # Start event
    ev_start = MockDragEvent(mid0, mid0, is_start=True, is_finish=False, is_shift=True)
    roi.mouseDragEvent(ev_start)
    assert roi.edge_drag_idx == 0, f"Expected edge_drag_idx == 0, got {roi.edge_drag_idx}"
    assert roi.edge_drag_start_corners is not None

    # Move event: drag outward by 20 px along normal
    cur_pos = mid0 + 20.0 * n0
    ev_move = MockDragEvent(cur_pos, mid0, is_start=False, is_finish=False, is_shift=True)
    roi.mouseDragEvent(ev_move)

    # Verify handles were updated
    new_h_corners = np.array([[h['item'].pos().x(), h['item'].pos().y()] for h in roi.handles])
    angs_after_move = internal_angles(new_h_corners)
    for a in angs_after_move:
        assert np.isclose(a, 120.0, atol=1e-3), f"Handle angle {a} != 120°"

    # Finish event
    ev_finish = MockDragEvent(cur_pos, mid0, is_start=False, is_finish=True, is_shift=True)
    roi.mouseDragEvent(ev_finish)
    assert roi.edge_drag_idx is None, "edge_drag_idx should be cleaned up after finish"
    assert roi.edge_drag_start_corners is None, "edge_drag_start_corners should be cleaned up after finish"

    print("  Qt mouseDragEvent simulation: PASSED perfectly!")

    # 3. Parameter computation verification
    px = 2.5e-9
    py = 2.5e-9
    params = compute_facet_parameters(new_h_corners.tolist(), px, py, is_manual=True)
    assert params["shape_type"] == "hexagon"
    assert "diameter_nm" in params
    assert "edge_lengths_nm" in params
    assert len(params["edge_lengths_nm"]) == 6
    print("  Parameter computation: PASSED!")

    print("\nALL EDGE SHIFT-DRAG TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_edge_shift_drag()
