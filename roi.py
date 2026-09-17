import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from models import Particle

SHAPE_MODES = [
    {"key": "hexagon", "name": "Hexagon (Cryst 6-gon)", "n_sides": 6, "type": "crystallographic"},
    {"key": "triangle", "name": "Triangle (3-gon)", "n_sides": 3, "type": "general_polygon"},
    {"key": "quadrilateral", "name": "Quadrilateral (4-gon)", "n_sides": 4, "type": "general_polygon"},
    {"key": "pentagon", "name": "Pentagon (5-gon)", "n_sides": 5, "type": "general_polygon"},
    {"key": "polygon_6", "name": "General 6-gon", "n_sides": 6, "type": "general_polygon"},
    {"key": "polygon_8", "name": "General 8-gon", "n_sides": 8, "type": "general_polygon"},
    {"key": "ellipse", "name": "Ellipse / Circle", "n_sides": 0, "type": "ellipse"},
]

def compute_facet_parameters(
    corners_px: List[List[float]],
    pixel_size_x: float,
    pixel_size_y: float,
    ellipse_bse: Optional[List[float]] = None,
    is_manual: bool = False,
    shape_type: str = "hexagon"
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of crystallographic, morphological, and geometric
    parameters from the facet polygon corners for ground-truth dataset collection.
    Supports arbitrary polygons (triangles, quads, pentagons, hexagons, octagons).
    """
    c = np.array(corners_px, dtype=float)
    n = len(c)
    px = float(pixel_size_x)
    py = float(pixel_size_y)
    
    cx = float(np.mean(c[:, 0]))
    cy = float(np.mean(c[:, 1]))
    
    # Edges and edge lengths in pixels and nm
    edges = np.roll(c, -1, axis=0) - c
    edge_lens_px = [float(np.linalg.norm(e)) for e in edges]
    edge_lens_nm = [float(l * px * 1e9) for l in edge_lens_px]
    
    # Perpendicular face distances (Wulff apothem distances) from center to each facet face
    face_dists_px = []
    face_dists_nm = []
    for k in range(n):
        e = edges[k]
        norm_e = np.linalg.norm(e)
        if norm_e > 1e-12:
            n_vec = np.array([e[1], -e[0]]) / norm_e
            d = abs(float(np.dot(c[k] - np.array([cx, cy]), n_vec)))
        else:
            d = 0.0
        face_dists_px.append(d)
        face_dists_nm.append(d * px * 1e9)
        
    # Area via shoelace formula
    area_px2 = float(0.5 * np.abs(np.dot(c[:, 0], np.roll(c[:, 1], 1)) - np.dot(c[:, 1], np.roll(c[:, 0], 1))))
    area_m2 = float(area_px2 * px * py)
    ecd_px = float(2.0 * np.sqrt(area_px2 / np.pi)) if area_px2 > 0 else 0.0
    ecd_m = float(ecd_px * px)
    perimeter_px = float(np.sum(edge_lens_px))
    perimeter_m = float(perimeter_px * px)
    roundness = float(4.0 * np.pi * area_px2 / (perimeter_px ** 2)) if perimeter_px > 0 else 0.0
    
    # Orientation: angle of corner 0 from center
    ang0 = float(np.degrees(np.arctan2(c[0, 1] - cy, c[0, 0] - cx))) if n > 0 else 0.0
    
    # Internal angles at all vertices
    angles_deg = []
    for k in range(n):
        e1 = c[k] - c[(k - 1) % n]
        e2 = c[(k + 1) % n] - c[k]
        turn = np.degrees(np.arctan2(e1[0] * e2[1] - e1[1] * e2[0], e1[0] * e2[0] + e1[1] * e2[1]))
        angles_deg.append(float(round(180.0 - turn, 4)))
        
    min_edge = min(edge_lens_nm) if edge_lens_nm else 0.0
    max_edge = max(edge_lens_nm) if edge_lens_nm else 1.0
    min_dist = min(face_dists_nm) if face_dists_nm else 0.0
    max_dist = max(face_dists_nm) if face_dists_nm else 1.0
    
    params: Dict[str, Any] = {
        'shape_type': shape_type,
        'n_sides': n,
        'is_manually_fitted': is_manual,
        'center_px': [round(cx, 3), round(cy, 3)],
        'center_um': [round(cx * px * 1e6, 4), round(cy * py * 1e6, 4)],
        'diameter_nm': round(ecd_m * 1e9, 2),
        'diameter_m': ecd_m,
        'area_um2': round(area_m2 * 1e12, 5),
        'area_m2': area_m2,
        'perimeter_um': round(perimeter_m * 1e6, 4),
        'roundness': round(roundness, 4),
        'orientation_deg': round(ang0, 2),
        'edge_lengths_nm': [round(l, 2) for l in edge_lens_nm],
        'face_distances_nm': [round(d, 2) for d in face_dists_nm],
        'edge_length_min_max_ratio': round(min_edge / max_edge, 4) if max_edge > 0 else 1.0,
        'face_distance_min_max_ratio': round(min_dist / max_dist, 4) if max_dist > 0 else 1.0,
        'internal_angles_deg': angles_deg,
        'corners_px': [[round(x, 3), round(y, 3)] for x, y in c],
        'corners_um': [[round(x * px * 1e6, 4), round(y * py * 1e6, 4)] for x, y in c]
    }
    
    if ellipse_bse and len(ellipse_bse) >= 5:
        bse_cx, bse_cy, bse_a, bse_b, bse_th = ellipse_bse
        dx_nm = (cx - bse_cx) * px * 1e9
        dy_nm = (cy - bse_cy) * py * 1e9
        offset_dist_nm = float(np.hypot(dx_nm, dy_nm))
        bse_d_m = 2.0 * float(np.sqrt(bse_a * bse_b)) * px
        
        params['bse_particle'] = {
            'center_px': [round(bse_cx, 3), round(bse_cy, 3)],
            'ecd_nm': round(bse_d_m * 1e9, 2),
            'semi_major_nm': round(bse_a * px * 1e9, 2),
            'semi_minor_nm': round(bse_b * py * 1e9, 2),
            'orientation_deg': round(float(np.degrees(bse_th)), 2),
            'aspect_ratio': round(float(bse_a / bse_b), 4) if bse_b > 0 else 1.0
        }
        params['facet_to_particle_center_offset_nm'] = [round(dx_nm, 2), round(dy_nm, 2)]
        params['facet_to_particle_center_offset_dist_nm'] = round(offset_dist_nm, 2)
        params['facet_to_outer_diameter_ratio'] = round(ecd_m / bse_d_m, 4) if bse_d_m > 0 else None
        
    return params


def compute_facet_parameters_ellipse(
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of morphological parameters for an elliptical / circular facet.
    Supports either:
      compute_facet_parameters_ellipse(ellipse_params, pixel_size_x, pixel_size_y, ellipse_bse=..., is_manual=...)
      or
      compute_facet_parameters_ellipse(cx_px, cy_px, a_px, b_px, theta_rad, pixel_size_x, pixel_size_y, ...)
    """
    if len(args) > 0 and isinstance(args[0], (list, tuple, np.ndarray)) and len(args[0]) >= 5:
        cx_px, cy_px, a_px, b_px, theta_rad = [float(v) for v in args[0][:5]]
        rem_args = args[1:]
    else:
        cx_px, cy_px, a_px, b_px, theta_rad = [float(v) for v in args[:5]]
        rem_args = args[5:]

    pixel_size_x = kwargs.get('pixel_size_x', rem_args[0] if len(rem_args) > 0 else 1.0)
    pixel_size_y = kwargs.get('pixel_size_y', rem_args[1] if len(rem_args) > 1 else pixel_size_x)
    ellipse_bse = kwargs.get('ellipse_bse', rem_args[2] if len(rem_args) > 2 else None)
    is_manual = kwargs.get('is_manual', rem_args[3] if len(rem_args) > 3 else False)
    px = float(pixel_size_x)
    py = float(pixel_size_y)
    
    a_m = float(a_px * px)
    b_m = float(b_px * py)
    area_m2 = float(np.pi * a_m * b_m)
    ecd_m = float(2.0 * np.sqrt(a_m * b_m))
    
    # Ramanujan's perimeter approximation
    h_param = ((a_m - b_m) ** 2) / ((a_m + b_m) ** 2) if (a_m + b_m) > 0 else 0.0
    perimeter_m = float(np.pi * (a_m + b_m) * (1.0 + (3.0 * h_param) / (10.0 + np.sqrt(4.0 - 3.0 * h_param))))
    roundness = float(4.0 * np.pi * area_m2 / (perimeter_m ** 2)) if perimeter_m > 0 else 0.0
    
    ang_deg = float(np.degrees(theta_rad))
    aspect_ratio = float(max(a_px, b_px) / min(a_px, b_px)) if min(a_px, b_px) > 0 else 1.0
    
    params: Dict[str, Any] = {
        'shape_type': 'ellipse',
        'is_manually_fitted': is_manual,
        'center_px': [round(cx_px, 3), round(cy_px, 3)],
        'center_um': [round(cx_px * px * 1e6, 4), round(cy_px * py * 1e6, 4)],
        'diameter_nm': round(ecd_m * 1e9, 2),
        'diameter_m': ecd_m,
        'area_um2': round(area_m2 * 1e12, 5),
        'area_m2': area_m2,
        'perimeter_um': round(perimeter_m * 1e6, 4),
        'roundness': round(roundness, 4),
        'orientation_deg': round(ang_deg, 2),
        'semi_major_nm': round(max(a_m, b_m) * 1e9, 2),
        'semi_minor_nm': round(min(a_m, b_m) * 1e9, 2),
        'semi_major_axis_nm': round(max(a_m, b_m) * 1e9, 2),
        'semi_minor_axis_nm': round(min(a_m, b_m) * 1e9, 2),
        'aspect_ratio': round(aspect_ratio, 4),
        'ellipse_params': [cx_px, cy_px, a_px, b_px, theta_rad]
    }
    
    if ellipse_bse and len(ellipse_bse) >= 5:
        bse_cx, bse_cy, bse_a, bse_b, bse_th = ellipse_bse
        dx_nm = (cx_px - bse_cx) * px * 1e9
        dy_nm = (cy_px - bse_cy) * py * 1e9
        offset_dist_nm = float(np.hypot(dx_nm, dy_nm))
        bse_d_m = 2.0 * float(np.sqrt(bse_a * bse_b)) * px
        
        params['bse_particle'] = {
            'center_px': [round(bse_cx, 3), round(bse_cy, 3)],
            'ecd_nm': round(bse_d_m * 1e9, 2),
            'semi_major_nm': round(bse_a * px * 1e9, 2),
            'semi_minor_nm': round(bse_b * py * 1e9, 2),
            'orientation_deg': round(float(np.degrees(bse_th)), 2),
            'aspect_ratio': round(float(bse_a / bse_b), 4) if bse_b > 0 else 1.0
        }
        params['facet_to_particle_center_offset_nm'] = [round(dx_nm, 2), round(dy_nm, 2)]
        params['facet_to_particle_center_offset_dist_nm'] = round(offset_dist_nm, 2)
        params['facet_to_outer_diameter_ratio'] = round(ecd_m / bse_d_m, 4) if bse_d_m > 0 else None
        
    return params


class GeneralPolygonROI(pg.ROI):
    """
    An interactive general polygon ROI for arbitrary n-gons (triangles, quadrilaterals,
    pentagons, general 6-gons, general 8-gons).
    
    Corner Drag: Moves ONLY the dragged corner freely while all other corners stay completely fixed!
    Body Drag: Translates the entire polygon.
    """
    def __init__(self, corners: List[List[float]], **kwargs):
        super().__init__([0, 0], [1, 1], **kwargs)
        self.translatable = True
        
        self.setPen(pg.mkPen(color='#FFD600', width=2))
        self.handlePen = pg.mkPen(color='#FFD600', width=1.5)
        self.handleHoverPen = pg.mkPen(color='#FFFFFF', width=2.5)
        self.handleSize = 8
        
        for k in range(len(corners)):
            pt = QtCore.QPointF(corners[k][0], corners[k][1])
            self.addFreeHandle(pt, name=f'corner_{k}')

    def shape(self) -> QtGui.QPainterPath:
        p = QtGui.QPainterPath()
        if len(self.handles) < 3:
            return p
        p.moveTo(self.handles[0]['item'].pos())
        for i in range(1, len(self.handles)):
            p.lineTo(self.handles[i]['item'].pos())
        p.closeSubpath()
        return p

    def paint(self, p: QtGui.QPainter, opt: QtWidgets.QStyleOptionGraphicsItem, widget: QtWidgets.QWidget):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setPen(self.currentPen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 214, 0, 30)))
        p.drawPath(self.shape())

    def get_hexagon_corners(self) -> List[List[float]]:
        """Returns all corners in parent/view physical coordinates (meters)."""
        corners = []
        for h in self.handles:
            pt = self.mapToParent(h['item'].pos())
            corners.append([float(pt.x()), float(pt.y())])
        return corners

    def get_hexagon_center(self) -> Tuple[float, float]:
        corners = np.array(self.get_hexagon_corners())
        return float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1]))

    def get_equivalent_diameter_and_area(self) -> Tuple[float, float]:
        corners = np.array(self.get_hexagon_corners())
        x = corners[:, 0]
        y = corners[:, 1]
        area = float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        diameter = float(2.0 * np.sqrt(area / np.pi)) if area > 0 else 0.0
        return diameter, area

    get_corners = get_hexagon_corners
    get_center = get_hexagon_center


class EllipseFacetROI(pg.ROI):
    """
    An interactive ellipse ROI for circular and elliptical facets.
    
    Features:
    - Scale handles for major axis and minor axis.
    - Rotate handle for orientation angle.
    - Body drag for translation.
    """
    def __init__(self, cx: float, cy: float, rx: float, ry: float, angle: float = 0.0, **kwargs):
        pos = [cx - rx, cy - ry]
        size = [2.0 * rx, 2.0 * ry]
        super().__init__(pos, size, **kwargs)
        self.setAngle(angle, center=[0.5, 0.5])
        
        self.setPen(pg.mkPen(color='#FFD600', width=2))
        self.handlePen = pg.mkPen(color='#FFD600', width=1.5)
        self.handleHoverPen = pg.mkPen(color='#FFFFFF', width=2.5)
        self.handleSize = 8
        
        # Scale handles on right (axis_x) and top (axis_y)
        self.addScaleHandle([1.0, 0.5], [0.5, 0.5], name='axis_x')
        self.addScaleHandle([0.5, 0.0], [0.5, 0.5], name='axis_y')
        # Rotate handle at corner
        self.addRotateHandle([1.0, 0.0], [0.5, 0.5], name='rotate')

    def paint(self, p: QtGui.QPainter, opt: QtWidgets.QStyleOptionGraphicsItem, widget: QtWidgets.QWidget):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setPen(self.currentPen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 214, 0, 30)))
        r = QtCore.QRectF(0, 0, self.state['size'][0], self.state['size'][1])
        p.drawEllipse(r)

    def shape(self) -> QtGui.QPainterPath:
        p = QtGui.QPainterPath()
        r = QtCore.QRectF(0, 0, self.state['size'][0], self.state['size'][1])
        p.addEllipse(r)
        return p

    def get_ellipse_params(self) -> Tuple[float, float, float, float, float]:
        """Returns [cx, cy, a, b, theta_rad] in parent coordinates (meters)."""
        size = self.state['size']
        rx = 0.5 * abs(size[0])
        ry = 0.5 * abs(size[1])
        center = self.mapToParent(QtCore.QPointF(0.5 * size[0], 0.5 * size[1]))
        cx = float(center.x())
        cy = float(center.y())
        th_rad = float(np.radians(self.state['angle']))
        return cx, cy, rx, ry, th_rad

    get_ellipse_params_m = get_ellipse_params

    def get_hexagon_center(self) -> Tuple[float, float]:
        cx, cy, _, _, _ = self.get_ellipse_params()
        return cx, cy

    def get_equivalent_diameter_and_area(self) -> Tuple[float, float]:
        _, _, rx, ry, _ = self.get_ellipse_params()
        area = float(np.pi * rx * ry)
        ecd = float(2.0 * np.sqrt(rx * ry))
        return ecd, area

    def get_hexagon_corners(self) -> List[List[float]]:
        """Generates 36-point boundary approximation for rendering / saving."""
        cx, cy, rx, ry, th = self.get_ellipse_params()
        t = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        x_rot = rx * np.cos(t) * np.cos(th) - ry * np.sin(t) * np.sin(th)
        y_rot = rx * np.cos(t) * np.sin(th) + ry * np.sin(t) * np.cos(th)
        return [[float(cx + x), float(cy + y)] for x, y in zip(x_rot, y_rot)]


class HexagonFacetROI(pg.ROI):
    """
    An interactive regular / equiangular hexagonal ROI for crystallographic particle facets.
    
    Interaction Modes:
    1. Normal Drag (Corner): Scales and rotates the entire hexagon self-similarly around
       its geometric center, maintaining exact shape similarity.
    2. Shift + Drag (Corner): Extends or retracts the dragged corner and its two adjacent
       corners strictly along the direction of the second-next edges, keeping the other
       three corners stationary and PRESERVING all 120.0° internal angles.
    3. Body Drag: Translates the entire hexagon across the particle surface.
    """
    def __init__(
        self,
        corners: Optional[List[List[float]]] = None,
        cx: float = 0.0,
        cy: float = 0.0,
        radius: float = 0.0,
        angle: float = 0.0,
        **kwargs
    ):
        super().__init__([0, 0], [1, 1], **kwargs)
        self.translatable = True
        self.resizable = False
        self.rotatable = False
        
        self.setPen(pg.mkPen(color='#FFD600', width=2))
        self.handlePen = pg.mkPen(color='#FFD600', width=1.5)
        self.handleHoverPen = pg.mkPen(color='#FFFFFF', width=2.5)
        self.handleSize = 8
        
        self.drag_start_corners: Optional[np.ndarray] = None
        self.drag_start_pos: Optional[np.ndarray] = None
        self.active_handle_idx: Optional[int] = None
        self.drag_is_shift: Optional[bool] = None
        
        self.edge_drag_idx: Optional[int] = None
        self.edge_drag_start_corners: Optional[np.ndarray] = None
        self.edge_drag_start_pos: Optional[np.ndarray] = None
        
        if corners is not None and len(corners) == 6:
            init_corners = np.array(corners, dtype=float)
        else:
            th_deg = np.radians(angle)
            init_corners = np.array([
                [cx + radius * np.cos(k * np.pi / 3.0 + th_deg),
                 cy + radius * np.sin(k * np.pi / 3.0 + th_deg)]
                for k in range(6)
            ])
            
        for k in range(6):
            pt = QtCore.QPointF(init_corners[k, 0], init_corners[k, 1])
            self.addFreeHandle(pt, name=f'corner_{k}')

    def shape(self) -> QtGui.QPainterPath:
        p = QtGui.QPainterPath()
        if len(self.handles) < 6:
            return p
        p.moveTo(self.handles[0]['item'].pos())
        for i in range(1, 6):
            p.lineTo(self.handles[i]['item'].pos())
        p.closeSubpath()
        return p

    def paint(self, p: QtGui.QPainter, opt: QtWidgets.QStyleOptionGraphicsItem, widget: QtWidgets.QWidget):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setPen(self.currentPen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 214, 0, 30)))
        p.drawPath(self.shape())
        
        # Highlight active dragged edge during Shift+drag
        if self.edge_drag_idx is not None and len(self.handles) >= 6:
            k = self.edge_drag_idx
            k_next = (k + 1) % 6
            p1 = self.handles[k]['item'].pos()
            p2 = self.handles[k_next]['item'].pos()
            
            p.setPen(pg.mkPen(color='#00E5FF', width=3, cosmetic=True))
            p.drawLine(p1, p2)

    def _find_closest_edge(self, pos: np.ndarray, corners: np.ndarray) -> Tuple[int, float]:
        best_k = 0
        min_dist = float('inf')
        for k in range(6):
            p1 = corners[k]
            p2 = corners[(k + 1) % 6]
            v = p2 - p1
            len_v = np.linalg.norm(v)
            if len_v < 1e-12:
                continue
            t = np.clip(np.dot(pos - p1, v) / (len_v ** 2), 0.0, 1.0)
            proj = p1 + t * v
            dist = float(np.linalg.norm(pos - proj))
            if dist < min_dist:
                min_dist = dist
                best_k = k
        return best_k, min_dist

    def perform_edge_shift_drag(self, edge_k: int, disp_vec: np.ndarray, start_corners: np.ndarray) -> np.ndarray:
        k = edge_k
        k_prev = (k - 1) % 6
        k_next = (k + 1) % 6
        k_next2 = (k + 2) % 6

        P = start_corners
        A = P[k]
        B = P[k_next]
        A_prev = P[k_prev]
        B_next = P[k_next2]

        len_A = np.linalg.norm(A - A_prev)
        len_B = np.linalg.norm(B - B_next)
        if len_A < 1e-12 or len_B < 1e-12:
            return P.copy()

        u_A = (A - A_prev) / len_A
        u_B = (B - B_next) / len_B

        n = u_A + u_B
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            return P.copy()
        n = n / norm_n

        proj_uA = np.dot(u_A, n)
        if abs(proj_uA) < 1e-12:
            return P.copy()

        # Displacement along normal
        disp_normal = float(np.dot(disp_vec, n))
        delta = disp_normal / proj_uA

        # Compute candidate endpoints
        A_new = A + delta * u_A
        B_new = B + delta * u_B

        # Constrain: ensure edge length remains at least 10% of original
        orig_edge_vec = B - A
        orig_edge_len2 = float(np.dot(orig_edge_vec, orig_edge_vec))
        diff_u = u_B - u_A
        denom = float(np.dot(diff_u, orig_edge_vec))
        if abs(denom) > 1e-12:
            limit_delta = -0.9 * orig_edge_len2 / denom
            if denom < 0 and delta > limit_delta:
                delta = limit_delta
            elif denom > 0 and delta < limit_delta:
                delta = limit_delta
            A_new = A + delta * u_A
            B_new = B + delta * u_B

        new_P = P.copy()
        new_P[k] = A_new
        new_P[k_next] = B_new
        return new_P

    def mouseDragEvent(self, ev):
        if ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            if ev.isStart():
                pos_pt = ev.buttonDownPos()
                pos = np.array([pos_pt.x(), pos_pt.y()])
                current_corners = np.array([[h['item'].pos().x(), h['item'].pos().y()] for h in self.handles])
                best_k, min_dist = self._find_closest_edge(pos, current_corners)
                self.edge_drag_idx = best_k
                self.edge_drag_start_corners = current_corners.copy()
                self.edge_drag_start_pos = pos
                ev.accept()
                return

            if self.edge_drag_idx is not None and self.edge_drag_start_corners is not None:
                cur_pt = ev.pos()
                cur_pos = np.array([cur_pt.x(), cur_pt.y()])
                disp_vec = cur_pos - self.edge_drag_start_pos
                new_corners = self.perform_edge_shift_drag(self.edge_drag_idx, disp_vec, self.edge_drag_start_corners)

                k = self.edge_drag_idx
                k_next = (k + 1) % 6
                
                pt_k = QtCore.QPointF(new_corners[k, 0], new_corners[k, 1])
                self.handles[k]['item'].setPos(pt_k)
                self.handles[k]['pos'] = pt_k
                
                pt_next = QtCore.QPointF(new_corners[k_next, 0], new_corners[k_next, 1])
                self.handles[k_next]['item'].setPos(pt_next)
                self.handles[k_next]['pos'] = pt_next

                self.prepareGeometryChange()
                self.update()
                self.stateChanged(finish=ev.isFinish())

                if ev.isFinish():
                    self.edge_drag_idx = None
                    self.edge_drag_start_corners = None
                    self.edge_drag_start_pos = None

                ev.accept()
                return

        super().mouseDragEvent(ev)

    def get_hexagon_corners(self) -> List[List[float]]:
        corners = []
        for h in self.handles:
            pt = self.mapToParent(h['item'].pos())
            corners.append([float(pt.x()), float(pt.y())])
        return corners

    def get_hexagon_center(self) -> Tuple[float, float]:
        corners = np.array(self.get_hexagon_corners())
        return float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1]))

    def get_equivalent_diameter_and_area(self) -> Tuple[float, float]:
        corners = np.array(self.get_hexagon_corners())
        x = corners[:, 0]
        y = corners[:, 1]
        area = float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        diameter = float(2.0 * np.sqrt(area / np.pi)) if area > 0 else 0.0
        return diameter, area

    get_corners = get_hexagon_corners
    get_center = get_hexagon_center

    def movePoint(self, handle, pos, modifiers=None, finish=True, coords='parent'):
        idx = self.indexOfHandle(handle)
        if idx < 0 or idx >= len(self.handles):
            return

        if coords == 'scene':
            p1 = self.mapSceneToParent(pos)
        elif isinstance(pos, QtCore.QPointF):
            p1 = pos
        else:
            p1 = QtCore.QPointF(pos[0], pos[1])
            
        local_p1 = self.mapFromParent(p1)
        cur_pos = np.array([local_p1.x(), local_p1.y()])

        if self.drag_start_corners is None or self.active_handle_idx != idx:
            self.drag_start_corners = np.array([[h['item'].pos().x(), h['item'].pos().y()] for h in self.handles])
            self.drag_start_pos = np.array([self.handles[idx]['item'].pos().x(), self.handles[idx]['item'].pos().y()])
            self.active_handle_idx = idx
            self.drag_is_shift = bool(modifiers and (modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier))

        if self.drag_is_shift:
            # Shift+Drag: Extend corner k and adjacent corners k-1, k+1 along second-next edges
            k = idx
            k_prev = (k - 1) % 6
            k_next = (k + 1) % 6
            k_prev2 = (k - 2) % 6
            k_next2 = (k + 2) % 6

            P_start = self.drag_start_corners
            u = P_start[k_prev] - P_start[k_prev2]
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-12:
                return
            u = u / norm_u

            d = cur_pos - self.drag_start_pos
            delta = float(np.dot(d, u))

            len_prev = np.linalg.norm(P_start[k_prev] - P_start[k_prev2])
            len_next = np.linalg.norm(P_start[k_next] - P_start[k_next2])
            min_len = min(len_prev, len_next)
            if delta < -0.85 * min_len:
                delta = -0.85 * min_len

            shift_vec = delta * u
            new_corners = P_start.copy()
            new_corners[k] += shift_vec
            new_corners[k_prev] += shift_vec
            new_corners[k_next] += shift_vec

            for i in range(6):
                pt = QtCore.QPointF(new_corners[i, 0], new_corners[i, 1])
                self.handles[i]['item'].setPos(pt)
                self.handles[i]['pos'] = pt

        else:
            # Normal Drag: Scale and rotate self-similarly around geometric center
            P_start = self.drag_start_corners
            center = np.mean(P_start, axis=0)
            v0 = self.drag_start_pos - center
            v1 = cur_pos - center
            len0 = np.linalg.norm(v0)
            len1 = np.linalg.norm(v1)

            if len0 > 1e-12 and len1 > 1e-12:
                scale = len1 / len0
                ang0 = np.arctan2(v0[1], v0[0])
                ang1 = np.arctan2(v1[1], v1[0])
                d_ang = ang1 - ang0

                R = np.array([
                    [np.cos(d_ang), -np.sin(d_ang)],
                    [np.sin(d_ang), np.cos(d_ang)]
                ])

                for i in range(6):
                    rel = P_start[i] - center
                    new_pt_arr = center + scale * (R @ rel)
                    pt = QtCore.QPointF(new_pt_arr[0], new_pt_arr[1])
                    self.handles[i]['item'].setPos(pt)
                    self.handles[i]['pos'] = pt

        self.prepareGeometryChange()
        self.update()
        self.stateChanged(finish=finish)

        if finish:
            self.drag_start_corners = None
            self.drag_start_pos = None
            self.active_handle_idx = None
            self.drag_is_shift = None


def create_roi_for_particle(
    particle: Particle,
    pixel_size_x: float,
    pixel_size_y: float
) -> Optional[pg.ROI]:
    """
    Factory function creating the appropriate interactive ROI for the particle
    based on its shape_type (HexagonFacetROI, GeneralPolygonROI, or EllipseFacetROI).
    """
    st = getattr(particle, 'shape_type', 'hexagon') or 'hexagon'
    px = float(pixel_size_x)
    py = float(pixel_size_y)
    
    if st == 'ellipse':
        if particle.ellipse_se and len(particle.ellipse_se) >= 5:
            cx, cy, a, b, theta = particle.ellipse_se
            return EllipseFacetROI(cx * px, cy * py, a * px, b * py, angle=float(np.degrees(theta)))
        elif particle.top_facet_diameter:
            r = 0.5 * particle.top_facet_diameter
            return EllipseFacetROI(particle.pixel_x * px, particle.pixel_y * py, r, r, angle=0.0)
        elif particle.ecd:
            r = 0.5 * (particle.ecd * 0.6)
            return EllipseFacetROI(particle.pixel_x * px, particle.pixel_y * py, r, r, angle=0.0)
        return None

    elif st == 'hexagon':
        if particle.facet_polygon and len(particle.facet_polygon) == 6:
            corners_px = np.array(particle.facet_polygon, dtype=float)
            corners_m = [
                [float(corners_px[k, 0] * px), float(corners_px[k, 1] * py)]
                for k in range(6)
            ]
            return HexagonFacetROI(corners=corners_m)
        elif particle.top_facet_diameter:
            r = 0.5 * particle.top_facet_diameter
            return HexagonFacetROI(cx=particle.pixel_x * px, cy=particle.pixel_y * py, radius=r, angle=0.0)
        elif particle.ecd:
            r = 0.5 * (particle.ecd * 0.6)
            return HexagonFacetROI(cx=particle.pixel_x * px, cy=particle.pixel_y * py, radius=r, angle=0.0)
        return None

    else:
        # General polygon (triangle, quadrilateral, pentagon, polygon_6, polygon_8)
        if particle.facet_polygon and len(particle.facet_polygon) >= 3:
            corners_px = np.array(particle.facet_polygon, dtype=float)
            corners_m = [
                [float(corners_px[k, 0] * px), float(corners_px[k, 1] * py)]
                for k in range(len(corners_px))
            ]
            return GeneralPolygonROI(corners=corners_m)
        return None


# Backward-compatibility alias
create_hexagon_roi_for_particle = create_roi_for_particle
