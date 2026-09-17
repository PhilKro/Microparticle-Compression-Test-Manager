import cv2
import numpy as np
import scipy.ndimage as ndimage
from skimage.measure import EllipseModel, ransac
from typing import Optional, Dict, Any, Tuple, List

def fit_particle_ransac(
    img: np.ndarray,
    cx: float,
    cy: float,
    pixel_size: float,
    search_radius: int = 300,
    sigma: float = 2.0,
    residual_threshold: float = 4.0,
    n_rays: int = 180
) -> Optional[Dict[str, Any]]:
    """
    Measures a particle contour around (cx, cy) using radial profiles,
    gradient edge detection, and RANSAC ellipse fitting.
    
    Returns:
        dict with keys:
            cx, cy: fitted center (pixels)
            a, b: semi-major and semi-minor axis lengths (pixels)
            theta: orientation angle (radians)
            diameter_px: equivalent circular diameter (pixels)
            diameter_m: equivalent circular diameter (meters)
            area_m2: surface area (square meters)
            ellipse_params: [cx, cy, a, b, theta]
            inlier_ratio: float
    """
    if img.ndim == 3:
        img = img[..., 0]
        
    H, W = img.shape
    # Bound search radius within image
    max_possible_r = min(cx, cy, W - 1 - cx, H - 1 - cy)
    if max_possible_r < 15:
        return None
        
    r_max = int(min(search_radius, max_possible_r))
    if r_max < 15:
        return None
        
    smoothed_img = ndimage.gaussian_filter(img.astype(float), sigma)
    
    angles = np.deg2rad(np.linspace(0, 360, n_rays, endpoint=False))
    r_vals = np.arange(8, r_max)
    
    angle_grid, r_grid = np.meshgrid(angles, r_vals, indexing='ij')
    x_rays = cx + r_grid * np.cos(angle_grid)
    y_rays = cy + r_grid * np.sin(angle_grid)
    
    profiles = ndimage.map_coordinates(smoothed_img, [y_rays, x_rays], order=1, mode='nearest')
    
    # Outer edge: intensity drops moving outward from bright particle to dark background
    grads = -np.diff(profiles, axis=1)
    
    boundary_points = []
    for i, angle in enumerate(angles):
        ray_grad = grads[i]
        if len(ray_grad) > 0:
            idx = np.argmax(ray_grad)
            if ray_grad[idx] > 0: # Valid edge transition
                best_r = r_vals[idx] + 0.5
                bx = cx + best_r * np.cos(angle)
                by = cy + best_r * np.sin(angle)
                boundary_points.append((bx, by))
                
    boundary_points = np.array(boundary_points)
    if len(boundary_points) < 8:
        return None
        
    try:
        model, inliers = ransac(
            boundary_points,
            EllipseModel,
            min_samples=5,
            residual_threshold=residual_threshold,
            max_trials=250
        )
        if model is None:
            return None
            
        try:
            exc, eyc = model.center
            a, b = model.axis_lengths
            theta = model.theta
        except AttributeError:
            exc, eyc, a, b, theta = model.params
            
        # Ensure a >= b > 0
        if a <= 0 or b <= 0 or np.isnan(a) or np.isnan(b):
            return None
            
        ecd_px = float(2.0 * np.sqrt(a * b))
        ecd_m = float(ecd_px * pixel_size)
        area_m2 = float(np.pi * a * b * (pixel_size ** 2))
        
        inlier_ratio = float(np.sum(inliers) / len(inliers)) if inliers is not None else 1.0
        
        return {
            'cx': float(exc),
            'cy': float(eyc),
            'a': float(a),
            'b': float(b),
            'theta': float(theta),
            'diameter_px': ecd_px,
            'diameter_m': ecd_m,
            'area_m2': area_m2,
            'ellipse_params': [float(exc), float(eyc), float(a), float(b), float(theta)],
            'inlier_ratio': inlier_ratio
        }
    except Exception as e:
        print(f"RANSAC fitting exception: {e}")
        return None

def get_ellipse_points(
    cx: float,
    cy: float,
    a: float,
    b: float,
    theta: float,
    pixel_size_x: float = 1.0,
    pixel_size_y: float = 1.0,
    n_pts: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates (x, y) coordinates for drawing an ellipse curve in physical units.
    """
    t = np.linspace(0, 2 * np.pi, n_pts)
    # Ellipse in local coordinates
    x_local = a * np.cos(t)
    y_local = b * np.sin(t)
    
    # Rotate by theta
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = x_local * cos_t - y_local * sin_t
    y_rot = x_local * sin_t + y_local * cos_t
    
    # Translate and scale to physical coordinates
    x_phys = (cx + x_rot) * pixel_size_x
    y_phys = (cy + y_rot) * pixel_size_y
    
    return x_phys, y_phys


def get_polygon_points(
    corners: List[List[float]],
    pixel_size_x: float = 1.0,
    pixel_size_y: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts list of polygon [x, y] corners to closed (xp, yp) coordinate arrays
    scaled to physical units for display.
    """
    if not corners:
        return np.array([]), np.array([])
    pts = np.array(corners, dtype=float)
    # Close the polygon loop
    pts_closed = np.vstack([pts, pts[0]])
    return pts_closed[:, 0] * pixel_size_x, pts_closed[:, 1] * pixel_size_y


def fit_facet_polygon_gradient_ransac(
    se_img: np.ndarray,
    cx: float,
    cy: float,
    pixel_size: float,
    r_outer: Optional[float] = None,
    n_sides: int = 6,
    sigma: float = 1.5,
    n_rays: int = 180,
    residual_threshold: float = 2.8,
    regular_polygon: bool = True,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Modular standalone function to measure a polygonal crystal top facet on an SE image
    using directional radial gradient inflection peak detection and RANSAC facet fitting.
    
    This replaces wide Difference-of-Gaussians (DoG) filtering to eliminate the spatial
    asymmetry caused by directional SE detector illumination (where one flank is intensely bright
    and the opposite flank is shadowed). Along each radial ray, the absolute radial directional
    derivative |dI/dr| peaks cleanly at the facet inflection edge regardless of local brightness.
    
    Pipeline:
    1. Constrain search corridor using outer particle radius r_outer (from BSE / outer RANSAC).
    2. Compute directional radial gradient |dI/dr| along radial rays.
    3. Extract ray peak inflection points at the top facet perimeter.
    4. Apply RANSAC outlier rejection to isolate points adhering to the true facet contour.
    5. Fit robust straight facet faces using crystallographic normal constraints (or free lines),
       and intersect adjacent face lines to construct the n_sides crystal polygon.
       
    Args:
        se_img: 2D numpy array (grayscale SE image)
        cx, cy: Facet center coordinates in pixels (e.g. from outer particle RANSAC)
        pixel_size: Pixel size in meters
        r_outer: Outer particle radius in pixels (if None, estimated locally)
        n_sides: Number of polygon faces (default 6 for hexagonal crystals)
        sigma: Gaussian smoothing sigma prior to gradient computation (default 1.5)
        n_rays: Number of radial rays cast around the center (default 180)
        residual_threshold: RANSAC distance threshold in pixels (default 2.8)
        regular_polygon: If True, constrains face normals to crystallographic symmetry angles
                         (phi + k*2*pi/n_sides) with independent face distances d_k (default True)
        
    Returns:
        Dict with keys:
            corners: List of [x, y] vertex coordinates in pixels
            diameter_px: Equivalent circular diameter (pixels)
            diameter_m: Equivalent circular diameter (meters)
            area_px2: Facet surface area (square pixels)
            area_m2: Facet surface area (square meters)
            inlier_ratio: Fraction of radial rays confirmed as inliers
            lines: List of line tuples (nx, ny, c) representing nx*x + ny*y + c = 0
            n_sides: Number of polygon sides
            r_outer: Outer radius used
        or None if measurement failed.
    """
    if se_img.ndim == 3:
        se_img = se_img[..., 0]
        
    H, W = se_img.shape[:2]
    if not (0 <= cx < W and 0 <= cy < H):
        return None
        
    # Estimate r_outer if not provided
    if r_outer is None or r_outer <= 15.0:
        max_r_poss = min(cx, cy, W - 1 - cx, H - 1 - cy)
        r_outer = min(150.0, max_r_poss)
        
    # Constrain search corridor inside the particle
    # The top facet is strictly located well within the particle perimeter (typically 35%-55% r_outer)
    r_max = int(min(0.65 * r_outer, cx, cy, W - 1 - cx, H - 1 - cy))
    r_min = max(6, int(0.20 * r_outer))
    if r_max <= r_min + 4:
        return None
        
    # 1. Compute Directional Radial Gradient |dI/dr|
    norm = cv2.normalize(se_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    f = norm.astype(np.float32)
    smooth = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    
    # 2. Extract candidate contour inflection points along radial rays
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    num_steps = max(25, int(r_max - r_min) * 2)
    r_vals = np.linspace(r_min, r_max, num_steps)
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)
    
    cand_pts = []
    for i in range(n_rays):
        xs = cx + r_vals * cos_t[i]
        ys = cy + r_vals * sin_t[i]
        sgx = cv2.remap(gx, xs.astype(np.float32).reshape(-1, 1), ys.astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).flatten()
        sgy = cv2.remap(gy, xs.astype(np.float32).reshape(-1, 1), ys.astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).flatten()
        # Radial directional derivative |grad . r_hat|
        dI_dr = np.abs(sgx * cos_t[i] + sgy * sin_t[i])
        max_idx = np.argmax(dI_dr)
        best_r = r_vals[max_idx]
        cand_pts.append([cx + best_r * cos_t[i], cy + best_r * sin_t[i]])
        
    cand_pts = np.array(cand_pts, dtype=np.float32)
    
    # 3. RANSAC outlier rejection
    try:
        model, inliers = ransac(
            cand_pts,
            EllipseModel,
            min_samples=8,
            residual_threshold=residual_threshold,
            max_trials=250
        )
    except Exception:
        model, inliers = None, None
        
    if inliers is None:
        radii = np.linalg.norm(cand_pts - [cx, cy], axis=1)
        med = np.median(radii)
        mad = np.median(np.abs(radii - med))
        inliers = np.abs(radii - med) < 2.5 * (mad if mad > 1.0 else 2.0)
        
    inlier_pts = cand_pts[inliers]
    inlier_angles = angles[inliers]
    if len(inlier_pts) < n_sides * 3:
        return None
        
    inlier_ratio = float(np.sum(inliers) / len(inliers))
    
    # 4. Group inliers by face angle and fit crystal polygon
    sector_w = 2.0 * np.pi / n_sides
    best_lines = []
    min_res = float('inf')
    
    if regular_polygon:
        # Physics-informed crystallographic regular normal constraint:
        # Crystal faces have strict angular spacing (e.g. 60 deg for hexagonal crystals).
        # Normal vectors are constrained to phi + k*(2*pi/n_sides), while each face distance d_k
        # is independently and robustly fitted via median projection.
        for trial_phi in np.linspace(0, sector_w, 72, endpoint=False):
            lines = []
            total_res = 0.0
            for k in range(n_sides):
                mid_angle = trial_phi + k * sector_w
                nx = float(np.cos(mid_angle))
                ny = float(np.sin(mid_angle))
                
                rel_angles = (inlier_angles - mid_angle + np.pi) % (2.0 * np.pi) - np.pi
                side_mask = np.abs(rel_angles) < (0.35 * sector_w)
                side_pts = inlier_pts[side_mask]
                if len(side_pts) < 3:
                    total_res += 1e5
                    continue
                
                proj = (side_pts[:, 0] - cx) * nx + (side_pts[:, 1] - cy) * ny
                d_k = float(np.median(proj))
                c = float(-(nx * cx + ny * cy + d_k))
                total_res += float(np.sum(np.abs(proj - d_k)))
                lines.append((nx, ny, c))
                
            if len(lines) == n_sides and total_res < min_res:
                min_res = total_res
                best_lines = lines
    else:
        # Free orientation line fitting via cv2.fitLine
        for trial_phi in np.linspace(0, sector_w, 48, endpoint=False):
            lines = []
            total_res = 0.0
            for k in range(n_sides):
                mid_angle = trial_phi + k * sector_w
                rel_angles = (inlier_angles - mid_angle + np.pi) % (2.0 * np.pi) - np.pi
                side_mask = np.abs(rel_angles) < (0.28 * sector_w)
                side_pts = inlier_pts[side_mask]
                if len(side_pts) < 4:
                    total_res += 1e5
                    continue
                line = cv2.fitLine(side_pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                vx, vy, x0, y0 = line
                nx, ny = -vy, vx
                c = -(nx * x0 + ny * y0)
                norm_fact = np.hypot(nx, ny)
                nx, ny, c = nx / norm_fact, ny / norm_fact, c / norm_fact
                dists = np.abs(nx * side_pts[:, 0] + ny * side_pts[:, 1] + c)
                total_res += np.sum(dists)
                lines.append((float(nx), float(ny), float(c)))
                
            if len(lines) == n_sides and total_res < min_res:
                min_res = total_res
                best_lines = lines
                
    if len(best_lines) != n_sides:
        return None
        
    # 5. Intersect adjacent face lines to compute corner vertices
    corners = []
    for k in range(n_sides):
        nx1, ny1, c1 = best_lines[k]
        nx2, ny2, c2 = best_lines[(k + 1) % n_sides]
        A = np.array([[nx1, ny1], [nx2, ny2]])
        b = np.array([-c1, -c2])
        if abs(np.linalg.det(A)) < 1e-5:
            return None
        pt = np.linalg.solve(A, b)
        corners.append([float(pt[0]), float(pt[1])])
        
    corners_np = np.array(corners, dtype=np.float32)
    area_px2 = float(cv2.contourArea(corners_np))
    if area_px2 <= 0:
        return None
        
    ecd_px = float(2.0 * np.sqrt(area_px2 / np.pi))
    ecd_m = float(ecd_px * pixel_size)
    area_m2 = float(area_px2 * (pixel_size ** 2))
    
    return {
        'corners': corners,
        'diameter_px': ecd_px,
        'diameter_m': ecd_m,
        'area_px2': area_px2,
        'area_m2': area_m2,
        'inlier_ratio': inlier_ratio,
        'lines': best_lines,
        'n_sides': n_sides,
        'r_outer': float(r_outer)
    }


# Backward-compatibility alias so existing scripts and callers seamlessly use the new gradient method
fit_facet_polygon_dog_ransac = fit_facet_polygon_gradient_ransac


def fit_facet_ellipse_ransac(
    se_img: np.ndarray,
    cx: float,
    cy: float,
    pixel_size: float,
    r_outer: Optional[float] = None,
    sigma: float = 1.5,
    residual_threshold: float = 3.0
) -> Optional[Dict[str, Any]]:
    """
    Fits an elliptical / circular facet on the inner SE image constrained by r_outer.
    Uses radial inflection and edge detection with RANSAC ellipse fitting.
    """
    if se_img.ndim == 3:
        se_img = se_img[..., 0]
        
    H, W = se_img.shape
    if r_outer is None:
        r_outer = min(cx, cy, W - 1 - cx, H - 1 - cy) * 0.8
        
    r_max = int(min(r_outer * 0.85, cx, cy, W - 1 - cx, H - 1 - cy))
    if r_max < 10:
        return None
        
    res = fit_particle_ransac(
        se_img, cx, cy, pixel_size,
        search_radius=r_max,
        sigma=sigma,
        residual_threshold=residual_threshold
    )
    if not res:
        return None
        
    return {
        'cx': res['cx'],
        'cy': res['cy'],
        'a': res['a'],
        'b': res['b'],
        'theta': res['theta'],
        'diameter_px': res['diameter_px'],
        'diameter_m': res['diameter_m'],
        'area_m2': res['area_m2'],
        'ellipse_params': res['ellipse_params'],
        'inlier_ratio': res['inlier_ratio']
    }
