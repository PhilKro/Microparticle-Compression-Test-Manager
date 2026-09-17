import re
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from PySide6.QtGui import QTransform
from PySide6.QtCore import QPointF

from models import Sample, ImageRecord, ImageType, Particle

@dataclass
class ParticleGlobalInfo:
    uid: str                                    # Short format {Site}{ParticleID}, e.g. "61", "183"
    site_filename: str
    site_number: int
    particle_id: int
    particle: Particle
    flat_x_m: float                             # X in meters on sample plane
    flat_y_m: float                             # Y in meters on sample plane
    flat_x_um: float                            # X in micrometers
    flat_y_um: float                            # Y in micrometers
    site_top_left_x_m: float
    site_top_left_y_m: float
    pixel_size_x: float
    pixel_size_y: float
    tested: bool = False
    test_timestamp: Optional[str] = None
    test_notes: str = ""

def extract_site_number(filename: str) -> int:
    match = re.search(r"Site(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def extract_all_particles(sample: Sample) -> Dict[str, ParticleGlobalInfo]:
    """
    Extracts all measured particles across all sites in the sample,
    computing their absolute physical coordinates on the flat sample plane
    using the Region overview image as reference.
    UID format is {Site}_{ParticleID} (e.g. '2_23', '22_3', '6_1').
    """
    result: Dict[str, ParticleGlobalInfo] = {}
    if not sample or not sample.images:
        return result

    region = next((r for r in sample.images if r.classification == ImageType.REGION), None)
    if not region:
        return result

    meta_reg = region.metadata.get('MAIN', {})
    px_reg = float(meta_reg.get('PixelSizeX', 1.0))
    py_reg = float(meta_reg.get('PixelSizeY', 1.0))
    
    # Overview center in physical units
    cx_reg = 2048 * px_reg / 2.0
    cy_reg = 2048 * py_reg / 2.0

    for site in sample.images:
        if site.classification != ImageType.SITE or not site.particles:
            continue

        site_num = extract_site_number(site.filename)
        meta_site = site.metadata.get('MAIN', {})
        px_site = float(meta_site.get('PixelSizeX', 1.0))
        py_site = float(meta_site.get('PixelSizeY', 1.0))
        site_w = 2048 * px_site
        site_h = 2048 * py_site

        dx = site.stage_x - region.stage_x
        dy = site.stage_y - region.stage_y

        off_x = getattr(site, 'offset_x_pixels', 0.0)
        off_y = getattr(site, 'offset_y_pixels', 0.0)

        site_top_left_x = cx_reg - dx - site_w / 2.0 + off_x * px_site
        site_top_left_y = cy_reg + dy - site_h / 2.0 + off_y * py_site

        for p in site.particles:
            part_x_m = site_top_left_x + p.pixel_x * px_site
            part_y_m = site_top_left_y + p.pixel_y * py_site
            uid = f"{site_num}_{p.id}"

            info = ParticleGlobalInfo(
                uid=uid,
                site_filename=site.filename,
                site_number=site_num,
                particle_id=p.id,
                particle=p,
                flat_x_m=part_x_m,
                flat_y_m=part_y_m,
                flat_x_um=part_x_m * 1e6,
                flat_y_um=part_y_m * 1e6,
                site_top_left_x_m=site_top_left_x,
                site_top_left_y_m=site_top_left_y,
                pixel_size_x=px_site,
                pixel_size_y=py_site
            )
            result[uid] = info

    return result

def get_tilt_transform(cx: float, cy: float, tilt_deg: float, rot_deg: float, px: float = 1.0, py: float = 1.0) -> QTransform:
    """
    Constructs a QTransform applying pixel-to-physical scaling, 2D in-plane rotation,
    and 70° tilt foreshortening centered at (cx, cy).
    """
    t = QTransform()
    t.translate(cx, cy)
    # Foreshortening along vertical SEM column axis
    t.scale(1.0, math.cos(math.radians(tilt_deg)))
    t.rotate(rot_deg)
    t.translate(-cx, -cy)
    t.scale(px, py)
    return t

def map_point_tilt(x: float, y: float, cx: float, cy: float, tilt_deg: float, rot_deg: float) -> Tuple[float, float]:
    """
    Maps a single 2D point (x, y) through rotation and tilt foreshortening.
    """
    rad = math.radians(rot_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    cos_tilt = math.cos(math.radians(tilt_deg))

    # Center-relative
    dx = x - cx
    dy = y - cy

    # Rotate
    rx = dx * cos_r - dy * sin_r
    ry = dx * sin_r + dy * cos_r

    # Tilt foreshorten
    tx = rx
    ty = ry * cos_tilt

    return (cx + tx, cy + ty)

def map_polygon_tilt(polygon: List[List[float]], cx: float, cy: float, tilt_deg: float, rot_deg: float) -> List[Tuple[float, float]]:
    """
    Maps polygon vertices through rotation and tilt foreshortening.
    """
    return [map_point_tilt(pt[0], pt[1], cx, cy, tilt_deg, rot_deg) for pt in polygon]

def predict_smaract_translation(
    curr_pos_um: Tuple[float, float],
    next_pos_um: Tuple[float, float],
    rot_deg: float,
    scale: float = 1.0,
    invert_x: bool = False,
    invert_y: bool = False,
    flip_xy: bool = False
) -> Tuple[float, float, float]:
    """
    Calculates SmarAct piezo stage translation in micrometers from current particle to next particle.
    Returns: (delta_x_um, delta_y_um, total_distance_um)
    """
    dx_flat = next_pos_um[0] - curr_pos_um[0]
    dy_flat = next_pos_um[1] - curr_pos_um[1]

    rad = math.radians(rot_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    # In-plane rotation
    dx = scale * (dx_flat * cos_r - dy_flat * sin_r)
    dy = scale * (dx_flat * sin_r + dy_flat * cos_r)

    if flip_xy:
        dx, dy = dy, dx

    if invert_x:
        dx = -dx
    if invert_y:
        dy = -dy

    dist = math.hypot(dx, dy)
    return (dx, dy, dist)

@dataclass
class CalibrationResult:
    rot_deg: float                              # Calibrated rotation angle in degrees [0, 360)
    scale: float                                # Calibrated stage travel scale factor (typically ~1.0)
    rot_std_deg: float                          # Standard error of rotation in degrees (1-sigma)
    scale_std: float                            # Standard error of scale factor (1-sigma)
    inlier_mask: List[bool]                     # True for inlier moves, False for rejected outliers
    inlier_count: int                           # Number of consensus inlier moves
    total_count: int                            # Total moves evaluated
    mean_residual_um: float                     # Mean Euclidean stage residual in micrometers
    confidence_pct: float                       # Convergence confidence score [0, 100]%
    status_text: str                            # Human-readable summary badge text
    scale_x: float = 1.0                        # Diagnostic X-axis scale factor
    scale_y: float = 1.0                        # Diagnostic Y-axis scale factor

def robust_refine_orientation_scale(
    flat_displacements: List[Tuple[float, float]],
    actual_displacements: List[Tuple[float, float]],
    prior_rot_deg: Optional[float] = None,
    prior_scale: Optional[float] = None,
    invert_x: bool = False,
    invert_y: bool = False,
    flip_xy: bool = False,
    sigma_prior_rot_deg: float = 8.0,
    sigma_prior_scale: float = 0.12,
    noise_sigma_um: float = 0.25
) -> CalibrationResult:
    """
    Robust Bayesian Pose and Scale Estimator (RB-PSE).
    Combines:
    1. 1-Hop Minimal RANSAC Outlier Rejection: Identifies consensus moves and prunes blunders/drift.
    2. Bayesian Prior Regularization: When prior_rot_deg is provided, anchors short initial hops (<= 2 um)
       to the visual coarse alignment prior, preventing wild angular swings while smoothly converging.
       When prior is None, performs exact pure least-squares estimation.
    3. Analytical Fisher Information Hessian: Computes exact standard errors (sigma_phi, sigma_scale)
       and a continuous confidence convergence metric.
    4. Anisotropic Scale Diagnostics: Computes directional scales (scale_x, scale_y) for aspect ratio inspection.
    """
    default_rot = 0.0 if prior_rot_deg is None else prior_rot_deg
    default_scale = 1.0 if prior_scale is None else prior_scale

    N = len(flat_displacements)
    if N == 0 or len(actual_displacements) == 0:
        return CalibrationResult(
            rot_deg=default_rot % 360.0,
            scale=default_scale,
            rot_std_deg=sigma_prior_rot_deg,
            scale_std=sigma_prior_scale,
            inlier_mask=[],
            inlier_count=0,
            total_count=0,
            mean_residual_um=0.0,
            confidence_pct=0.0,
            status_text="Prior only (no moves logged yet)",
            scale_x=default_scale,
            scale_y=default_scale
        )

    # Clean actual displacements according to stage polarity settings before fitting rotation
    adj_actual: List[Tuple[float, float]] = []
    for ax, ay in actual_displacements:
        x, y = ax, ay
        if invert_x: x = -x
        if invert_y: y = -y
        if flip_xy: x, y = y, x
        adj_actual.append((x, y))

    P = np.array(flat_displacements, dtype=float)   # (N, 2)
    S = np.array(adj_actual, dtype=float)           # (N, 2)
    lengths = np.linalg.norm(P, axis=1)

    # 1. 1-Hop Minimal RANSAC Outlier Rejection if N >= 3
    inlier_mask = [True] * N
    if N >= 3:
        best_inliers = list(range(N))
        best_score = -1e9
        res_threshold = max(0.65, 2.5 * noise_sigma_um)

        for k in range(N):
            pk = P[k]
            sk = S[k]
            lk = lengths[k]
            if lk < 0.5:
                continue

            # Single-hop minimal hypothesis
            phi_k = math.atan2(sk[1], sk[0]) - math.atan2(pk[1], pk[0])
            s_k = float(np.linalg.norm(sk) / lk)
            if not (0.5 <= s_k <= 2.0):
                continue

            R_k = np.array([[math.cos(phi_k), -math.sin(phi_k)], [math.sin(phi_k), math.cos(phi_k)]])
            preds_k = s_k * (P @ R_k.T)
            res_k = np.linalg.norm(S - preds_k, axis=1)

            inliers = [idx for idx, r in enumerate(res_k) if r <= res_threshold]
            score = len(inliers) - 0.05 * float(np.sum(res_k[inliers]))
            if score > best_score:
                best_score = score
                best_inliers = inliers

        if len(best_inliers) >= 2:
            inlier_mask = [i in best_inliers for i in range(N)]
        else:
            inlier_mask = [True] * N

    # 2. Bayesian MAP Estimation using inliers
    inlier_indices = [i for i, inl in enumerate(inlier_mask) if inl]
    P_in = P[inlier_indices]
    S_in = S[inlier_indices]

    # Cross-covariance matrix
    H = S_in.T @ P_in

    if prior_rot_deg is not None:
        prior_rad = math.radians(prior_rot_deg)
        sigma_prior_rad = math.radians(sigma_prior_rot_deg)
        lambda_phi = (noise_sigma_um / max(1e-4, sigma_prior_rad)) ** 2
        R_prior = np.array([
            [math.cos(prior_rad), -math.sin(prior_rad)],
            [math.sin(prior_rad), math.cos(prior_rad)]
        ])
        H_reg = H + lambda_phi * R_prior
    else:
        lambda_phi = 0.0
        H_reg = H

    phi_est = math.atan2(H_reg[1, 0] - H_reg[0, 1], H_reg[0, 0] + H_reg[1, 1])
    rot_deg = float(math.degrees(phi_est) % 360.0)

    R_est = np.array([
        [math.cos(phi_est), -math.sin(phi_est)],
        [math.sin(phi_est), math.cos(phi_est)]
    ])

    sum_dot = float(np.sum(S_in * (P_in @ R_est.T)))
    sum_flat_sq = float(np.sum(P_in**2))

    if prior_scale is not None:
        lambda_scale = (noise_sigma_um / max(1e-4, sigma_prior_scale)) ** 2
        scale_est = float((sum_dot + lambda_scale * prior_scale) / (sum_flat_sq + lambda_scale))
    else:
        lambda_scale = 0.0
        scale_est = float(sum_dot / sum_flat_sq) if sum_flat_sq > 1e-12 else 1.0

    scale_est = float(np.clip(scale_est, 0.5, 2.0))

    # 3. Residuals and Analytical Uncertainty (Fisher Information)
    preds_in = scale_est * (P_in @ R_est.T)
    residuals = np.linalg.norm(S_in - preds_in, axis=1)
    mean_res = float(np.mean(residuals)) if len(residuals) > 0 else 0.0

    sample_var = float(np.mean(residuals**2)) if len(residuals) > 0 else noise_sigma_um**2
    effective_sigma_sq = max(sample_var, (0.5 * noise_sigma_um)**2)

    var_phi_rad = effective_sigma_sq / (scale_est**2 * sum_flat_sq + lambda_phi) if (scale_est**2 * sum_flat_sq + lambda_phi) > 1e-12 else 1.0
    rot_std_deg = float(math.degrees(math.sqrt(var_phi_rad)))

    var_scale = effective_sigma_sq / (sum_flat_sq + lambda_scale) if (sum_flat_sq + lambda_scale) > 1e-12 else 1.0
    scale_std = float(math.sqrt(var_scale))

    # Continuous confidence percentage
    conf_rot = max(0.0, min(1.0, 0.25 / max(0.05, rot_std_deg)))
    conf_scale = max(0.0, min(1.0, 0.01 / max(0.002, scale_std)))
    confidence_pct = float(round((0.7 * conf_rot + 0.3 * conf_scale) * 100.0, 1))

    # 4. Anisotropic Aspect Ratio Diagnostics
    P_rot = P_in @ R_est.T
    denom_x = float(np.sum(np.abs(P_rot[:, 0])))
    denom_y = float(np.sum(np.abs(P_rot[:, 1])))
    scale_x = float(np.sum(np.abs(S_in[:, 0])) / denom_x) if denom_x > 1e-4 else scale_est
    scale_y = float(np.sum(np.abs(S_in[:, 1])) / denom_y) if denom_y > 1e-4 else scale_est
    scale_x = float(np.clip(scale_x, 0.5, 2.0))
    scale_y = float(np.clip(scale_y, 0.5, 2.0))

    # 5. Status Text
    in_count = len(inlier_indices)
    tot_count = N
    if in_count < 2:
        status_text = f"Initial (1 move): φ={rot_deg:.1f}°±{rot_std_deg:.1f}°, Scale={scale_est:.3f}±{scale_std:.3f}"
    elif rot_std_deg <= 0.35 and scale_std <= 0.015:
        status_text = f"Converged ({in_count}/{tot_count} inliers): φ={rot_deg:.2f}°±{rot_std_deg:.2f}°, Scale={scale_est:.3f}±{scale_std:.3f} (Conf: {confidence_pct:.0f}%)"
    else:
        status_text = f"Refining ({in_count}/{tot_count} inliers): φ={rot_deg:.2f}°±{rot_std_deg:.2f}°, Scale={scale_est:.3f}±{scale_std:.3f} (Conf: {confidence_pct:.0f}%)"

    return CalibrationResult(
        rot_deg=rot_deg,
        scale=scale_est,
        rot_std_deg=rot_std_deg,
        scale_std=scale_std,
        inlier_mask=inlier_mask,
        inlier_count=in_count,
        total_count=tot_count,
        mean_residual_um=mean_res,
        confidence_pct=confidence_pct,
        status_text=status_text,
        scale_x=scale_x,
        scale_y=scale_y
    )

def refine_orientation_scale(
    flat_displacements: List[Tuple[float, float]],
    actual_displacements: List[Tuple[float, float]],
    invert_x: bool = False,
    invert_y: bool = False,
    flip_xy: bool = False,
    prior_rot_deg: Optional[float] = None,
    prior_scale: Optional[float] = None
) -> Tuple[float, float, float]:
    """
    Convenience wrapper returning (rot_deg, scale, mean_residual_um)
    for backward compatibility with existing callers.
    """
    cal = robust_refine_orientation_scale(
        flat_displacements,
        actual_displacements,
        prior_rot_deg=prior_rot_deg,
        prior_scale=prior_scale,
        invert_x=invert_x,
        invert_y=invert_y,
        flip_xy=flip_xy
    )
    return cal.rot_deg, cal.scale, cal.mean_residual_um

def optimize_particle_route(
    start_uid: str,
    untested_uids: List[str],
    particles: Dict[str, ParticleGlobalInfo]
) -> Tuple[List[str], float]:
    """
    Computes an optimized shortest travel path sequence starting from start_uid
    through all remaining untested particles.
    Combines Greedy Nearest Neighbor with 2-Opt local search refinement (executes in < 8 ms).
    Returns: (ordered_uids, total_distance_um)
    """
    if not start_uid or start_uid not in particles:
        valid_uids = [u for u in untested_uids if u in particles]
        if not valid_uids:
            return [], 0.0
        start_uid = valid_uids[0]

    unvisited = [u for u in untested_uids if u != start_uid and u in particles]
    if not unvisited:
        return [start_uid], 0.0

    # 1. Greedy Nearest Neighbor
    route = [start_uid]
    remaining = set(unvisited)
    curr = start_uid

    while remaining:
        curr_pos = (particles[curr].flat_x_um, particles[curr].flat_y_um)
        next_uid = min(
            remaining,
            key=lambda u: math.hypot(particles[u].flat_x_um - curr_pos[0], particles[u].flat_y_um - curr_pos[1])
        )
        route.append(next_uid)
        remaining.remove(next_uid)
        curr = next_uid

    # 2. 2-Opt Local Search (keep route[0] fixed as start particle)
    N = len(route)
    if N >= 4:
        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            for i in range(1, N - 1):
                for j in range(i + 1, N):
                    p_im1 = particles[route[i - 1]]
                    p_i = particles[route[i]]
                    p_j = particles[route[j]]
                    p_jp1 = particles[route[j + 1]] if j + 1 < N else None

                    d_old = math.hypot(p_i.flat_x_um - p_im1.flat_x_um, p_i.flat_y_um - p_im1.flat_y_um)
                    if p_jp1:
                        d_old += math.hypot(p_jp1.flat_x_um - p_j.flat_x_um, p_jp1.flat_y_um - p_j.flat_y_um)

                    d_new = math.hypot(p_j.flat_x_um - p_im1.flat_x_um, p_j.flat_y_um - p_im1.flat_y_um)
                    if p_jp1:
                        d_new += math.hypot(p_jp1.flat_x_um - p_i.flat_x_um, p_jp1.flat_y_um - p_i.flat_y_um)

                    if d_new < d_old - 1e-5:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        break
                if improved:
                    break

    # Calculate total travel distance
    total_dist = 0.0
    for k in range(len(route) - 1):
        pk = particles[route[k]]
        pkp1 = particles[route[k + 1]]
        total_dist += math.hypot(pkp1.flat_x_um - pk.flat_x_um, pkp1.flat_y_um - pk.flat_y_um)

    return route, total_dist

