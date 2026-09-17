from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ImageType(str, Enum):
    OVERVIEW = "Overview"
    REGION = "Region"
    REGION_POST = "Region Post"
    SITE = "Site"
    UNCLASSIFIED = "Unclassified"

class Particle(BaseModel):
    id: int
    pixel_x: float
    pixel_y: float
    ecd: Optional[float] = None                 # in meters
    top_facet_diameter: Optional[float] = None  # in meters
    top_facet_area: Optional[float] = None      # in square meters
    ellipse_bse: Optional[List[float]] = None   # [cx, cy, a, b, theta] in pixels
    ellipse_se: Optional[List[float]] = None    # [cx, cy, a, b, theta] in pixels
    facet_polygon: Optional[List[List[float]]] = None # [[x1, y1], [x2, y2], ...] in pixels
    shape_type: str = "hexagon"                  # "hexagon", "triangle", "quadrilateral", "pentagon", "polygon_6", "polygon_8", "ellipse"
    is_manually_fitted: bool = False             # Flag indicating if facet was manually fine-fitted
    facet_parameters: Optional[Dict[str, Any]] = None # Full crystallographic & morphological feature dict

class ImageRecord(BaseModel):
    filename: str
    classification: ImageType = ImageType.UNCLASSIFIED
    acquisition_time: Optional[datetime] = None
    stage_x: float = 0.0
    stage_y: float = 0.0
    offset_x_pixels: float = 0.0
    offset_y_pixels: float = 0.0
    use_refined_alignment: bool = True
    alignment_status: str = "Pending"
    metadata: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    particles: List[Particle] = Field(default_factory=list)

class Sample(BaseModel):
    name: str
    directory_path: str
    images: List[ImageRecord] = Field(default_factory=list)
    hysteresis_error_x: float = 0.0
    hysteresis_error_y: float = 0.0

class CompressionTestRecord(BaseModel):
    test_index: int
    particle_uid: str                           # Short unique format {Site}{ParticleID}, e.g. "61", "183"
    site_filename: str
    site_number: int
    particle_id: int
    timestamp: str                              # ISO format timestamp
    flat_x_um: float                            # Flat coordinate X in µm
    flat_y_um: float                            # Flat coordinate Y in µm
    particle_properties: Dict[str, Any] = Field(default_factory=dict) # ECD, facet diameter, area, shape type, etc.
    from_particle_uid: Optional[str] = None     # Previous particle UID
    smaract_predicted_dx_um: Optional[float] = None
    smaract_predicted_dy_um: Optional[float] = None
    smaract_actual_dx_um: Optional[float] = None
    smaract_actual_dy_um: Optional[float] = None
    smaract_actual_distance_um: Optional[float] = None
    stage_tilt_deg: float = 70.0
    stage_rotation_deg: float = 0.0
    stage_scale_factor: float = 1.0
    stage_axis_polarity: Dict[str, bool] = Field(default_factory=dict) # invert_x, invert_y, flip_xy
    cumulative_stage_dx_um: float = 0.0         # Accumulated stage travel X from origin
    cumulative_stage_dy_um: float = 0.0         # Accumulated stage travel Y from origin
    calibration_snapshot: Dict[str, Any] = Field(default_factory=dict) # Active phi, scale, uncertainties, inliers
    notes: str = ""

class CompressionSession(BaseModel):
    session_id: str                             # e.g. "compression_test_26091709"
    sample_name: str
    sample_directory: str
    created_at: str
    last_updated_at: Optional[str] = None
    tilt_angle_deg: float = 70.0
    rotation_angle_deg: float = 0.0
    scale_factor: float = 1.0
    total_particles_tested: int = 0
    total_travel_distance_um: float = 0.0
    calibration_summary: Dict[str, Any] = Field(default_factory=dict)
    stage_settings: Dict[str, Any] = Field(default_factory=dict)
    tests: List[CompressionTestRecord] = Field(default_factory=list)


