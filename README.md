# Microparticle Compression Test Manager (v1.0)

A high-precision desktop application for dual-signal (BSE + SE) scanning electron microscopy (SEM) particle characterization, crystallographic facet fitting, and in-situ micro-compression stage navigation.

---

## Project Structure

```text
Microparticle-Compression-Test-Manager/
├── main.py                     # Application entry point & session controller
├── ui.py                       # Qt layout, visual components & custom widgets
├── models.py                   # Pydantic data schemas & session models
├── measurement.py              # Particle boundary & facet fitting algorithms
├── compression_planner.py      # RB-PSE Bayesian pose/scale solver & TSP route optimizer
├── roi.py                      # Interactive PyQtGraph ROI classes (120° locked hexagons & polygons)
├── alignment.py                # Multi-scale FFT cross-correlation auto-alignment
├── help_content.json           # In-app help dialog contents (F1)
├── requirements.txt            # Python package dependencies
├── README.md                   # Documentation & setup guide
├── .gitignore                  # Git exclusion rules
└── tests/                      # Automated test suite
    ├── __init__.py
    ├── test_complete_integration.py
    ├── test_compression_tab_complete.py
    ├── test_edge_shift_drag_complete.py
    ├── test_help_dialog.py
    ├── test_robust_calibration.py
    └── fixtures/
        └── sample_test_data.json
```

> **Note on Datasets:** Experimental micrographs and local test folders are intentionally excluded from version control. Micrograph datasets should be loaded directly in the UI using **File -> Open Sample Folder** (`Ctrl+O`).

---

## Features

### 1. Region Overview & Automated Alignment (Tab 1)
- **Multi-Scale FFT Cross-Correlation:** Automatically co-registers high-magnification site micrographs onto the macro region overview.
- **Interactive Manual Fine-Tuning:** Click-and-drag site bounding boxes directly on the canvas with full undo (`Ctrl+Z`) support.
- **Dual-Signal Inspection:** Instant switching between Secondary Electron (SE) and Backscattered Electron (BSE) modes.

### 2. Site Analysis & Crystallographic Facet Measurement (Tab 2)
- **Sequential Site Browsing:** One-click `Next Site ➔` navigation cycles through all sites in the session.
- **Dual-Signal Particle Fitting:** Fits outer boundary ellipses on BSE to determine Equivalent Circular Diameter (ECD), then fits surface facets on SE.
- **7 Facet Geometry Modes (Slider 0–6):**
  - `0: Hexagon (Cryst 6-gon)`: Equiangular crystallographic hexagon with strictly locked 120.0° internal angles. Supports self-similar corner drag and inward/outward edge shift-drag.
  - `1: Triangle (3-gon)`: For octahedral and tetrahedral {111} facets.
  - `2: Quadrilateral (4-gon)`: For cubic {100}, tetragonal, or trapezoidal facets.
  - `3: Pentagon (5-gon)`: For decahedral twin nanoparticles and pyritohedral {210} facets.
  - `4: General 6-gon`: 6-sided polygon with freely adjustable vertex angles.
  - `5: General 8-gon`: 8-sided polygon for truncated cuboctahedra.
  - `6: Ellipse / Circle`: Curvilinear facet model with independent major/minor semi-axes.
- **Interactive Shift-Drag Editing:** Fine-adjust individual facet edges while maintaining strict crystallographic constraints.

### 3. In-Situ Micro-Compression Testing (Tab 3)
Designed for in-situ mechanical micro-compression testing on SEM stages (e.g., SmarAct piezo positioners) with 70° tilted sample mounts and arbitrary rotation:
- **Two-Mode Experimental Workflow:**
  - **Mode A (Ready for Test):** Shows large Current Particle UID badge, test queue status, and target confirmation on the SE site image.
  - **Mode B (Stage Translation Guidance):** Activated on `Enter`. Shows Target Particle UID, target SE site preview with marked particle, exact SmarAct translation vector ($\Delta X, \Delta Y$ in µm), and actual travel sliders with reset.
- **Shortest TSP Tour Optimization:** Automatically computes the minimum translation path (Euclidean Nearest Neighbor heuristic) across all remaining particles when setting a current particle.
- **Robust Bayesian Calibration (RB-PSE):** 1-hop RANSAC outlier rejection with closed-form uncertainty estimation ($\sigma_\phi$, $\sigma_s$) that adapts future stage travel vectors based on operator inputs.
- **Stage Scale Factor ($s$):** Interactive spinbox and automatic adaptation for piezo travel scaling ($\mu\text{m}/\mu\text{m}$).
- **Axis Transformations:** One-click `Flip X/Y`, `Invert X`, and `Invert Y` for piezo motor wiring configurations.
- **Compact & Unique Particle UIDs:** Unambiguous IDs formatted as `{Site}_{ParticleID}` (e.g., `2_23`, `22_3`, `6_1`).
- **Comprehensive Session Logging:** Every test is logged to `compression_test_YYMMDDHH.json` with timestamps, particle geometry (ECD, facet diameter, area), SmarAct coordinates, and calibration state.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` or `M` | Measure particle (BSE outer + SE facet) |
| `Enter` | Toggle Mode A / Mode B, accept stage move, and log test |
| `Delete` | Remove selected particle |
| `Ctrl + S` | Save session metadata (`sample_data.json`) |
| `Ctrl + O` | Open sample directory |
| `Ctrl + Z` | Undo site alignment drag |
| `F1` | Open in-app Help documentation |
| `Ctrl + Q` | Exit application |

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Recommended environment: virtualenv or conda

### Dependencies
```bash
pip install -r requirements.txt
```

*(Optional: PyTorch + Segment Anything for neural segmentation features if enabled)*

### Running the Application
```bash
python main.py
```

---

## Running Automated Tests

All tests are located in `tests/` and can be executed via:

```bash
# Run individual test suites:
python tests/test_complete_integration.py
python tests/test_compression_tab_complete.py
python tests/test_robust_calibration.py
python tests/test_edge_shift_drag_complete.py
python tests/test_help_dialog.py
```

---

## License
MIT License.
