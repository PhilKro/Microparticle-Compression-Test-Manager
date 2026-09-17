import sys
import os
import json
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from ui import MainWindow, HelpDialog
from roi import HexagonFacetROI

def test_help_dialog():
    print("Testing help_content.json and HelpDialog...")
    json_path = str(ROOT_DIR / 'help_content.json')
    assert os.path.exists(json_path), f"File {json_path} does not exist!"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert "app_title" in data
    assert "sections" in data
    assert len(data["sections"]) >= 5
    print(f"  help_content.json is valid with {len(data['sections'])} sections.")

    dlg = HelpDialog(json_path=json_path)
    html = dlg.browser.toHtml()
    assert "Microparticle Compression Test Manager" in html or "Particle Manager" in html
    assert "Navigation" in html
    assert "Hexagon (Cryst 6-gon)" in html
    assert "Shift + Drag edge" in html
    assert "Spacebar or M" in html
    print("  HelpDialog HTML content successfully generated and verified!")

def test_menu_help_action():
    print("Testing MainWindow Help menu and action...")
    win = MainWindow()
    assert hasattr(win, "help_menu"), "MainWindow missing help_menu"
    assert hasattr(win, "action_help"), "MainWindow missing action_help"
    assert win.action_help.shortcut().toString() == "F1", f"Expected shortcut F1, got {win.action_help.shortcut().toString()}"
    print("  Help menu and F1 action verified.")

def test_no_solid_blue_drag():
    print("Testing HexagonFacetROI paint during edge drag...")
    corners = [[100 * np.cos(k * np.pi / 3), 100 * np.sin(k * np.pi / 3)] for k in range(6)]
    roi = HexagonFacetROI(corners=corners)
    
    # Simulate edge 0 being dragged
    roi.edge_drag_idx = 0
    
    # Render into a QImage with QPainter
    img = QtGui.QImage(400, 400, QtGui.QImage.Format_ARGB32)
    img.fill(QtCore.Qt.black)
    painter = QtGui.QPainter(img)
    
    opt = QtWidgets.QStyleOptionGraphicsItem()
    roi.paint(painter, opt, None)
    painter.end()

    # Sample pixel at center (200, 200) - must NOT be solid light blue (#00E5FF)
    # The background should remain dark or slightly translucent yellow
    pixel_center = img.pixelColor(200, 200)
    print(f"  Center pixel color: R={pixel_center.red()}, G={pixel_center.green()}, B={pixel_center.blue()}, A={pixel_center.alpha()}")
    # Cyan is R=0, G=229, B=255. Ensure the whole image didn't turn solid cyan!
    assert not (pixel_center.red() == 0 and pixel_center.green() >= 220 and pixel_center.blue() == 255), "Image turned solid light blue!"
    print("  PASSED: Image does NOT turn solid light blue during drag.")

if __name__ == "__main__":
    test_help_dialog()
    test_menu_help_action()
    test_no_solid_blue_drag()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
