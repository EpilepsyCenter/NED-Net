# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the NED-Net desktop app.

Build on each platform (PyInstaller does not cross-compile), from a clean
environment with NED-Net installed:

    python -m venv .venv-build && source .venv-build/bin/activate   # Windows: .venv-build\\Scripts\\activate
    pip install -e ".[desktop,ml]" pyinstaller
    pyinstaller packaging/nednet.spec --noconfirm

Output in dist/:
    macOS    NED-Net.app
    Windows  NED-Net\\NED-Net.exe (with its folder)
    Linux    NED-Net/NED-Net (with its folder)

A folder build ("onedir"), not a single file: a single-file app unpacks
itself on every start, which is slow with PyTorch and makes virus scanners
(Windows Defender) check it again each time. See docs/PACKAGING.md.
"""
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
APP = ROOT / "eeg_seizure_analyzer" / "dash_app"
VERSION = re.search(r'__version__\s*=\s*"([^"]+)"',
                    (ROOT / "eeg_seizure_analyzer" / "__init__.py").read_text()).group(1)

# The app's own files (stylesheet, scripts, logo, icons, manual + screenshots)
datas = [
    (str(APP / "assets"), "eeg_seizure_analyzer/dash_app/assets"),
    (str(APP / "vendor"), "eeg_seizure_analyzer/dash_app/vendor"),
]
# Dash, Plotly, dash-bootstrap-components, PyTorch and pywebview have hooks in
# pyinstaller-hooks-contrib; AG Grid's JavaScript has to be added here.
datas += collect_data_files("dash_ag_grid")

hiddenimports = (
    collect_submodules("eeg_seizure_analyzer")
    # Windows file dialogs run tkinter from a string (file_dialogs._tk_frozen)
    + ["tkinter", "tkinter.filedialog", "waitress"]
)
try:  # ADICHT reading (Windows only), when installed
    import adi  # noqa: F401
    hiddenimports.append("adi")
except Exception:
    pass

excludes = ["IPython", "jupyter_client", "jupyter_core", "notebook", "pytest",
            "torchvision", "torchaudio"]
if not sys.platform.startswith("linux"):
    # pywebview uses the system web engine on macOS and Windows; Qt only on Linux
    excludes += ["PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy"]

if sys.platform == "darwin":
    ICON = APP / "assets" / "nednet.icns"
elif sys.platform.startswith("win"):
    ICON = APP / "assets" / "nednet_icon.ico"
else:
    ICON = APP / "assets" / "nednet_icon.png"

a = Analysis(
    [str(ROOT / "packaging" / "nednet_entry.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NED-Net",
    console=False,          # a desktop app: no terminal window
    icon=str(ICON),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="NED-Net", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="NED-Net.app",
        icon=str(ICON),
        bundle_identifier="se.lu.med.nednet",
        version=VERSION,
        info_plist={
            "CFBundleName": "NED-Net",
            "CFBundleDisplayName": "NED-Net",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSHumanReadableCopyright": "NED-Net, GNU GPL v3.0",
        },
    )
