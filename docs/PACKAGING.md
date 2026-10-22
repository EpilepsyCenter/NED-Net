# Packaging NED-Net as a desktop app

The packaged app is a self-contained folder (or `.app` on macOS) with Python, NED-Net and all
its libraries, including PyTorch. Users need no Python installation.

PyInstaller does not cross-compile: each platform is built on that platform. GitHub does this
for you (next section); building by hand is described after it.

## Releases on GitHub (recommended)

`.github/workflows/build.yml` builds all three platforms on GitHub's machines:

1. Set the version in `eeg_seizure_analyzer/__init__.py` (and `pyproject.toml`), commit.
2. Tag and push: `git tag v1.0.2 && git push origin v1.0.2`.
3. GitHub Actions builds the apps (about 20-30 minutes) and creates a release with:
   - `NED-Net-<version>-mac-arm64.dmg` (Apple Silicon);
   - `NED-Net-<version>-Setup.exe` (Windows installer);
   - `NED-Net-<version>-linux-x64.tar.gz` (CPU-only PyTorch).

For a test build without a release, open **Actions › Build desktop apps › Run workflow**; the
files appear under the run's **Artifacts**.

Tag the **public** repository (EpilepsyCenter/NED-Net) for releases colleagues can download:
Actions are free there. On the private research repository a run uses the account's Actions
minutes (macOS minutes count ten times), and its releases are private.

## Building by hand

From the repository root, in a clean environment that holds only NED-Net and its
dependencies (a busy environment makes the app larger and slower to start):

```bash
python -m venv .venv-build
source .venv-build/bin/activate        # Windows: .venv-build\Scripts\activate
pip install -e ".[desktop,ml]" pyinstaller
# Windows, to open ADICHT files: use ".[desktop,ml,windows]" instead
pyinstaller packaging/nednet.spec --noconfirm
```

| Platform | Result in `dist/` | Ship it as |
|---|---|---|
| macOS | `NED-Net.app` | zip it, or a `.dmg` (see below) |
| Windows | `NED-Net\` with `NED-Net.exe` | the installer from `packaging/windows/nednet.iss` (Inno Setup), or a zip of the folder |
| Linux | `NED-Net/` with `NED-Net` | a `.tar.gz` of the folder |

**GPU support** comes from the PyTorch in the build environment:
- macOS: the standard PyTorch uses the Apple-silicon GPU.
- Windows / Linux with NVIDIA: install the CUDA build of PyTorch
  ([pytorch.org](https://pytorch.org/get-started/locally/)) *before* `pip install -e`. A CPU-only
  PyTorch makes a much smaller app.

## How the packaged app differs from running from source

`packaging/nednet_entry.py` is the entry point. It:
- calls `multiprocessing.freeze_support()`, so worker processes (dataset building, background
  proposals, training data loaders) work in the bundle;
- runs Windows file dialogs in a second instance of the app (`--nednet-run-script`), because
  there is no separate Python to run them with (`file_dialogs._tk_frozen`);
- writes output and errors to a log file, since a windowed app has no console:

| Platform | Log file |
|---|---|
| macOS | `~/Library/Logs/NED-Net.log` |
| Windows | `%LOCALAPPDATA%\NED-Net\NED-Net.log` |
| Linux | `~/.local/state/nednet/NED-Net.log` |

User data stays where it always is (`~/.eeg_seizure_analyzer/`), so models, projects and
settings are shared with a source install.

It is a folder build ("onedir"), not a single file: a single-file app unpacks itself on every
start, which is slow with PyTorch and makes virus scanners check it again each time.

## First start and virus scanners

The first start after installing can take a minute: the operating system (on Windows, Defender)
scans the libraries, PyTorch most of all. The splash screen shows while this happens. Later
starts are fast.

## Signing (recommended before sharing widely)

Unsigned apps run, but the system warns first.

**macOS.** An unsigned app from the internet is blocked by Gatekeeper ("cannot be opened because
the developer cannot be verified"). Users can right-click the app and choose **Open** once, or run
`xattr -dr com.apple.quarantine /Applications/NED-Net.app`. To avoid the warning, sign and
notarize with an Apple Developer ID (paid Apple Developer Program):

```bash
codesign --deep --force --options runtime --timestamp \
    --sign "Developer ID Application: <Name> (<TEAMID>)" dist/NED-Net.app
ditto -c -k --keepParent dist/NED-Net.app NED-Net.zip
xcrun notarytool submit NED-Net.zip --apple-id <id> --team-id <TEAMID> --wait
xcrun stapler staple dist/NED-Net.app
```

A `.dmg` for distribution: `hdiutil create -volname NED-Net -srcfolder dist/NED-Net.app -ov
-format UDZO NED-Net.dmg`.

**Windows.** SmartScreen shows "Windows protected your PC" for unsigned apps (**More info → Run
anyway**). A code-signing certificate removes it: sign `NED-Net.exe` and the installer with
`signtool sign /fd SHA256 /tr <timestamp-url> /td SHA256 ...`.

## Windows installer

`packaging/windows/nednet.iss` is an [Inno Setup](https://jrsoftware.org/isinfo.php) script. After
the PyInstaller build, open it in Inno Setup and press **Compile** (or run `iscc
packaging\windows\nednet.iss`). The installer goes to `dist\installer\` and adds NED-Net to the
Start menu, with an optional desktop shortcut and an uninstaller. It installs per user, so no
administrator rights are needed.

## Checking a build

1. Start the app: the splash screen appears at once, then the app.
2. Load a recording, open the Viewer, run a detection.
3. **File › Open Recording…** opens the system file dialog (on Windows this checks the
   second-instance dialog path).
4. **Help › NED-Net Manual** shows the manual with its screenshots.
5. If anything fails, read the log file (table above).
