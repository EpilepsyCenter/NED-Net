; Inno Setup script for the NED-Net Windows installer.
; Build the app first (pyinstaller packaging\nednet.spec --noconfirm), then
; compile this script with Inno Setup (iscc packaging\windows\nednet.iss).
; Installs per user (no administrator rights needed).

#define AppName "NED-Net"
; The release workflow passes the version (iscc /DAppVersion=x.y.z); otherwise
; keep this in step with eeg_seizure_analyzer/__init__.py
#ifndef AppVersion
  #define AppVersion "1.0.1"
#endif
#define AppExe "NED-Net.exe"

[Setup]
AppId={{4825C3A5-4035-4A22-BEF2-B40010ECDE66}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=NED-Net
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
OutputDir=..\..\dist\installer
OutputBaseFilename=NED-Net-{#AppVersion}-Setup
SetupIconFile=..\..\eeg_seizure_analyzer\dash_app\assets\nednet_icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\..\dist\NED-Net\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent
