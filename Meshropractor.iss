; Compile after: .\.venv\Scripts\python.exe -m PyInstaller --noconfirm Meshropractor.spec
; Inno Setup 6.3+ / 7, Windows x64. Copy the entire PyInstaller onedir distribution.
#define MyAppName "Meshropractor"
#ifndef MyAppVersion
  #define MyAppVersion "0.2.5"
#endif
#define MyAppExeName "Meshropractor.exe"
#define BuildDir SourcePath + "dist\Meshropractor"

#if !FileExists(BuildDir + "\" + MyAppExeName)
  #error Build dist\Meshropractor with PyInstaller before compiling this installer.
#endif

[Setup]
; Keep the ID of the existing installation for upgrades.
AppId={{1B9FDA16-1332-410F-BFE5-3D07B66F7B35}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=MeshropractorTeam
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UsePreviousAppDir=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
OutputDir={#SourcePath}dist\installer
OutputBaseFilename=Meshropractor-Setup-{#MyAppVersion}-x64
SetupIconFile={#SourcePath}assets\logo.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent runasoriginaluser
