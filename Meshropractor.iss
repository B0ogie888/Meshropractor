; First build RepairEngine.spec, then Meshropractor.spec with PyInstaller.
; Inno Setup 6.3+ / 7, Windows x64. Copy the entire PyInstaller onedir distribution.
#define MyAppName "Meshropractor"
#ifndef MyAppVersion
  #define VersionFile FileOpen(SourcePath + "VERSION")
  #if VersionFile < 0
    #error Cannot read VERSION from the project directory.
  #endif
  #define MyAppVersion Trim(FileRead(VersionFile))
  #expr FileClose(VersionFile)
#endif

#define MyAppExeName "Meshropractor.exe"
#define BuildDir SourcePath + "dist\Meshropractor"

#if !FileExists(BuildDir + "\_internal\repair_engine\MeshRepairEngine.exe")
  #error Build RepairEngine.spec and then Meshropractor.spec to include the full repair engine.
#endif

#if !FileExists(BuildDir + "\" + MyAppExeName)
  #error Build dist\Meshropractor with PyInstaller before compiling this installer.
#endif

#if !FileExists(BuildDir + "\_internal\VERSION")
  #error Missing bundled VERSION. Rebuild Meshropractor.spec before compiling this installer.
#endif
#define BuildVersionFile FileOpen(BuildDir + "\_internal\VERSION")
#if BuildVersionFile < 0
  #error Cannot read the bundled VERSION from dist\Meshropractor\_internal.
#endif
#define BuildAppVersion Trim(FileRead(BuildVersionFile))
#expr FileClose(BuildVersionFile)
#if BuildAppVersion != MyAppVersion
  #error Installer version differs from the bundled app VERSION. Rebuild Meshropractor.spec or correct /DMyAppVersion.
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
; GitHub Releases requires each asset to be smaller than 2 GiB.
; Keep the complete CUDA/CAD/repair runtime and compress it for distribution.
Compression=lzma2/ultra64
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
Filename: "{app}\{#MyAppExeName}"; Flags: nowait runasoriginaluser; Check: IsSilentUpdate

[Code]
function IsSilentUpdate: Boolean;
begin
  Result := WizardSilent and (ExpandConstant('{param:UPDATE|0}') = '1');
end;
