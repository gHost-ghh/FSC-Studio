#ifndef SourceDir
  #error SourceDir must point to a packaged FSC Studio directory.
#endif

#ifndef OutputDir
  #define OutputDir "out\\installer"
#endif

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#ifndef SetupBaseName
  #define SetupBaseName "FSC-Studio-Setup-x64"
#endif

#define AppName "FSC Studio"
#define AppPublisher "FSC Studio"
#define AppExeName "FscStudioQt.exe"

[Setup]
AppId={{594F0143-C6C1-442A-9DE7-4D2528B3EA41}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\FSC Studio
DefaultGroupName=FSC Studio
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename={#SetupBaseName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=FSC Studio
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "Install-FSCStudioNative.bat,Install-FSCStudioNative.ps1,Uninstall-FSCStudioNative.ps1,Launch-FSCStudio.bat,*.zip,mesh_render*.png,qt_plugin_debug.txt"

[Icons]
Name: "{autoprograms}\FSC Studio"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\FSC Studio"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,FSC Studio}"; Flags: nowait postinstall skipifsilent
