#ifndef SourceDir
  #error SourceDir must point to a packaged FSC Studio directory.
#endif

#ifndef OutputDir
  #define OutputDir "out\\installer"
#endif

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

#ifndef Architecture
  #define Architecture "x64"
#endif

#ifndef SetupBaseName
  #define SetupBaseName "FSC-Studio-Setup-x64"
#endif

#ifndef VCRedistFile
  #define VCRedistFile "VC_redist.x64.exe"
#endif

#ifndef ApplicationId
  #define ApplicationId "{{594F0143-C6C1-442A-9DE7-4D2528B3EA41}"
#endif

#define AppName "FSC Studio"
#define AppPublisher "FSC Studio"
#define AppExeName "FscStudioQt.exe"

[Setup]
AppId={#ApplicationId}
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
#if Architecture == "arm64"
ArchitecturesAllowed=arm64
ArchitecturesInstallIn64BitMode=arm64
#else
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
#endif
UninstallDisplayName=FSC Studio
InfoBeforeFile={#SourceDir}\THIRD-PARTY-NOTICES.txt
#ifdef PerUserTestMode
PrivilegesRequired=lowest
#else
PrivilegesRequired=admin
#endif
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "_redist\*,*.zip,mesh_render*.png,qt_plugin_debug.txt"
Source: "{#SourceDir}\_redist\{#VCRedistFile}"; DestDir: "{tmp}"; Flags: ignoreversion deleteafterinstall

[Icons]
Name: "{autoprograms}\FSC Studio"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autoprograms}\FSC Studio User Guide"; Filename: "{app}\FSC-Studio-User-Guide.html"
Name: "{autodesktop}\FSC Studio"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
#ifndef PerUserTestMode
Filename: "{tmp}\{#VCRedistFile}"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing Microsoft Visual C++ Runtime..."; Flags: runhidden waituntilterminated
#endif
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,FSC Studio}"; Flags: nowait postinstall skipifsilent
