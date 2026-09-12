# SPDX-License-Identifier: MIT
# Build o_voxel_cpu with MSVC and put the module where the runner looks for it.
#
#   .\native\o_voxel_cpu\build.ps1 -Python C:\path\to\trellis2-venv\Scripts\python.exe
#
# Needs Visual Studio 2022 Build Tools with the "Desktop development with C++"
# workload (it brings the Windows SDK), and TRELLIS2_REPO in .env. Eigen is
# downloaded once into the output directory: it is header-only and MPL-2.0.
#
# The module is written to TRELLIS2_NATIVE_DIR (from .env), or next to this
# script when that is unset. **It is never tracked**: a compiled module is
# bound to the torch it was built against.
param(
    [Parameter(Mandatory = $true)][string]$Python,
    [string]$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    [string]$EigenVersion = "3.4.0"
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$repo = Resolve-Path (Join-Path $here "..\..")

# Read the two keys this needs from .env without importing anything.
$dotenv = @{}
Get-Content (Join-Path $repo ".env") | ForEach-Object {
    if ($_ -match '^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$') { $dotenv[$matches[1]] = $matches[2] }
}
$out = $dotenv["TRELLIS2_NATIVE_DIR"]
if (-not $out) { $out = $here }
New-Item -ItemType Directory -Force $out | Out-Null

$eigen = Join-Path $out "eigen-$EigenVersion"
if (-not (Test-Path (Join-Path $eigen "Eigen\Dense"))) {
    $zip = Join-Path $out "eigen-$EigenVersion.zip"
    Write-Host "downloading Eigen $EigenVersion (header-only, MPL-2.0)"
    Invoke-WebRequest -Uri "https://gitlab.com/libeigen/eigen/-/archive/$EigenVersion/eigen-$EigenVersion.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $out -Force
}

if (-not (Test-Path $VcVars)) { throw "vcvars64.bat not found at $VcVars (install Visual Studio 2022 Build Tools, C++ workload)" }

# The build runs inside cmd so that vcvars64 can set the compiler environment;
# DISTUTILS_USE_SDK tells torch's build helper to use it rather than look for one.
$bat = Join-Path $out "build_o_voxel_cpu.bat"
@(
    "@echo off",
    "call `"$VcVars`" >nul",
    "set DISTUTILS_USE_SDK=1",
    "set O_VOXEL_EIGEN_DIR=$eigen",
    "cd /d `"$here`"",
    "`"$Python`" setup.py build_ext --build-lib `"$out`" --build-temp `"$(Join-Path $out 'build_temp')`""
) | Set-Content -Path $bat -Encoding ASCII
cmd /c "`"$bat`""
if ($LASTEXITCODE -ne 0) { throw "build failed (exit $LASTEXITCODE)" }
Get-ChildItem $out -Filter "o_voxel_cpu*.pyd" | ForEach-Object { Write-Host "built $($_.FullName)" }
