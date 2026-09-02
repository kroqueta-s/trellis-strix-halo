# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
    TRELLIS on Strix Halo (gfx1151 / Windows / ROCm) - one-command install.

.DESCRIPTION
    Creates a dedicated virtual environment, installs ROCm PyTorch, clones the
    upstream TRELLIS repository at a pinned commit, downloads the weights and
    writes a .env file.

    **No CUDA-only package is installed.** spconv, flash-attn, xformers, kaolin
    and nvdiffrast are replaced at launch time by pure-torch shims that live in
    runners/trellis/. Upstream code is never patched.

.PARAMETER Root
    Where the virtual environment, the upstream clone and the weights go.
    Defaults to the parent of this repository.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Root D:\models\trellis
#>
[CmdletBinding()]
param(
    # Where the virtual environment, the upstream clone and the weights go.
    # Empty means: next to this repository, in trellis-strix-halo-data.
    [string]$Root = "",
    [string]$Python = "py -3.12"
)

# Native tools (git, pip) report progress on stderr. Under output redirection,
# Windows PowerShell 5.1 turns those lines into error records, and a "Stop"
# preference would kill the script on the first one. So the preference stays
# "Continue" and every native step is checked through its exit code instead.
$ErrorActionPreference = "Continue"
function Assert-Ok([string]$step) {
    if ($LASTEXITCODE) { throw "$step failed with exit code $LASTEXITCODE" }
}

# $PSScriptRoot can be empty while param defaults are evaluated under
# Windows PowerShell 5.1, so the paths are resolved here instead.
$repo = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Root) { $Root = Join-Path (Split-Path -Parent $repo) "trellis-strix-halo-data" }

# Pinned versions. Do not float these: the ROCm wheels and the upstream commit
# are the two things that decide whether this works at all.
# The device packages matter: torch ships its GPU kernels as external kernel
# packs, and **both** the family (gfx115x: flash-attention images) and the
# exact-arch (gfx1151: torch kernel pack + ROCm library kernels) wheels are
# required. Without the exact-arch one, the first kernel launch fails with
# `hipErrorInvalidImage`. On another GPU, replace both suffixes with yours.
$TorchIndex = "https://stable.repo.amd.com/rocm/whl-next/"
$TorchVersion = "2.13.0+rocm10.0.0"
$TorchvisionVersion = "0.28.0+rocm10.0.0"
$TorchDeviceWheels = @(
    "amd-torch-device-gfx115x==$TorchVersion",
    "amd-torch-device-gfx1151==$TorchVersion",
    "amd-torchvision-device-gfx1151==$TorchvisionVersion"
)
$UpstreamUrl = "https://github.com/microsoft/TRELLIS.git"
$UpstreamCommit = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
$WeightsRepo = "microsoft/TRELLIS-image-large"

$venv = Join-Path $Root ".venv"
$upstream = Join-Path $Root "TRELLIS"
$weights = Join-Path $Root "weights"
$py = Join-Path $venv "Scripts\python.exe"

Write-Host "==> Root: $Root"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

# 1. Virtual environment ------------------------------------------------------
if (-not (Test-Path $py)) {
    Write-Host "==> Creating virtual environment"
    & cmd /c "$Python -m venv `"$venv`""
    Assert-Ok "virtual environment creation"
}
& $py -m pip install --upgrade pip
Assert-Ok "pip upgrade"

# 2. ROCm PyTorch -------------------------------------------------------------
# torch pulls the `rocm` runtime packages from the same index; PyPI stays as a
# fallback for the pure-python dependencies only (the exact +rocm pins can
# never match anything on PyPI).
Write-Host "==> Installing ROCm PyTorch"
& $py -m pip install --no-cache-dir --index-url $TorchIndex `
    --extra-index-url https://pypi.org/simple `
    "torch==$TorchVersion" "torchvision==$TorchvisionVersion" @TorchDeviceWheels
Assert-Ok "PyTorch installation"

# 3. Upstream repository (never forked, never patched) ------------------------
if (-not (Test-Path $upstream)) {
    Write-Host "==> Cloning upstream TRELLIS"
    # A shallow clone: a full history (TRELLIS in particular) can stall for
    # minutes in server-side pack preparation. The pinned commit is fetched
    # right below, also shallow.
    git clone --depth 1 $UpstreamUrl $upstream 2>&1 | Out-Host
    Assert-Ok "git clone"
}
Push-Location $upstream
git fetch --depth 1 origin $UpstreamCommit 2>&1 | Out-Host
if ($LASTEXITCODE) { Pop-Location; throw "git fetch failed ($LASTEXITCODE)" }
git checkout $UpstreamCommit 2>&1 | Out-Host
if ($LASTEXITCODE) { Pop-Location; throw "git checkout failed ($LASTEXITCODE)" }
git submodule update --init --recursive 2>&1 | Out-Host
if ($LASTEXITCODE) { Pop-Location; throw "git submodule update failed ($LASTEXITCODE)" }
Pop-Location

# 4. Pure-python dependencies -------------------------------------------------
Write-Host "==> Installing dependencies"
& $py -m pip install --no-cache-dir -r (Join-Path $repo "requirements.txt")
Assert-Ok "dependency installation"

# 5. Weights ------------------------------------------------------------------
Write-Host "==> Downloading weights (about 3.1 GB)"
& $py -c "from huggingface_hub import snapshot_download; snapshot_download('$WeightsRepo', local_dir=r'$weights')"
Assert-Ok "weights download"

# 6. .env ---------------------------------------------------------------------
$envPath = Join-Path $repo ".env"
if (-not (Test-Path $envPath)) {
    Write-Host "==> Writing .env"
    (Get-Content (Join-Path $repo ".env.example") -Raw).
        Replace("__REPO__", $upstream).
        Replace("__WEIGHTS__", $weights) | Set-Content -Path $envPath -Encoding utf8
}

# 7. Verify the shims before trusting any mesh --------------------------------
Write-Host "==> Verifying the shims (exact agreement with dense reference)"
& $py (Join-Path $repo "tests\test_shims.py")
Assert-Ok "shim verification"
& $py (Join-Path $repo "tests\test_raster.py")
Assert-Ok "rasterizer verification"
& $py (Join-Path $repo "tests\test_drop_parts.py")
Assert-Ok "debris-filter verification"

Write-Host ""
Write-Host "Done. Generate a first mesh with:"
Write-Host "  $py $repo\tools\run_single.py --image $repo\assets\sample.png --out $Root\out"
Write-Host ""
Write-Host "Or point hearth at this checkout:"
Write-Host "  HEARTH_RUNNER_TRELLIS_PYTHON=$py"
Write-Host "  HEARTH_RUNNER_TRELLIS_MODULE=runners.trellis"
Write-Host "  HEARTH_RUNNER_TRELLIS_CWD=$repo"
