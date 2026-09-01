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

$ErrorActionPreference = "Stop"
# $PSScriptRoot can be empty while param defaults are evaluated under
# Windows PowerShell 5.1, so the paths are resolved here instead.
$repo = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Root) { $Root = Join-Path (Split-Path -Parent $repo) "trellis-strix-halo-data" }

# Pinned versions. Do not float these: the ROCm wheels and the upstream commit
# are the two things that decide whether this works at all.
$TorchIndex = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
$TorchVersion = "2.9.1+rocm7.2.1"
$TorchvisionVersion = "0.24.1+rocm7.2.1"
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
}
& $py -m pip install --upgrade pip

# 2. ROCm PyTorch -------------------------------------------------------------
# torch requires the `rocm` meta-package, which lives on the same index.
# Passing the wheel URL directly fails with "No matching distribution for rocm".
Write-Host "==> Installing ROCm PyTorch"
& $py -m pip install --no-cache-dir --find-links $TorchIndex `
    "torch==$TorchVersion" "torchvision==$TorchvisionVersion"

# 3. Upstream repository (never forked, never patched) ------------------------
if (-not (Test-Path $upstream)) {
    Write-Host "==> Cloning upstream TRELLIS"
    # A shallow clone: the full TRELLIS history in particular is so large that
    # the server-side pack preparation can stall for many minutes. The pinned
    # commit is fetched right below, also shallow.
    git clone --depth 1 $UpstreamUrl $upstream
}
Push-Location $upstream
git fetch --depth 1 origin $UpstreamCommit
git checkout $UpstreamCommit
git submodule update --init --recursive
Pop-Location

# 4. Pure-python dependencies -------------------------------------------------
Write-Host "==> Installing dependencies"
& $py -m pip install --no-cache-dir -r (Join-Path $repo "requirements.txt")

# 5. Weights ------------------------------------------------------------------
Write-Host "==> Downloading weights (about 3.1 GB)"
& $py -c "from huggingface_hub import snapshot_download; snapshot_download('$WeightsRepo', local_dir=r'$weights')"

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
& $py (Join-Path $repo "tests\test_raster.py")
& $py (Join-Path $repo "tests\test_drop_parts.py")

Write-Host ""
Write-Host "Done. Generate a first mesh with:"
Write-Host "  $py $repo\tools\run_single.py --image $repo\assets\sample.png --out $Root\out"
Write-Host ""
Write-Host "Or point hearth at this checkout:"
Write-Host "  HEARTH_RUNNER_TRELLIS_PYTHON=$py"
Write-Host "  HEARTH_RUNNER_TRELLIS_MODULE=runners.trellis"
Write-Host "  HEARTH_RUNNER_TRELLIS_CWD=$repo"
