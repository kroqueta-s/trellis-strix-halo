<#
.SYNOPSIS
    Install the TRELLIS.2 runner: its own virtual environment, upstream, weights.

.DESCRIPTION
    A second model means a second environment (one model per environment), a
    second upstream checkout and a second set of weights. Nothing is compiled:
    the CUDA halves are replaced at launch time by the shims in
    runners/trellis/shims.py, shared with the TRELLIS.1 runner.

    **The image conditioner is gated.** facebook/dinov3-vitl16-pretrain-lvd1689m
    needs an accepted licence and a read token; a fine-grained token with the
    Read-only preset is enough. Put it in HF_TOKEN, or pass -HfToken.

    What is downloaded: the four geometry checkpoints (about 8.1 GB), DINOv3
    (1.2 GB) and BiRefNet (about 0.9 GB). **The texture checkpoints are not
    fetched unless -WithTexture is given**, which adds 6.1 GB and is what
    TRELLIS2_VERTEX_COLORS needs. The two encoders, which are for training, are
    never fetched.

.EXAMPLE
    $env:HF_TOKEN = "hf_..."
    .\install-trellis2.ps1

.EXAMPLE
    .\install-trellis2.ps1 -WithTexture
#>
[CmdletBinding()]
param(
    # Where the virtual environment, the upstream clone and the weights go.
    # Empty means: next to this repository, in trellis-strix-halo-data.
    [string]$Root = "",
    [string]$WeightsRoot = "",
    [string]$Python = "py -3.12",
    [string]$HfToken = "",
    # Also fetch the texture flow models and their decoder (6.1 GB), which is
    # what colours the vertices. Geometry does not need them.
    [switch]$WithTexture
)

# Native tools report progress on stderr, and Windows PowerShell 5.1 turns
# those lines into error records under redirection, so every native step is
# checked by its exit code rather than by an error preference.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"   # the progress bar costs most of the download throughput
function Assert-Ok([string]$step) {
    if ($LASTEXITCODE) { throw "$step failed with exit code $LASTEXITCODE" }
}

$repo = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Root) { $Root = Join-Path (Split-Path -Parent $repo) "trellis-strix-halo-data" }
if (-not $WeightsRoot) { $WeightsRoot = "C:\dev\models\trellis2" }
if (-not $HfToken) { $HfToken = $env:HF_TOKEN }

# Pinned. Do not float these: the ROCm wheels and the upstream commit are what
# the measurements in docs/trellis2.md were taken against.
$TorchIndex = "https://stable.repo.amd.com/rocm/whl-next/"
$TorchVersion = "2.13.0+rocm10.0.0"
$TorchvisionVersion = "0.28.0+rocm10.0.0"
$UpstreamUrl = "https://github.com/microsoft/TRELLIS.2.git"
$UpstreamCommit = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
$WeightsRepo = "microsoft/TRELLIS.2-4B"
$DinoRepo = "facebook/dinov3-vitl16-pretrain-lvd1689m"
$RembgRepo = "ZhengPeng7/BiRefNet"

$venv = Join-Path $Root "venv2"
$py = Join-Path $venv "Scripts\python.exe"
$upstream = Join-Path $Root "TRELLIS2"

if (-not $HfToken) {
    throw @"
No Hugging Face token. **$DinoRepo is gated**: accept its licence at
https://huggingface.co/$DinoRepo, create a read token at
https://huggingface.co/settings/tokens (the Read-only preset is enough), and
put it in HF_TOKEN or pass -HfToken. Nothing else here needs one.
"@
}

# 1. The virtual environment ---------------------------------------------------
if (-not (Test-Path $py)) {
    Write-Host "==> Creating the virtual environment ($venv)"
    Invoke-Expression "$Python -m venv `"$venv`""
    Assert-Ok "venv creation"
}
& $py -m pip install --quiet --upgrade pip
Assert-Ok "pip upgrade"

# 2. ROCm PyTorch --------------------------------------------------------------
Write-Host "==> Installing ROCm PyTorch (about 4 GB)"
& $py -m pip install --index-url $TorchIndex --extra-index-url https://pypi.org/simple `
    "torch==$TorchVersion" "torchvision==$TorchvisionVersion" `
    "amd-torch-device-gfx115x==$TorchVersion" "amd-torch-device-gfx1151==$TorchVersion" `
    "amd-torchvision-device-gfx1151==$TorchvisionVersion"
Assert-Ok "PyTorch installation"

Write-Host "==> Installing the runtime dependencies"
& $py -m pip install -r (Join-Path $repo "requirements-trellis2.txt")
Assert-Ok "dependency installation"

# 3. Upstream, at a pinned commit ----------------------------------------------
if (-not (Test-Path $upstream)) {
    Write-Host "==> Cloning upstream TRELLIS.2 (shallow)"
    & git clone --depth 1 $UpstreamUrl $upstream
    Assert-Ok "git clone"
}
Push-Location $upstream
& git fetch --depth 1 origin $UpstreamCommit
& git checkout $UpstreamCommit
if ($LASTEXITCODE) { Pop-Location; throw "git checkout failed ($LASTEXITCODE)" }
Pop-Location

# 4. Weights -------------------------------------------------------------------
# **Fetched over plain HTTPS on purpose.** The Xet transfer path stalls at zero
# bytes on this class of machine; resolve/main does not.
function Get-HfFile([string]$repoId, [string]$file, [string]$target, [string]$token) {
    if (Test-Path $target) {
        Write-Host ("    have {0} ({1:N0} bytes)" -f $file, (Get-Item $target).Length)
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    $headers = @{}
    if ($token) { $headers["Authorization"] = "Bearer $token" }
    $started = Get-Date
    Invoke-WebRequest -Uri "https://huggingface.co/$repoId/resolve/main/$file" `
        -OutFile "$target.part" -UseBasicParsing -Headers $headers
    Move-Item -Path "$target.part" -Destination $target -Force
    $length = (Get-Item $target).Length
    $seconds = [math]::Max(((Get-Date) - $started).TotalSeconds, 0.001)
    Write-Host ("    got  {0}  {1:N0} bytes  {2:N1} MB/s" -f $file, $length, ($length / 1MB / $seconds))
}

$geometry = if ($WithTexture) { "8.1 GB plus 6.1 GB of texture" } else { "about 8.1 GB; the texture ones are skipped" }
Write-Host "==> Downloading the weights ($geometry)"
foreach ($file in @(
    "pipeline.json", "README.md",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.json", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json", "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json", "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    "ckpts/shape_dec_next_dc_f16c32_fp16.json", "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
)) {
    Get-HfFile $WeightsRepo $file (Join-Path $WeightsRoot ($file -replace '/', '\')) ""
}

# **Texture is opt-in.** `write_local_pipeline.py` keeps the texture entries in
# the description only when their checkpoints are present, so this switch is the
# whole difference between a geometry runner and one that can colour vertices.
if ($WithTexture) {
    foreach ($file in @(
        "ckpts/tex_dec_next_dc_f16c32_fp16.json", "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors"
    )) {
        Get-HfFile $WeightsRepo $file (Join-Path $WeightsRoot ($file -replace '/', '\')) ""
    }
}

# The sparse-structure decoder is the same checkpoint TRELLIS.1 uses. It comes
# from that repository, and lands here so this directory is self-contained.
foreach ($file in @("ckpts/ss_dec_conv3d_16l8_fp16.json", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors")) {
    Get-HfFile "microsoft/TRELLIS-image-large" $file (Join-Path $WeightsRoot ($file -replace '/', '\')) ""
}

Write-Host "==> Downloading the image conditioner (gated; 1.2 GB)"
foreach ($file in @("config.json", "preprocessor_config.json", "model.safetensors", "LICENSE.md", "README.md")) {
    Get-HfFile $DinoRepo $file (Join-Path $WeightsRoot "dinov3-vitl16-pretrain-lvd1689m\$file") $HfToken
}

Write-Host "==> Downloading the background remover (about 0.9 GB)"
foreach ($file in @("config.json", "birefnet.py", "BiRefNet_config.py", "model.safetensors")) {
    Get-HfFile $RembgRepo $file (Join-Path $WeightsRoot "BiRefNet\$file") ""
}

# **The licence sits beside the weights.** They are the upstream terms, not
# this repository's.
$license = Join-Path $WeightsRoot "LICENSE"
if (-not (Test-Path $license)) {
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/microsoft/TRELLIS.2/main/LICENSE" `
        -OutFile $license -UseBasicParsing
}

# 5. A pipeline description that names local files only ------------------------
# **The downloaded pipeline.json is never edited.** `from_pretrained` takes the
# config file's name as an argument, so this one sits beside it.
Write-Host "==> Writing pipeline.local.json"
& $py (Join-Path $repo "tools\write_local_pipeline.py") $WeightsRoot
Assert-Ok "pipeline description"

# 6. .env ----------------------------------------------------------------------
$envPath = Join-Path $repo ".env"
if (-not (Test-Path $envPath)) { Copy-Item (Join-Path $repo ".env.example") $envPath }
$envText = Get-Content $envPath -Raw
if ($envText -notmatch "TRELLIS2_REPO=") {
    Write-Host "==> Adding the TRELLIS2_* settings to .env"
    $block = (Get-Content (Join-Path $repo ".env.example") -Raw)
    $block = ($block -split "# TRELLIS\.2 \(runners/trellis2\)")[1]
    $block = "# TRELLIS.2 (runners/trellis2)" + $block
    $block = $block.Replace("__TRELLIS2_REPO__", $upstream).Replace("__TRELLIS2_WEIGHTS__", $WeightsRoot)
    Add-Content -Path $envPath -Value "`n$block" -Encoding utf8
} else {
    Write-Host "==> .env already carries TRELLIS2_* settings; leaving it alone"
}

# 7. Verify before trusting any mesh -------------------------------------------
Write-Host "==> Verifying the shims and the mesh repairs"
foreach ($test in @("tests\test_shims.py", "tests\test_shims2.py", "tests\test_close_holes.py", "tests\test_split_manifold.py")) {
    & $py (Join-Path $repo $test)
    Assert-Ok $test
}

Write-Host ""
Write-Host "Done. Generate a first mesh with:"
Write-Host "  $py -m runners.trellis2   (then speak the protocol on stdin)"
Write-Host ""
Write-Host "Or point hearth at this checkout:"
Write-Host "  HEARTH_RUNNER_TRELLIS2_PYTHON=$py"
Write-Host "  HEARTH_RUNNER_TRELLIS2_MODULE=runners.trellis2"
Write-Host "  HEARTH_RUNNER_TRELLIS2_CWD=$repo"
