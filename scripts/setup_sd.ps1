# Stable Diffusion Setup Script for RTX 4060
# Run this script to set up local image generation

Write-Host "🚀 Setting up Stable Diffusion for RTX 4060..." -ForegroundColor Green

# Check if we're in the right directory
if (!(Test-Path "image_generation")) {
    Write-Host "❌ Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

cd image_generation

# Download Automatic1111 WebUI
Write-Host "📥 Downloading Automatic1111 Stable Diffusion WebUI..." -ForegroundColor Yellow
try {
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
    Write-Host "✅ Repository cloned successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Git clone failed. Trying alternative download..." -ForegroundColor Red
    Write-Host "📥 Please manually download from: https://github.com/AUTOMATIC1111/stable-diffusion-webui/archive/refs/heads/master.zip" -ForegroundColor Yellow
    Write-Host "📂 Extract to: image_generation/stable-diffusion-webui" -ForegroundColor Yellow
    Read-Host "Press Enter after downloading and extracting"
}

# Download SD 1.5 model (smaller, faster for RTX 4060)
Write-Host "📥 Downloading Stable Diffusion 1.5 model..." -ForegroundColor Yellow
$modelUrl = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
$modelPath = "stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors"

if (!(Test-Path "stable-diffusion-webui")) {
    Write-Host "❌ stable-diffusion-webui directory not found. Please download it first." -ForegroundColor Red
    exit 1
}

if (!(Test-Path $modelPath)) {
    Write-Host "Downloading model to: $modelPath" -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
        Write-Host "✅ Model downloaded successfully!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Model download failed. You can download it manually from:" -ForegroundColor Red
        Write-Host $modelUrl -ForegroundColor Yellow
        Write-Host "Save to: $modelPath" -ForegroundColor Yellow
    }
} else {
    Write-Host "✅ Model already exists!" -ForegroundColor Green
}

# Create startup script
$startupScript = @"
@echo off
echo 🚀 Starting Stable Diffusion WebUI...
cd /d "%~dp0stable-diffusion-webui"
webui-user.bat --listen --port 7860 --enable-insecure-extension-access --opt-sdp-attention
"@

$startupScript | Out-File -FilePath "start_sd.bat" -Encoding UTF8

Write-Host "`n🎉 Setup Complete!" -ForegroundColor Green
Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Run: .\image_generation\start_sd.bat" -ForegroundColor White
Write-Host "2. Open browser to: http://localhost:7860" -ForegroundColor White
Write-Host "3. First run will download additional components (takes 5-10 minutes)" -ForegroundColor White
Write-Host "`n⚡ Performance Expectations:" -ForegroundColor Yellow
Write-Host "• RTX 4060: 512x512 image in ~3-5 seconds" -ForegroundColor White
Write-Host "• RTX 4060: 1024x1024 image in ~15-25 seconds" -ForegroundColor White
Write-Host "• Quality: Excellent for your GPU generation" -ForegroundColor White

Write-Host "`n💡 Tips:" -ForegroundColor Cyan
Write-Host "• Use 'Euler a' sampler for fastest generation" -ForegroundColor White
Write-Host "• 20-30 steps for good quality/speed balance" -ForegroundColor White
Write-Host "• CFG Scale 7-12 works well" -ForegroundColor White
Write-Host "• Negative prompts: 'blurry, low quality, deformed'" -ForegroundColor White

Read-Host "`nPress Enter to continue"