@echo off
echo 🚀 Stable Diffusion Setup for RTX 4060
echo.

cd image_generation

if not exist "stable-diffusion-webui" (
    echo 📥 Cloning Automatic1111 WebUI...
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
    if errorlevel 1 (
        echo ❌ Git clone failed!
        echo 📥 Please manually download from:
        echo https://github.com/AUTOMATIC1111/stable-diffusion-webui/archive/refs/heads/master.zip
        echo 📂 Extract to: image_generation\stable-diffusion-webui
        pause
        exit /b 1
    )
) else (
    echo ✅ Stable Diffusion WebUI already exists
)

echo.
echo 📥 Checking for SD 1.5 model...
if not exist "stable-diffusion-webui\models\Stable-diffusion\v1-5-pruned-emaonly.safetensors" (
    echo Downloading SD 1.5 model (smaller, faster for RTX 4060)...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors' -OutFile 'stable-diffusion-webui\models\Stable-diffusion\v1-5-pruned-emaonly.safetensors'}"
    if errorlevel 1 (
        echo ❌ Model download failed!
        echo 📥 Download manually from:
        echo https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
        echo Save to: image_generation\stable-diffusion-webui\models\Stable-diffusion\v1-5-pruned-emaonly.safetensors
        pause
    ) else (
        echo ✅ Model downloaded successfully!
    )
) else (
    echo ✅ Model already exists
)

echo.
echo 🎉 Setup Complete!
echo.
echo 📋 To start Stable Diffusion:
echo 1. Run: .\image_generation\start_sd.bat
echo 2. Open: http://localhost:7860
echo 3. First run downloads components (5-10 minutes)
echo.
echo ⚡ Your RTX 4060 Performance:
echo • 512x512: ~3-5 seconds per image
echo • 1024x1024: ~15-25 seconds per image
echo.
pause