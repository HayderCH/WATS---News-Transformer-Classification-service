@echo off
echo 🚀 Starting Stable Diffusion WebUI for RTX 4060...
echo.
echo 💡 Tips for best performance:
echo • Use 'Euler a' sampler (fastest)
echo • 20-30 steps for quality/speed balance
echo • CFG Scale: 7-12
echo • Negative prompt: "blurry, low quality, deformed"
echo.
echo ⚡ Expected performance on RTX 4060:
echo • 512x512 images: 3-5 seconds
echo • 1024x1024 images: 15-25 seconds
echo.

cd stable-diffusion-webui
webui-user.bat --skip-python-version-check --listen --port 7860 --enable-insecure-extension-access --opt-sdp-attention

pause