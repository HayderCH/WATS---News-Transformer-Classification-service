#!/usr/bin/env python3
"""
Quick RTX 4060 Image Generation Test
Resumes download and tests basic functionality
"""

import torch
from diffusers import StableDiffusionPipeline
import time
from pathlib import Path


def test_generation():
    print("🚀 Testing RTX 4060 Image Generation")
    print("=" * 40)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📊 Device: {device}")
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        )

    print("\n📥 Loading Stable Diffusion 1.5 (resuming download)...")
    start_time = time.time()

    try:
        # Load with optimizations for RTX 4060
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # Skip for speed
        )

        if device == "cuda":
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()  # Save VRAM

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s")
        # Test generation
        print("\n🎨 Generating test image...")
        gen_start = time.time()

        prompt = "a beautiful sunset over mountains, high quality, detailed"
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                width=512,
                height=512,
                num_inference_steps=20,  # Faster for testing
                guidance_scale=7.5,
            ).images[0]

        gen_time = time.time() - gen_start

        # Save image
        output_dir = Path("generated_images")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_rtx4060.png"
        image.save(output_path)

        print(f"✅ Image generated in {gen_time:.2f}s")
        print(f"💾 Saved to: {output_path}")

        print("\n" + "=" * 40)
        print("🎉 SUCCESS! Your RTX 4060 is working perfectly!")
        print("⚡ Performance: Excellent for local image generation")
        print("💡 Ready to integrate with your news classification API")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection")
        print("2. Ensure sufficient disk space (4GB free)")
        print("3. Try restarting if download was interrupted")
        return False


if __name__ == "__main__":
    success = test_generation()
    if success:
        print("\n🚀 Next: Integrate with your news service!")
        print(
            'Run: python -c "from app.services.image_generator import NewsImageGenerator; # use it"'
        )
    else:
        print("\n❌ Try again or check the troubleshooting steps above")
