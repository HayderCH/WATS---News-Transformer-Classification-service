#!/usr/bin/env python3
"""
Simple Stable Diffusion Image Generator for RTX 4060
Uses Hugging Face diffusers library for local generation
"""

import torch
from diffusers import StableDiffusionPipeline
import os
import time
from pathlib import Path


class SimpleImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """
        Initialize the image generator

        Args:
            model_id: Hugging Face model ID (default: SD 1.5)
        """
        self.model_id = model_id
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"🚀 Initializing Simple Image Generator")
        print(f"📊 Device: {self.device}")
        if self.device == "cuda":
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
            )

    def load_model(self):
        """Load the Stable Diffusion model"""
        if self.pipe is not None:
            return

        print(f"📥 Loading model: {self.model_id}")
        start_time = time.time()

        # Use float16 for RTX 4060 (saves VRAM)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for speed
        )

        if self.device == "cuda":
            self.pipe = self.pipe.to(self.device)
            # Enable memory efficient attention
            self.pipe.enable_attention_slicing()

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, deformed, ugly",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ) -> str:
        """
        Generate a single image

        Args:
            prompt: Description of the image to generate
            negative_prompt: What to avoid in the image
            width: Image width (512 recommended for RTX 4060)
            height: Image height (512 recommended for RTX 4060)
            num_inference_steps: Number of denoising steps (20-30 good balance)
            guidance_scale: How closely to follow the prompt (7-12 works well)

        Returns:
            Path to the generated image file
        """

        if self.pipe is None:
            self.load_model()

        print(f"🎨 Generating image: {prompt[:50]}...")
        start_time = time.time()

        # Generate the image
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

        generation_time = time.time() - start_time

        # Save the image
        output_dir = Path("generated_images")
        output_dir.mkdir(exist_ok=True)

        timestamp = int(time.time())
        filename = f"image_{timestamp}.png"
        output_path = output_dir / filename

        image.save(output_path)
        print(f"✅ Image saved to: {output_path} (generated in {generation_time:.2f}s)")
        return str(output_path)

    def generate_news_image(self, title: str, summary: str = "") -> str:
        """
        Generate an image specifically for news articles

        Args:
            title: News article title
            summary: Optional article summary

        Returns:
            Path to generated image
        """

        # Create a news-focused prompt
        base_prompt = f"professional news illustration: {title}"

        if summary:
            # Add context from summary
            if any(
                word in summary.lower()
                for word in ["technology", "ai", "digital", "software"]
            ):
                base_prompt += ", technology theme, digital art"
            elif any(
                word in summary.lower()
                for word in ["politics", "government", "election"]
            ):
                base_prompt += ", political illustration, professional"
            elif any(
                word in summary.lower() for word in ["business", "economy", "market"]
            ):
                base_prompt += ", business theme, corporate"

        full_prompt = f"{base_prompt}, high quality, detailed, news magazine style"

        return self.generate_image(
            prompt=full_prompt,
            width=512,
            height=512,
            num_inference_steps=25,
            guidance_scale=8.0,
        )


# Example usage
if __name__ == "__main__":
    print("📰 News Image Generator Demo")
    print("=" * 40)

    generator = SimpleImageGenerator()

    # Test basic generation
    print("\n🎨 Testing basic image generation...")
    try:
        image_path = generator.generate_image(
            "a beautiful mountain landscape at sunset, high quality, detailed"
        )
        print(f"✅ Basic generation successful: {image_path}")
    except Exception as e:
        print(f"❌ Basic generation failed: {e}")

    # Test news-specific generation
    print("\n📰 Testing news image generation...")
    try:
        news_image_path = generator.generate_news_image(
            "Tesla Unveils New Autonomous Driving Technology",
            "Tesla CEO Elon Musk announced breakthrough advances in self-driving car technology during the AI Safety Summit.",
        )
        print(f"✅ News generation successful: {news_image_path}")
    except Exception as e:
        print(f"❌ News generation failed: {e}")

    print("\n" + "=" * 40)
    print("🎉 Demo complete!")
    print("💡 Your RTX 4060 can generate images in 3-5 seconds each!")
    print(
        "🔧 Integrate this into your news classification API for automatic image generation."
    )
