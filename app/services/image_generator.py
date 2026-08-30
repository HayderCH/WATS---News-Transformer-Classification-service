"""
Image Generation Service for News Articles
Uses Hugging Face diffusers library for local RTX 4060 generation
"""

import torch
from diffusers import StableDiffusionPipeline
import os
import time
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class NewsImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """
        Initialize the image generator

        Args:
            model_id: Hugging Face model ID (default: SD 1.5)
        """
        self.model_id = model_id
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"🚀 Initializing News Image Generator")
        logger.info(f"📊 Device: {self.device}")
        if self.device == "cuda":
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
            )

    def load_model(self):
        """Load the Stable Diffusion model"""
        if self.pipe is not None:
            return

        logger.info(f"📥 Loading model: {self.model_id}")
        start_time = time.time()

        # Use float16 for RTX 4060 (saves VRAM)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for speed
        )

        if self.device == "cuda":
            self.pipe = self.pipe.to(self.device)

        # Enable attention slicing for lower VRAM usage
        self.pipe.enable_attention_slicing()

        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time:.2f}s")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, deformed, ugly",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ) -> Optional[str]:
        """
        Generate an image and save it to file

        Args:
            prompt: Main prompt describing the image
            negative_prompt: What to avoid in the image
            width: Image width (512 recommended for RTX 4060)
            height: Image height (512 recommended for RTX 4060)
            num_inference_steps: Number of denoising steps (20-30 good balance)
            guidance_scale: Classifier-free guidance scale (7-12 works well)

        Returns:
            Path to generated image file if successful, None if failed
        """
        try:
            self.load_model()

            logger.info(f"🎨 Generating image with prompt: {prompt[:50]}...")
            start_time = time.time()

            # Generate the image
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

            generation_time = time.time() - start_time
            logger.info(f"✅ Image generated in {generation_time:.2f}s")

            # Save the image
            os.makedirs("generated_images", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_path = f"generated_images/image_{timestamp}.png"

            result.images[0].save(image_path)
            logger.info(f"💾 Image saved to: {image_path}")

            return image_path

        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            return None

    def generate_news_image(
        self, article_title: str, article_summary: str = ""
    ) -> Optional[str]:
        """
        Generate an image based on news article content

        Args:
            article_title: The article headline
            article_summary: Optional article summary for better prompts

        Returns:
            Path to generated image file
        """

        # Create a descriptive prompt from the article
        base_prompt = f"professional news illustration of: {article_title}"

        if article_summary:
            # Extract key visual elements from summary
            visual_keywords = []
            if any(
                word in article_summary.lower()
                for word in ["president", "leader", "politician"]
            ):
                visual_keywords.append("political figure")
            if any(
                word in article_summary.lower()
                for word in ["technology", "ai", "digital"]
            ):
                visual_keywords.append("technology theme")
            if any(
                word in article_summary.lower()
                for word in ["economy", "market", "business"]
            ):
                visual_keywords.append("business professional")

            if visual_keywords:
                base_prompt += f", {', '.join(visual_keywords)}"

        # Add quality enhancers
        full_prompt = f"{base_prompt}, high quality, professional, detailed, news illustration style"

        return self.generate_image(
            prompt=full_prompt,
            width=512,  # Optimal for RTX 4060 speed
            height=512,
            num_inference_steps=25,  # Good quality/speed balance
            guidance_scale=8.0,
        )
