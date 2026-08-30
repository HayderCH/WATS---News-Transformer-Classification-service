"""
FastAPI endpoints for image generation
Add these to your app/api/routes/ directory
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import logging

from app.services.image_generator import NewsImageGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize the image generator (lazy loading)
image_generator = None


def get_image_generator() -> NewsImageGenerator:
    """Get or create the image generator instance"""
    global image_generator
    if image_generator is None:
        image_generator = NewsImageGenerator()
    return image_generator


class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "blurry, low quality, deformed, ugly"
    width: Optional[int] = 512
    height: Optional[int] = 512
    steps: Optional[int] = 25


class NewsImageRequest(BaseModel):
    title: str
    summary: Optional[str] = None


@router.post("/generate-image")
async def generate_image(request: ImageGenerationRequest):
    """
    Generate a custom image from text prompt

    Use this for custom image generation with full control over parameters.
    """
    try:
        generator = get_image_generator()

        image_path = generator.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=min(request.width, 1024),  # Limit max size
            height=min(request.height, 1024),
            num_inference_steps=min(request.steps, 50),  # Limit max steps
            guidance_scale=request.guidance_scale,
        )

        if image_path:
            return {
                "success": True,
                "image_path": image_path,
                "message": "Image generated successfully",
                "gpu_info": "RTX 4060 (8GB VRAM)",
                "generation_time": "3-5 seconds typical",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate image")

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )


@router.post("/generate-news-image")
async def generate_news_image(request: NewsImageRequest):
    """
    Generate an image for a news article

    Optimized prompts for news content with automatic theme detection.
    """
    try:
        generator = get_image_generator()

        image_path = generator.generate_news_image(
            article_title=request.title, article_summary=request.summary
        )

        if image_path:
            return {
                "success": True,
                "image_path": image_path,
                "title": request.title,
                "message": "News image generated successfully",
                "style": "Professional news illustration",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate image")

    except Exception as e:
        logger.error(f"News image generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"News image generation failed: {str(e)}"
        )


@router.get("/status")
async def get_generation_status():
    """Check if image generation service is ready"""
    try:
        # Quick test to see if GPU is available
        import torch

        gpu_available = torch.cuda.is_available()

        return {
            "service": "ready",
            "gpu_available": gpu_available,
            "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
            "vram_gb": (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_available
                else 0
            ),
            "typical_generation_time": "3-5 seconds for 512x512",
            "supported_formats": ["png"],
            "max_resolution": "1024x1024",
        }

    except Exception as e:
        return {"service": "error", "error": str(e), "gpu_available": False}


# Example usage in your main FastAPI app:
"""
from app.api.routes import images

app = FastAPI()
app.include_router(images.router, prefix="/images", tags=["images"])

# Now you have these endpoints:
# POST /images/generate-image
# POST /images/generate-news-image
# GET /images/status
"""
