# AI Image Generation for News Articles

## Overview

The News Topic Intelligence Service includes AI-powered image generation capabilities using Stable Diffusion 1.5 running on RTX 4060 GPU acceleration. This feature generates context-aware visualizations for news articles in 3-5 seconds, enhancing content engagement and providing visual context for text-based news.

### Key Features

- **GPU-Accelerated Generation**: RTX 4060 with CUDA optimization for fast image creation
- **Context-Aware Prompts**: Automatic prompt engineering based on article title, category, and content
- **Multiple Generation Modes**: Custom prompts, news article mode, and category-based generation
- **REST API Integration**: FastAPI endpoints for programmatic image generation
- **Streamlit Dashboard**: Interactive UI for real-time generation and visualization
- **Production-Ready**: Error handling, logging, and performance monitoring

## Architecture

```
News Article / Custom Prompt
              ↓
     Prompt Engineering
     (Category + Content Analysis)
              ↓
   RTX 4060 GPU Acceleration
   (Stable Diffusion 1.5)
              ↓
   Image Post-Processing
   (Resize, Format, Save)
              ↓
   File Path Response
   (generated_images/*.png)
```

## Setup and Installation

### Prerequisites

- **GPU Requirements**: NVIDIA RTX 4060 or higher with 8GB+ VRAM
- **CUDA**: CUDA 11.8+ compatible with PyTorch
- **Python**: 3.11+
- **Dependencies**: All packages in `requirements.txt`

### GPU Setup

1. **Verify GPU Availability**:

```bash
python scripts/check_gpu.py
```

2. **Install CUDA (if needed)**:

   - Download CUDA 11.8 from NVIDIA website
   - Install following NVIDIA instructions
   - Restart system

3. **Test PyTorch CUDA**:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Model Setup

The system uses Hugging Face's `runwayml/stable-diffusion-v1-5` model. It's automatically downloaded on first use, but you can pre-download it:

```bash
# Pre-download the model
python -c "from diffusers import StableDiffusionPipeline; pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)"
```

## API Usage

### Endpoints

#### Health Check

```bash
GET /images/status
```

Returns GPU availability and service health.

**Response**:

```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4060",
  "vram_gb": 8.0
}
```

#### Custom Image Generation

```bash
POST /images/generate-image
Content-Type: application/json
X-API-Key: your-api-key

{
  "prompt": "A futuristic city skyline at sunset",
  "negative_prompt": "blurry, low quality",
  "width": 512,
  "height": 512
}
```

**Response**:

```json
{
  "image_path": "generated_images/image_20251010_143022.png",
  "generation_time": 3.2,
  "prompt_used": "A futuristic city skyline at sunset"
}
```

#### News Article Image Generation

```bash
POST /images/generate-news-image
Content-Type: application/json
X-API-Key: your-api-key

{
  "title": "Apple Announces New iPhone",
  "category": "Technology",
  "summary": "Apple unveiled the latest iPhone with revolutionary features including advanced AI capabilities and improved camera system."
}
```

**Response**:

```json
{
  "image_path": "generated_images/news_apple_iphone_20251010_143025.png",
  "generation_time": 4.1,
  "prompt_used": "Technology news: Apple Announces New iPhone - Apple unveiled the latest iPhone with revolutionary features including advanced AI capabilities and improved camera system. Create a professional news visualization showing modern technology innovation."
}
```

### Error Handling

**GPU Not Available**:

```json
{
  "detail": "GPU not available for image generation",
  "error_code": "GPU_UNAVAILABLE"
}
```

**Invalid Prompt**:

```json
{
  "detail": "Prompt cannot be empty",
  "error_code": "INVALID_PROMPT"
}
```

## Streamlit Dashboard Integration

### Images Tab Features

1. **Custom Generation Mode**:

   - Free-form prompt input
   - Real-time generation with progress bar
   - Image preview and download

2. **News Article Mode**:

   - Title, category, and summary inputs
   - Automatic prompt engineering
   - Context-aware image generation

3. **Category-Based Mode**:
   - Pre-configured prompts for news categories
   - Quick generation for common topics

### Usage Example

```python
import streamlit as st
from app.services.image_generator import NewsImageGenerator

# Initialize generator
generator = NewsImageGenerator()

# Generate image
image_path = generator.generate_news_image(
    title="Breaking: Major Scientific Discovery",
    category="Science",
    summary="Scientists announce groundbreaking research..."
)

# Display in Streamlit
st.image(image_path, caption="Generated News Image")
```

## Performance Optimization

### GPU Memory Management

- **FP16 Precision**: Uses `torch.float16` to reduce VRAM usage (~4GB for 512x512 images)
- **Model Caching**: Model stays loaded in memory for subsequent generations
- **Batch Processing**: Single image generation to maintain quality

### Generation Parameters

- **Resolution**: 512x512 (optimal balance of quality vs speed)
- **Inference Steps**: 20-30 steps (default: 20 for speed)
- **Guidance Scale**: 7.5 (balanced creativity vs prompt adherence)

### Performance Metrics

- **Generation Time**: 3-5 seconds per image
- **VRAM Usage**: ~4GB during generation
- **CPU Usage**: Minimal (GPU handles computation)
- **Concurrent Requests**: Single GPU instance (scale with multiple GPUs)

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. Reduce image resolution: `width=384, height=384`
2. Use fewer inference steps: `num_inference_steps=15`
3. Restart the service to clear GPU memory
4. Check for other GPU processes

#### Model Download Issues

**Error**: `ConnectionError` during model loading

**Solutions**:

1. Check internet connection
2. Use VPN if needed (Hugging Face may be blocked)
3. Pre-download model manually
4. Use local model cache

#### GPU Not Detected

**Error**: `GPU not available`

**Solutions**:

1. Verify CUDA installation: `nvcc --version`
2. Check GPU drivers: `nvidia-smi`
3. Install PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
4. Restart system

### Debug Commands

```bash
# Check GPU status
python scripts/check_gpu.py

# Test image generation
python simple_image_generator.py

# Monitor GPU usage
nvidia-smi --loop=1

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"
```

## Configuration

### Environment Variables

```bash
# Image generation settings
IMAGE_WIDTH=512
IMAGE_HEIGHT=512
IMAGE_STEPS=20
IMAGE_GUIDANCE_SCALE=7.5

# GPU settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Storage settings
GENERATED_IMAGES_DIR=generated_images
MAX_IMAGES_PER_DAY=1000
```

### Model Configuration

```python
# In image_generator.py
model_config = {
    "model_id": "runwayml/stable-diffusion-v1-5",
    "torch_dtype": torch.float16,
    "safety_checker": None,  # Disabled for performance
    "use_auth_token": None
}
```

## Security Considerations

### Input Validation

- Prompt length limits (max 500 characters)
- Content filtering for inappropriate prompts
- Rate limiting per API key

### Output Safety

- Generated images stored securely
- File path randomization
- Access control via API keys

### Resource Protection

- GPU memory monitoring
- Request queuing for high load
- Automatic cleanup of old images

## Integration Examples

### Python Client

```python
import requests

def generate_news_image(title, category, summary):
    response = requests.post(
        "http://localhost:8001/images/generate-news-image",
        json={
            "title": title,
            "category": category,
            "summary": summary
        },
        headers={"X-API-Key": "your-key"}
    )
    return response.json()

# Usage
result = generate_news_image(
    "Tesla Stock Surges",
    "Business",
    "Tesla shares jumped 10% following positive earnings..."
)
print(f"Image generated: {result['image_path']}")
```

### Batch Processing

```python
import asyncio
from app.services.image_generator import NewsImageGenerator

async def batch_generate_images(articles):
    generator = NewsImageGenerator()
    tasks = []

    for article in articles:
        task = asyncio.create_task(
            generator.generate_news_image_async(
                article['title'],
                article['category'],
                article['summary']
            )
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results
```

## Future Enhancements

- **Multi-Model Support**: Integration with DALL-E, Midjourney APIs
- **Image Editing**: Inpainting and outpainting capabilities
- **Style Transfer**: Apply news-specific visual styles
- **Batch Generation**: Parallel processing on multiple GPUs
- **Quality Metrics**: Automated image quality assessment
- **Caching**: Smart caching of similar prompts

## Support

For issues with image generation:

1. Check the troubleshooting section above
2. Verify GPU and CUDA setup
3. Review application logs
4. Test with simple prompts first
5. Check GitHub issues for known problems

## References

- [Stable Diffusion Documentation](https://huggingface.co/docs/diffusers)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [RTX 4060 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
