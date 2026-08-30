#!/usr/bin/env python3
"""
Test script for image generation service
"""

import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.image_generator import NewsImageGenerator


def test_image_generation():
    """Test the image generation service"""
    print("🧪 Testing News Image Generator...")

    generator = NewsImageGenerator()

    # Test basic image generation
    test_prompt = "a beautiful sunset over mountains, high quality, detailed"
    print(f"📝 Generating test image with prompt: {test_prompt}")

    try:
        image_data = generator.generate_image(
            prompt=test_prompt,
            width=512,
            height=512,
            steps=20,  # Faster for testing
            cfg_scale=7.5,
        )

        if image_data:
            # Save test image
            output_path = "test_generated_image.png"
            with open(output_path, "wb") as f:
                f.write(image_data)
            print(f"✅ Test image saved as '{output_path}'")
            print(f"📊 Image size: {len(image_data)} bytes")
            return True
        else:
            print("❌ No image data returned")
            return False

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return False


def test_news_image_generation():
    """Test news-specific image generation"""
    print("\n📰 Testing News Article Image Generation...")

    generator = NewsImageGenerator()

    test_title = "Tesla Unveils New Electric Vehicle with Advanced Autopilot"
    test_summary = "Tesla CEO Elon Musk announced the new Model X with cutting-edge autonomous driving technology and extended range capabilities."

    print(f"📝 Generating news image for: {test_title}")

    try:
        image_data = generator.generate_news_image(test_title, test_summary)

        if image_data:
            output_path = "test_news_image.png"
            with open(output_path, "wb") as f:
                f.write(image_data)
            print(f"✅ News image saved as '{output_path}'")
            return True
        else:
            print("❌ Failed to generate news image")
            return False

    except Exception as e:
        print(f"❌ Error during news image generation: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Image Generation Tests")
    print("=" * 50)

    # Test basic generation
    basic_success = test_image_generation()

    # Test news-specific generation
    news_success = test_news_image_generation()

    print("\n" + "=" * 50)
    if basic_success and news_success:
        print("🎉 All tests passed! Your RTX 4060 image generation is working!")
        print("\n💡 Next steps:")
        print("1. Integrate into your news classification API")
        print("2. Add image endpoints to FastAPI routes")
        print("3. Test with real news articles")
    else:
        print("❌ Some tests failed. Make sure:")
        print("1. Stable Diffusion WebUI is running on http://localhost:7860")
        print("2. The SD 1.5 model is properly loaded")
        print("3. Your RTX 4060 GPU drivers are up to date")

    print("\n⚡ Performance Reminder:")
    print("• RTX 4060: 512x512 images in 3-5 seconds")
    print("• RTX 4060: 1024x1024 images in 15-25 seconds")
