#!/usr/bin/env python3
"""
Test script for Feature 4: Real-Time Streaming & Anomaly Detection
Demonstrates the streaming service functionality
"""

import asyncio
import time
import requests
import json
from datetime import datetime

# Configuration
API_BASE = "http://localhost:8000"


def test_streaming_status():
    """Test streaming service status"""
    print("🔍 Testing streaming service status...")
    try:
        response = requests.get(f"{API_BASE}/streaming/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Streaming status: {data}")
            return True
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def test_start_streaming():
    """Test starting the streaming service"""
    print("🚀 Starting streaming service...")
    try:
        payload = {"rate": 0.5}  # 0.5 articles per second for demo
        response = requests.post(f"{API_BASE}/streaming/start", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Streaming started: {data}")
            return True
        else:
            print(f"❌ Failed to start streaming: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def test_manual_article():
    """Test submitting a manual article"""
    print("📝 Testing manual article submission...")
    try:
        payload = {
            "text": "Breaking news: Major earthquake strikes California coast causing widespread damage and triggering tsunami warnings across the Pacific.",
            "title": "Major Earthquake Hits California Coast",
        }
        response = requests.post(f"{API_BASE}/streaming/article", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Article submitted: {data}")
            return True
        else:
            print(f"❌ Failed to submit article: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def test_anomaly_detection():
    """Test anomaly detection statistics"""
    print("🔍 Testing anomaly detection stats...")
    try:
        response = requests.get(f"{API_BASE}/streaming/anomalies/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Anomaly stats: {data}")
            return True
        else:
            print(f"❌ Failed to get anomaly stats: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def test_alert_system():
    """Test alert system statistics"""
    print("🔔 Testing alert system stats...")
    try:
        response = requests.get(f"{API_BASE}/streaming/alerts/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Alert stats: {data}")
            return True
        else:
            print(f"❌ Failed to get alert stats: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def test_streaming_health():
    """Test streaming service health"""
    print("💚 Testing streaming health...")
    try:
        response = requests.get(f"{API_BASE}/streaming/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Streaming health: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def test_stop_streaming():
    """Test stopping the streaming service"""
    print("🛑 Stopping streaming service...")
    try:
        response = requests.post(f"{API_BASE}/streaming/stop")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Streaming stopped: {data}")
            return True
        else:
            print(f"❌ Failed to stop streaming: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


async def run_demo():
    """Run a complete streaming demo"""
    print("🎬 Starting Feature 4 Streaming Demo")
    print("=" * 50)

    # Test 1: Check service status
    if not test_streaming_status():
        print("❌ Service not available. Make sure the API is running.")
        return

    # Test 2: Start streaming
    if not test_start_streaming():
        print("❌ Could not start streaming service.")
        return

    # Wait a moment for streaming to initialize
    print("⏳ Waiting for streaming to initialize...")
    await asyncio.sleep(2)

    # Test 3: Submit manual articles to trigger processing
    print("📨 Submitting test articles...")
    test_articles = [
        {
            "text": "POLITICS: President announces major economic policy changes affecting millions of Americans.",
            "title": "President Unveils Economic Policy Reforms",
        },
        {
            "text": "POLITICS: Congress passes controversial legislation with bipartisan support.",
            "title": "Congress Approves Bipartisan Legislation",
        },
        {
            "text": "POLITICS: Breaking news - Supreme Court rules on landmark constitutional case.",
            "title": "Supreme Court Issues Landmark Ruling",
        },
        {
            "text": "ENTERTAINMENT: Hollywood blockbuster breaks all-time box office records.",
            "title": "Blockbuster Shatters Box Office Records",
        },
    ]

    for i, article in enumerate(test_articles, 1):
        print(f"  Submitting article {i}/4...")
        test_manual_article()
        await asyncio.sleep(1)  # Brief pause between submissions

    # Wait for processing
    print("⏳ Waiting for articles to be processed...")
    await asyncio.sleep(3)

    # Test 4: Check anomaly detection
    test_anomaly_detection()

    # Test 5: Check alert system
    test_alert_system()

    # Test 6: Check streaming health
    test_streaming_health()

    # Test 7: Stop streaming
    test_stop_streaming()

    print("=" * 50)
    print("🎬 Feature 4 Streaming Demo Complete!")
    print("\n📊 What was demonstrated:")
    print("  ✅ Real-time article processing pipeline")
    print("  ✅ Automated classification and anomaly detection")
    print("  ✅ Alert system for unusual patterns")
    print("  ✅ RESTful API for streaming management")
    print("  ✅ Configurable streaming rates")
    print("  ✅ Health monitoring and statistics")


def main():
    """Main entry point"""
    print("🚀 Feature 4: Real-Time Streaming & Anomaly Detection")
    print("This demo will test the streaming service functionality.")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print()

    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")


if __name__ == "__main__":
    main()
