"""Test A/B testing functionality."""

import asyncio
from app.api.routes.ab_test import ab_test_classification, ABTestRequest


async def test_ab_endpoint():
    """Test the A/B testing endpoint."""
    request = ABTestRequest(text="Apple announces new iPhone", user_id="test_user")
    try:
        response = await ab_test_classification(request, "test_key")
        print("✅ A/B test endpoint works!")
        print(f"   Assigned variant: {response.assigned_variant}")
        print(f'   Prediction: {response.prediction.get("top_category", "unknown")}')
        print(f"   Latency: {response.latency_ms:.1f}ms")
        if response.experiment_results:
            ctrl_reqs = response.experiment_results["control_requests"]
            treat_reqs = response.experiment_results["treatment_requests"]
            print(
                f"   Experiment data: {ctrl_reqs} control, "
                f"{treat_reqs} treatment requests"
            )
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_ab_endpoint())
