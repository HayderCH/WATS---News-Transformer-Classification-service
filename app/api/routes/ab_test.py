"""A/B testing routes for model comparison."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import time

from app.services.ab_testing import get_ab_testing_service
from app.services.classifier import classify_text
from app.core.security import require_api_key

router = APIRouter()


class ABTestRequest(BaseModel):
    """Request for A/B testing classification."""

    text: str
    user_id: Optional[str] = None  # For consistent variant assignment
    experiment_name: str = "ensemble_vs_transformer"


class ABTestResponse(BaseModel):
    """Response from A/B testing classification."""

    experiment_name: str
    assigned_variant: str
    prediction: Dict[str, Any]
    latency_ms: float
    experiment_results: Optional[Dict[str, Any]] = None


@router.post("/ab_test", response_model=ABTestResponse)
async def ab_test_classification(
    request: ABTestRequest, api_key: str = Depends(require_api_key)
) -> ABTestResponse:
    """
    A/B test classification with automatic variant assignment.

    WHY: A/B testing allows gradual rollout of new models while measuring
    real-world performance impact, preventing deploying broken models.
    """
    ab_service = get_ab_testing_service()

    # Assign user to control or treatment variant
    assigned_variant = ab_service.assign_variant(
        request.experiment_name, request.user_id
    )

    # WHY: Start timing here to measure end-to-end latency
    start_time = time.time()

    try:
        # Classify using assigned variant
        # WHY: This simulates real traffic routing in production
        prediction = classify_text(request.text, backend=assigned_variant)

        latency_ms = (time.time() - start_time) * 1000

        # Record the result for experiment analysis
        # WHY: Tracking latency and confidence allows measuring full UX impact
        ab_service.record_result(
            experiment_name=request.experiment_name,
            variant=assigned_variant,
            latency=latency_ms,
            confidence=prediction.get("confidence_score", 0.0),
            correct=None,  # Would need ground truth for accuracy tracking
        )

        # Get current experiment results
        results = ab_service.get_experiment_results(request.experiment_name)
        experiment_data = None
        if results:
            experiment_data = {
                "control_requests": results.control_requests,
                "treatment_requests": results.treatment_requests,
                "control_accuracy": results.control_accuracy,
                "treatment_accuracy": results.treatment_accuracy,
                "control_latency": results.control_latency,
                "treatment_latency": results.treatment_latency,
                "winner": results.winner,
            }

        return ABTestResponse(
            experiment_name=request.experiment_name,
            assigned_variant=assigned_variant,
            prediction=prediction,
            latency_ms=latency_ms,
            experiment_results=experiment_data,
        )

    except Exception as e:
        # WHY: Still record failed requests to track error rates
        latency_ms = (time.time() - start_time) * 1000
        ab_service.record_result(
            experiment_name=request.experiment_name,
            variant=assigned_variant,
            latency=latency_ms,
            confidence=0.0,
            correct=False,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab_test/results/{experiment_name}")
async def get_ab_test_results(
    experiment_name: str, api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    """
    Get current A/B test results.

    WHY: Real-time experiment monitoring allows data-driven decisions
    about model rollouts without waiting for experiment completion.
    """
    ab_service = get_ab_testing_service()
    results = ab_service.get_experiment_results(experiment_name)

    if not results:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "experiment_name": experiment_name,
        "control_requests": results.control_requests,
        "treatment_requests": results.treatment_requests,
        "control_accuracy": results.control_accuracy,
        "treatment_accuracy": results.treatment_accuracy,
        "control_latency": results.control_latency,
        "treatment_latency": results.treatment_latency,
        "winner": results.winner,
        "status": "completed" if results.winner else "running",
    }


@router.post("/ab_test/complete/{experiment_name}")
async def complete_ab_test(
    experiment_name: str, api_key: str = Depends(require_api_key)
) -> Dict[str, str]:
    """
    Complete an A/B test and determine winner.

    WHY: Automated winner determination prevents bias and ensures
    data-driven model selection for production deployment.
    """
    ab_service = get_ab_testing_service()
    winner = ab_service.complete_experiment(experiment_name)

    if not winner:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "experiment_name": experiment_name,
        "winner": winner,
        "message": f"Experiment completed. {winner} is the winning model.",
    }


@router.get("/ab_test/active")
async def get_active_experiments(
    api_key: str = Depends(require_api_key),
) -> Dict[str, Any]:
    """
    Get all active A/B experiments.

    WHY: Visibility into running experiments helps coordinate
    model development and deployment across teams.
    """
    ab_service = get_ab_testing_service()
    experiments = ab_service.get_active_experiments()

    return {
        "active_experiments": [
            {
                "name": name,
                "control_model": exp.control_model,
                "treatment_model": exp.treatment_model,
                "traffic_split": exp.traffic_split,
                "status": exp.status.value,
            }
            for name, exp in experiments.items()
        ]
    }
