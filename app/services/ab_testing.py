"""A/B testing service for model comparison and gradual rollouts."""

import random
import time
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B experiment."""

    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""

    name: str
    control_model: str  # e.g., "ensemble"
    treatment_model: str  # e.g., "transformer"
    traffic_split: float  # 0.5 = 50% control, 50% treatment
    status: ExperimentStatus = ExperimentStatus.ACTIVE
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()


@dataclass
class ExperimentResult:
    """Results from an A/B experiment."""

    experiment_name: str
    control_requests: int = 0
    treatment_requests: int = 0
    control_accuracy: float = 0.0
    treatment_accuracy: float = 0.0
    control_latency: float = 0.0
    treatment_latency: float = 0.0
    winner: Optional[str] = None


class ABTestingService:
    """
    A/B testing service for comparing model performance.

    WHY: A/B testing allows gradual rollout of new models while measuring
    real-world performance impact. This prevents deploying broken models
    and provides data-driven decisions for model updates.
    """

    def __init__(self):
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, ExperimentResult] = {}
        self.request_history: Dict[str, list] = {}

        # Initialize with a default experiment
        self._create_default_experiment()

    def _create_default_experiment(self):
        """Create a default A/B experiment comparing ensemble vs transformer."""
        experiment = ExperimentConfig(
            name="ensemble_vs_transformer",
            control_model="ensemble",  # Current production model
            treatment_model="transformer",  # New model to test
            traffic_split=0.5,  # 50/50 split
            status=ExperimentStatus.ACTIVE,
        )
        self.experiments[experiment.name] = experiment
        self.results[experiment.name] = ExperimentResult(experiment.name)
        self.request_history[experiment.name] = []

        logger.info("Created default A/B experiment: %s", experiment.name)

    def assign_variant(
        self, experiment_name: str, user_id: Optional[str] = None
    ) -> str:
        """
        Assign a user to control or treatment variant.

        WHY: Consistent assignment ensures each user always gets the same
        variant, preventing cross-contamination of results.
        """
        if experiment_name not in self.experiments:
            return "ensemble"  # Default fallback

        experiment = self.experiments[experiment_name]

        if experiment.status != ExperimentStatus.ACTIVE:
            return experiment.control_model

        # Use user_id for consistent assignment, fallback to random
        if user_id:
            # Simple hash-based assignment for consistency
            hash_value = hash(user_id + experiment_name) % 100
            threshold = int(experiment.traffic_split * 100)
            variant = (
                experiment.control_model
                if hash_value < threshold
                else experiment.treatment_model
            )
        else:
            # Random assignment for anonymous users
            variant = (
                experiment.control_model
                if random.random() < experiment.traffic_split
                else experiment.treatment_model
            )

        return variant

    def record_result(
        self,
        experiment_name: str,
        variant: str,
        latency: float,
        confidence: float,
        correct: Optional[bool] = None,
    ):
        """
        Record the result of an A/B test request.

        WHY: Tracking both latency and accuracy allows us to measure
        the full impact of model changes on user experience.
        """
        if experiment_name not in self.results:
            return

        result = self.results[experiment_name]

        # Record request counts
        if variant == self.experiments[experiment_name].control_model:
            result.control_requests += 1
            result.control_latency = (
                (result.control_latency * (result.control_requests - 1)) + latency
            ) / result.control_requests
            if correct is not None:
                result.control_accuracy = (
                    (result.control_accuracy * (result.control_requests - 1))
                    + (1 if correct else 0)
                ) / result.control_requests
        else:
            result.treatment_requests += 1
            result.treatment_latency = (
                (result.treatment_latency * (result.treatment_requests - 1)) + latency
            ) / result.treatment_requests
            if correct is not None:
                result.treatment_accuracy = (
                    (result.treatment_accuracy * (result.treatment_requests - 1))
                    + (1 if correct else 0)
                ) / result.treatment_requests

        # Store request history for analysis
        self.request_history[experiment_name].append(
            {
                "variant": variant,
                "latency": latency,
                "confidence": confidence,
                "timestamp": time.time(),
            }
        )

        # Keep only last 1000 requests to prevent memory issues
        max_history = 1000
        if len(self.request_history[experiment_name]) > max_history:
            self.request_history[experiment_name] = self.request_history[
                experiment_name
            ][-max_history:]

    def get_experiment_results(
        self, experiment_name: str
    ) -> Optional[ExperimentResult]:
        """Get current results for an experiment."""
        return self.results.get(experiment_name)

    def complete_experiment(self, experiment_name: str) -> Optional[str]:
        """
        Complete an experiment and determine the winner.

        WHY: Automated winner determination prevents bias and ensures
        data-driven model selection.
        """
        if experiment_name not in self.experiments:
            return None

        experiment = self.experiments[experiment_name]
        result = self.results[experiment_name]

        # Simple winner logic: better accuracy, then latency
        if result.control_accuracy > result.treatment_accuracy:
            winner = experiment.control_model
        elif result.treatment_accuracy > result.control_accuracy:
            winner = experiment.treatment_model
        else:
            # Tie-breaker: lower latency wins
            winner = (
                experiment.control_model
                if result.control_latency <= result.treatment_latency
                else experiment.treatment_model
            )

        result.winner = winner
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = time.time()

        logger.info("Experiment %s completed. Winner: %s", experiment_name, winner)
        return winner

    def get_active_experiments(self) -> Dict[str, ExperimentConfig]:
        """Get all active experiments."""
        return {
            name: exp
            for name, exp in self.experiments.items()
            if exp.status == ExperimentStatus.ACTIVE
        }


# Global A/B testing service instance
_ab_service = None


def get_ab_testing_service() -> ABTestingService:
    """Get global A/B testing service instance."""
    global _ab_service
    if _ab_service is None:
        _ab_service = ABTestingService()
    return _ab_service
