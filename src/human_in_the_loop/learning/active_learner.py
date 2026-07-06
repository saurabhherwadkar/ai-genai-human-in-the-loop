"""
Active learning: improve the system from human corrections.

Collects human feedback and tracks patterns to identify where
the AI model is weakest, guiding future improvement. Implements
uncertainty sampling, drift detection, and systematic error
pattern analysis to prioritize what the model should learn next.
"""

import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any

from human_in_the_loop.config.settings import get_settings
from human_in_the_loop.models.schemas import FeedbackRecord
from human_in_the_loop.utils.logger import get_logger

logger = get_logger(__name__)


class UncertaintySampler:
    """
    Identifies which items would be most informative for model improvement.

    Implements multiple uncertainty sampling strategies:
    - Least confidence: items where the model is least sure
    - Margin sampling: items where top-2 predictions are close
    - Entropy-based: items with highest prediction entropy
    """

    def __init__(self, strategy: str = "least_confidence") -> None:
        self._strategy = strategy
        self._confidence_history: list[tuple[str, float]] = []

    def record_confidence(self, item_id: str, confidence: float) -> None:
        """Record a confidence score for analysis."""
        self._confidence_history.append((item_id, confidence))

    def get_most_uncertain(self, n: int = 10) -> list[tuple[str, float]]:
        """Get the N most uncertain items (closest to 0.5 confidence)."""
        scored = [
            (item_id, self._uncertainty_score(conf))
            for item_id, conf in self._confidence_history
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def get_uncertainty_distribution(self) -> dict[str, int]:
        """Bin confidence scores to show the uncertainty distribution."""
        bins = {"very_low (0-0.2)": 0, "low (0.2-0.4)": 0, "medium (0.4-0.6)": 0,
                "high (0.6-0.8)": 0, "very_high (0.8-1.0)": 0}
        for _, conf in self._confidence_history:
            if conf < 0.2:
                bins["very_low (0-0.2)"] += 1
            elif conf < 0.4:
                bins["low (0.2-0.4)"] += 1
            elif conf < 0.6:
                bins["medium (0.4-0.6)"] += 1
            elif conf < 0.8:
                bins["high (0.6-0.8)"] += 1
            else:
                bins["very_high (0.8-1.0)"] += 1
        return bins

    def _uncertainty_score(self, confidence: float) -> float:
        """Compute uncertainty using selected strategy."""
        if self._strategy == "least_confidence":
            return 1.0 - confidence
        elif self._strategy == "entropy":
            if confidence <= 0 or confidence >= 1:
                return 0.0
            return -(confidence * math.log2(confidence) + (1 - confidence) * math.log2(1 - confidence))
        else:  # margin
            return 1.0 - abs(2 * confidence - 1)


class DriftDetector:
    """
    Detects distribution drift in model confidence and error rates.

    Monitors sliding windows to identify when the model's behavior
    is shifting, which may indicate data drift or model degradation.
    """

    def __init__(self, window_size: int = 50, drift_threshold: float = 0.15) -> None:
        self._window_size = window_size
        self._drift_threshold = drift_threshold
        self._confidence_stream: list[float] = []
        self._error_stream: list[bool] = []
        self._drift_events: list[dict[str, Any]] = []

    def add_observation(self, confidence: float, was_error: bool) -> dict[str, Any] | None:
        """
        Add an observation and check for drift.

        Returns drift alert if detected, None otherwise.
        """
        self._confidence_stream.append(confidence)
        self._error_stream.append(was_error)

        if len(self._confidence_stream) < self._window_size * 2:
            return None

        return self._check_drift()

    def _check_drift(self) -> dict[str, Any] | None:
        """Compare recent window against baseline window for drift."""
        n = self._window_size
        baseline_conf = self._confidence_stream[-2 * n: -n]
        recent_conf = self._confidence_stream[-n:]
        baseline_err = self._error_stream[-2 * n: -n]
        recent_err = self._error_stream[-n:]

        # Mean confidence drift
        baseline_mean = sum(baseline_conf) / len(baseline_conf)
        recent_mean = sum(recent_conf) / len(recent_conf)
        confidence_drift = abs(recent_mean - baseline_mean)

        # Error rate drift
        baseline_error_rate = sum(baseline_err) / len(baseline_err)
        recent_error_rate = sum(recent_err) / len(recent_err)
        error_drift = recent_error_rate - baseline_error_rate

        if confidence_drift > self._drift_threshold or error_drift > self._drift_threshold:
            event = {
                "type": "drift_detected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "confidence_drift": round(confidence_drift, 4),
                "error_rate_drift": round(error_drift, 4),
                "baseline_mean_confidence": round(baseline_mean, 4),
                "recent_mean_confidence": round(recent_mean, 4),
                "baseline_error_rate": round(baseline_error_rate, 4),
                "recent_error_rate": round(recent_error_rate, 4),
            }
            self._drift_events.append(event)
            logger.info("drift_detected", **event)
            return event

        return None

    def get_drift_history(self) -> list[dict[str, Any]]:
        """Get all detected drift events."""
        return self._drift_events

    def get_current_stats(self) -> dict[str, Any]:
        """Get current drift monitoring statistics."""
        if not self._confidence_stream:
            return {"status": "no_data"}

        n = min(self._window_size, len(self._confidence_stream))
        recent = self._confidence_stream[-n:]
        recent_errors = self._error_stream[-n:]

        return {
            "observations": len(self._confidence_stream),
            "recent_mean_confidence": round(sum(recent) / len(recent), 4),
            "recent_error_rate": round(sum(recent_errors) / len(recent_errors), 4) if recent_errors else 0,
            "drift_events_total": len(self._drift_events),
        }


class ActiveLearner:
    """
    Learns from human corrections to improve AI accuracy over time.

    Enhanced with:
    - Uncertainty sampling to identify most informative items for review
    - Drift detection to alert when model behavior changes
    - Systematic error pattern analysis with category tracking
    - Sliding window accuracy for trend monitoring
    - Correction clustering to identify systematic failures
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._feedback: list[FeedbackRecord] = []
        self._uncertainty_sampler = UncertaintySampler()
        self._drift_detector = DriftDetector()
        self._error_categories: dict[str, list[FeedbackRecord]] = defaultdict(list)
        self._correction_pairs: list[dict[str, Any]] = []
        self._retrain_events: list[dict[str, Any]] = []

    def record_feedback(self, feedback: FeedbackRecord) -> dict[str, Any] | None:
        """
        Record human feedback for learning.

        Returns a drift alert if drift is detected, None otherwise.
        """
        self._feedback.append(feedback)
        logger.info("feedback_recorded", was_correct=feedback.was_correct, type=feedback.feedback_type)

        # Track in uncertainty sampler
        self._uncertainty_sampler.record_confidence(feedback.item_id, 0.5)

        # Track drift
        drift_alert = self._drift_detector.add_observation(
            confidence=0.5,  # Will be enhanced when confidence is available
            was_error=not feedback.was_correct,
        )

        # Categorize errors
        if not feedback.was_correct:
            self._error_categories[feedback.feedback_type].append(feedback)
            # Store correction pair for training
            if feedback.corrected_output is not None:
                self._correction_pairs.append({
                    "item_id": feedback.item_id,
                    "original": feedback.original_output,
                    "corrected": feedback.corrected_output,
                    "category": feedback.feedback_type,
                    "timestamp": feedback.timestamp.isoformat() if feedback.timestamp else None,
                })

        return drift_alert

    def record_feedback_with_confidence(self, feedback: FeedbackRecord, confidence: float) -> dict[str, Any] | None:
        """Record feedback along with the original confidence score for better drift tracking."""
        self._feedback.append(feedback)
        self._uncertainty_sampler.record_confidence(feedback.item_id, confidence)

        drift_alert = self._drift_detector.add_observation(
            confidence=confidence,
            was_error=not feedback.was_correct,
        )

        if not feedback.was_correct:
            self._error_categories[feedback.feedback_type].append(feedback)
            if feedback.corrected_output is not None:
                self._correction_pairs.append({
                    "item_id": feedback.item_id,
                    "original": feedback.original_output,
                    "corrected": feedback.corrected_output,
                    "category": feedback.feedback_type,
                    "confidence": confidence,
                    "timestamp": feedback.timestamp.isoformat() if feedback.timestamp else None,
                })

        logger.info("feedback_with_confidence_recorded",
                    was_correct=feedback.was_correct, confidence=confidence)
        return drift_alert

    def get_accuracy(self) -> float:
        """Calculate current accuracy from feedback."""
        if not self._feedback:
            return 0.0
        correct = sum(1 for f in self._feedback if f.was_correct)
        return correct / len(self._feedback)

    def get_sliding_accuracy(self, window: int = 50) -> dict[str, float]:
        """
        Calculate accuracy over a sliding window to detect trends.

        Returns both overall and recent accuracy for comparison.
        """
        overall = self.get_accuracy()
        if len(self._feedback) < window:
            return {"overall": overall, "recent": overall, "trend": "stable"}

        recent_feedback = self._feedback[-window:]
        recent_correct = sum(1 for f in recent_feedback if f.was_correct)
        recent_accuracy = recent_correct / window

        # Determine trend
        if recent_accuracy > overall + 0.05:
            trend = "improving"
        elif recent_accuracy < overall - 0.05:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "overall": round(overall, 4),
            "recent": round(recent_accuracy, 4),
            "trend": trend,
            "window_size": window,
        }

    def get_error_patterns(self) -> dict:
        """
        Identify common error patterns from corrections with detailed analysis.

        Groups errors by type, calculates frequency, and identifies
        the most problematic categories for targeted improvement.
        """
        errors = [f for f in self._feedback if not f.was_correct]
        patterns: dict[str, int] = {}
        for error in errors:
            patterns[error.feedback_type] = patterns.get(error.feedback_type, 0) + 1

        # Rank error categories by frequency
        ranked = sorted(patterns.items(), key=lambda x: x[1], reverse=True)

        # Calculate error concentration (are errors spread or concentrated)
        total_errors = len(errors)
        concentration = 0.0
        if total_errors > 0 and ranked:
            top_category_share = ranked[0][1] / total_errors
            concentration = top_category_share

        return {
            "total_errors": total_errors,
            "by_type": patterns,
            "ranked_categories": [{"type": t, "count": c, "share": round(c / max(total_errors, 1), 3)} for t, c in ranked],
            "error_concentration": round(concentration, 3),
            "accuracy": self.get_accuracy(),
            "correction_pairs_available": len(self._correction_pairs),
        }

    def get_systematic_failures(self, min_occurrences: int = 3) -> list[dict[str, Any]]:
        """
        Identify systematic failure patterns that repeat across multiple items.

        These represent areas where the model consistently makes the same
        type of mistake, making them high-value targets for improvement.
        """
        failures: list[dict[str, Any]] = []
        for category, records in self._error_categories.items():
            if len(records) >= min_occurrences:
                failures.append({
                    "category": category,
                    "occurrence_count": len(records),
                    "sample_item_ids": [r.item_id for r in records[:5]],
                    "first_seen": records[0].timestamp.isoformat() if records[0].timestamp else None,
                    "last_seen": records[-1].timestamp.isoformat() if records[-1].timestamp else None,
                    "has_corrections": any(r.corrected_output is not None for r in records),
                })
        failures.sort(key=lambda x: x["occurrence_count"], reverse=True)
        return failures

    def should_retrain(self) -> bool:
        """Check if enough feedback has accumulated to warrant retraining."""
        errors = sum(1 for f in self._feedback if not f.was_correct)
        return errors >= self._settings.learning.retrain_threshold

    def trigger_retrain(self) -> dict[str, Any] | None:
        """
        Trigger a retraining cycle if conditions are met.

        Returns retraining metadata if triggered, None if conditions not met.
        """
        if not self.should_retrain():
            return None

        training_data = self.get_training_data()
        if not training_data:
            return None

        event = {
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "training_samples": len(training_data),
            "error_patterns": self.get_error_patterns(),
            "systematic_failures": self.get_systematic_failures(),
            "accuracy_at_trigger": self.get_accuracy(),
        }
        self._retrain_events.append(event)
        logger.info("retrain_triggered", samples=len(training_data))
        return event

    def get_training_data(self) -> list[FeedbackRecord]:
        """Get correction feedback suitable for fine-tuning."""
        return [f for f in self._feedback if f.corrected_output is not None]

    def get_prioritized_training_data(self) -> list[dict[str, Any]]:
        """
        Get training data prioritized by learning value.

        Items from systematic failure categories are ranked higher
        as fixing them would improve accuracy on recurring patterns.
        """
        corrections = self._correction_pairs.copy()
        # Boost priority for items in systematic failure categories
        failure_categories = {f["category"] for f in self.get_systematic_failures()}
        for correction in corrections:
            correction["priority"] = "high" if correction.get("category") in failure_categories else "normal"
        corrections.sort(key=lambda x: 0 if x["priority"] == "high" else 1)
        return corrections

    def get_uncertainty_insights(self) -> dict[str, Any]:
        """Get insights from uncertainty sampling."""
        return {
            "most_uncertain": self._uncertainty_sampler.get_most_uncertain(10),
            "confidence_distribution": self._uncertainty_sampler.get_uncertainty_distribution(),
        }

    def get_drift_status(self) -> dict[str, Any]:
        """Get current drift detection status."""
        return self._drift_detector.get_current_stats()

    def get_stats(self) -> dict:
        """Get comprehensive learning statistics."""
        sliding = self.get_sliding_accuracy()
        return {
            "total_feedback": len(self._feedback),
            "accuracy": round(self.get_accuracy(), 3),
            "sliding_accuracy": sliding,
            "corrections_available": len(self.get_training_data()),
            "correction_pairs": len(self._correction_pairs),
            "should_retrain": self.should_retrain(),
            "retrain_events": len(self._retrain_events),
            "systematic_failures": len(self.get_systematic_failures()),
            "drift_status": self.get_drift_status(),
        }
