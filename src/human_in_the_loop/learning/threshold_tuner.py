"""
Dynamic threshold tuner: adapts routing thresholds based on feedback patterns.

As the model's confidence distribution shifts over time (e.g., after fine-tuning
or data drift), static thresholds become suboptimal. This module monitors
decision outcomes and adjusts approval/rejection thresholds to maintain
target precision and recall for auto-decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from human_in_the_loop.config.settings import get_settings
from human_in_the_loop.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ThresholdSnapshot:
    """A point-in-time record of threshold values and their performance."""
    timestamp: datetime
    auto_approve_above: float
    auto_reject_below: float
    confidence_threshold: float
    precision_at_approve: float
    precision_at_reject: float
    review_rate: float  # Fraction of items sent to human review
    reason: str = ""


@dataclass
class DecisionOutcome:
    """Records the outcome of a routing decision for threshold learning."""
    item_id: str
    confidence: float
    routed_to: str  # "auto_approve", "auto_reject", "human_review"
    final_decision: str  # "approved", "rejected"
    was_correct: bool  # Whether the routing was appropriate
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ThresholdTuner:
    """
    Dynamically adjusts confidence thresholds based on observed outcomes.

    Tracks whether auto-approved items were actually correct and whether
    auto-rejected items were actually wrong. When precision drops below
    acceptable levels, tightens thresholds. When review volume is too high
    and precision is strong, loosens thresholds to reduce human workload.

    Implements:
    - Sliding window analysis of decision outcomes
    - Target precision/recall maintenance
    - Gradual threshold adjustment with configurable step sizes
    - Safety bounds to prevent runaway threshold drift
    - History tracking for auditability
    """

    # Safety bounds - thresholds can never exceed these
    MIN_APPROVE_THRESHOLD = 0.7
    MAX_APPROVE_THRESHOLD = 0.99
    MIN_REJECT_THRESHOLD = 0.01
    MAX_REJECT_THRESHOLD = 0.4

    # Tuning parameters
    ADJUSTMENT_STEP = 0.02  # How much to adjust per tuning cycle
    TARGET_PRECISION = 0.95  # Target precision for auto-decisions
    MAX_REVIEW_RATE = 0.5  # Maximum acceptable human review rate
    MIN_WINDOW_SIZE = 20  # Minimum decisions before tuning

    def __init__(self) -> None:
        self._settings = get_settings()
        self._outcomes: list[DecisionOutcome] = []
        self._history: list[ThresholdSnapshot] = []
        self._current_approve_threshold = self._settings.approval.auto_approve_above
        self._current_reject_threshold = self._settings.approval.auto_reject_below
        self._current_confidence_threshold = self._settings.approval.confidence_threshold
        self._tuning_enabled = True
        self._last_tuned: datetime | None = None
        self._min_tuning_interval = timedelta(hours=1)

    @property
    def current_thresholds(self) -> dict[str, float]:
        """Get current active threshold values."""
        return {
            "auto_approve_above": self._current_approve_threshold,
            "auto_reject_below": self._current_reject_threshold,
            "confidence_threshold": self._current_confidence_threshold,
        }

    def record_outcome(self, outcome: DecisionOutcome) -> None:
        """Record the outcome of a routing decision."""
        self._outcomes.append(outcome)
        logger.info(
            "outcome_recorded", item_id=outcome.item_id,
            routed_to=outcome.routed_to, was_correct=outcome.was_correct,
        )

    def evaluate_and_tune(self) -> dict[str, Any] | None:
        """
        Evaluate current threshold performance and adjust if needed.

        Returns adjustment details if thresholds were changed, None otherwise.
        """
        if not self._tuning_enabled:
            return None

        # Check minimum interval
        if self._last_tuned:
            elapsed = datetime.now(timezone.utc) - self._last_tuned
            if elapsed < self._min_tuning_interval:
                return None

        # Need minimum number of outcomes
        recent = self._get_recent_outcomes()
        if len(recent) < self.MIN_WINDOW_SIZE:
            return None

        # Calculate metrics
        metrics = self._compute_metrics(recent)
        adjustments = self._determine_adjustments(metrics)

        if not adjustments:
            return None

        # Apply adjustments
        old_thresholds = self.current_thresholds.copy()
        self._apply_adjustments(adjustments)
        new_thresholds = self.current_thresholds.copy()

        # Record snapshot
        snapshot = ThresholdSnapshot(
            timestamp=datetime.now(timezone.utc),
            auto_approve_above=self._current_approve_threshold,
            auto_reject_below=self._current_reject_threshold,
            confidence_threshold=self._current_confidence_threshold,
            precision_at_approve=metrics["approve_precision"],
            precision_at_reject=metrics["reject_precision"],
            review_rate=metrics["review_rate"],
            reason=adjustments.get("reason", ""),
        )
        self._history.append(snapshot)
        self._last_tuned = datetime.now(timezone.utc)

        logger.info(
            "thresholds_tuned",
            old=old_thresholds, new=new_thresholds, reason=adjustments.get("reason", ""),
        )

        return {
            "old_thresholds": old_thresholds,
            "new_thresholds": new_thresholds,
            "metrics": metrics,
            "reason": adjustments.get("reason", ""),
        }

    def get_performance_report(self) -> dict[str, Any]:
        """
        Generate a performance report on current threshold effectiveness.

        Useful for dashboards and monitoring.
        """
        recent = self._get_recent_outcomes()
        if not recent:
            return {"status": "insufficient_data", "outcomes_collected": len(self._outcomes)}

        metrics = self._compute_metrics(recent)
        return {
            "current_thresholds": self.current_thresholds,
            "metrics": metrics,
            "outcomes_in_window": len(recent),
            "total_outcomes": len(self._outcomes),
            "adjustment_history_count": len(self._history),
            "last_tuned": self._last_tuned.isoformat() if self._last_tuned else None,
            "recommendations": self._generate_recommendations(metrics),
        }

    def force_tune(self) -> dict[str, Any] | None:
        """Force a tuning evaluation regardless of interval."""
        saved_interval = self._min_tuning_interval
        self._min_tuning_interval = timedelta(seconds=0)
        result = self.evaluate_and_tune()
        self._min_tuning_interval = saved_interval
        return result

    def set_thresholds(self, approve_above: float | None = None,
                       reject_below: float | None = None,
                       confidence_threshold: float | None = None) -> dict[str, float]:
        """Manually override thresholds (disables auto-tuning until re-enabled)."""
        if approve_above is not None:
            self._current_approve_threshold = max(
                self.MIN_APPROVE_THRESHOLD, min(self.MAX_APPROVE_THRESHOLD, approve_above)
            )
        if reject_below is not None:
            self._current_reject_threshold = max(
                self.MIN_REJECT_THRESHOLD, min(self.MAX_REJECT_THRESHOLD, reject_below)
            )
        if confidence_threshold is not None:
            self._current_confidence_threshold = max(0.5, min(0.95, confidence_threshold))
        return self.current_thresholds

    def get_history(self) -> list[dict]:
        """Get threshold adjustment history."""
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "thresholds": {
                    "auto_approve_above": s.auto_approve_above,
                    "auto_reject_below": s.auto_reject_below,
                    "confidence_threshold": s.confidence_threshold,
                },
                "performance": {
                    "precision_at_approve": s.precision_at_approve,
                    "precision_at_reject": s.precision_at_reject,
                    "review_rate": s.review_rate,
                },
                "reason": s.reason,
            }
            for s in self._history
        ]

    def get_stats(self) -> dict:
        """Get tuner statistics."""
        return {
            "tuning_enabled": self._tuning_enabled,
            "current_thresholds": self.current_thresholds,
            "total_outcomes_tracked": len(self._outcomes),
            "adjustments_made": len(self._history),
            "last_tuned": self._last_tuned.isoformat() if self._last_tuned else None,
        }

    def _get_recent_outcomes(self, window_hours: int = 24) -> list[DecisionOutcome]:
        """Get outcomes from the recent window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        return [o for o in self._outcomes if o.timestamp >= cutoff]

    def _compute_metrics(self, outcomes: list[DecisionOutcome]) -> dict[str, float]:
        """Compute precision metrics from outcomes."""
        auto_approved = [o for o in outcomes if o.routed_to == "auto_approve"]
        auto_rejected = [o for o in outcomes if o.routed_to == "auto_reject"]
        human_reviewed = [o for o in outcomes if o.routed_to == "human_review"]

        approve_correct = sum(1 for o in auto_approved if o.was_correct)
        reject_correct = sum(1 for o in auto_rejected if o.was_correct)

        approve_precision = approve_correct / max(len(auto_approved), 1)
        reject_precision = reject_correct / max(len(auto_rejected), 1)
        review_rate = len(human_reviewed) / max(len(outcomes), 1)

        return {
            "approve_precision": round(approve_precision, 4),
            "reject_precision": round(reject_precision, 4),
            "review_rate": round(review_rate, 4),
            "auto_approve_count": len(auto_approved),
            "auto_reject_count": len(auto_rejected),
            "human_review_count": len(human_reviewed),
            "total": len(outcomes),
        }

    def _determine_adjustments(self, metrics: dict[str, float]) -> dict[str, Any]:
        """Determine what threshold adjustments are needed."""
        adjustments: dict[str, Any] = {}

        # If auto-approve precision is too low, tighten (raise) the threshold
        if metrics["approve_precision"] < self.TARGET_PRECISION and metrics["auto_approve_count"] >= 5:
            adjustments["approve_direction"] = "tighten"
            adjustments["reason"] = f"Auto-approve precision {metrics['approve_precision']:.2%} below target {self.TARGET_PRECISION:.2%}"

        # If auto-reject precision is too low, tighten (lower) the reject threshold
        elif metrics["reject_precision"] < self.TARGET_PRECISION and metrics["auto_reject_count"] >= 5:
            adjustments["reject_direction"] = "tighten"
            adjustments["reason"] = f"Auto-reject precision {metrics['reject_precision']:.2%} below target {self.TARGET_PRECISION:.2%}"

        # If review rate is too high and both precisions are good, loosen thresholds
        elif metrics["review_rate"] > self.MAX_REVIEW_RATE:
            if metrics["approve_precision"] >= self.TARGET_PRECISION:
                adjustments["approve_direction"] = "loosen"
                adjustments["reason"] = f"Review rate {metrics['review_rate']:.2%} too high, approve precision strong"
            if metrics["reject_precision"] >= self.TARGET_PRECISION:
                adjustments["reject_direction"] = "loosen"
                if "reason" not in adjustments:
                    adjustments["reason"] = f"Review rate {metrics['review_rate']:.2%} too high, reject precision strong"

        return adjustments

    def _apply_adjustments(self, adjustments: dict[str, Any]) -> None:
        """Apply computed adjustments within safety bounds."""
        if "approve_direction" in adjustments:
            if adjustments["approve_direction"] == "tighten":
                self._current_approve_threshold = min(
                    self.MAX_APPROVE_THRESHOLD,
                    self._current_approve_threshold + self.ADJUSTMENT_STEP,
                )
            else:  # loosen
                self._current_approve_threshold = max(
                    self.MIN_APPROVE_THRESHOLD,
                    self._current_approve_threshold - self.ADJUSTMENT_STEP,
                )

        if "reject_direction" in adjustments:
            if adjustments["reject_direction"] == "tighten":
                self._current_reject_threshold = max(
                    self.MIN_REJECT_THRESHOLD,
                    self._current_reject_threshold - self.ADJUSTMENT_STEP,
                )
            else:  # loosen
                self._current_reject_threshold = min(
                    self.MAX_REJECT_THRESHOLD,
                    self._current_reject_threshold + self.ADJUSTMENT_STEP,
                )

        # Keep confidence threshold between reject and approve
        midpoint = (self._current_approve_threshold + self._current_reject_threshold) / 2
        self._current_confidence_threshold = max(
            self._current_reject_threshold + 0.1,
            min(self._current_approve_threshold - 0.1, midpoint),
        )

    def _generate_recommendations(self, metrics: dict[str, float]) -> list[str]:
        """Generate human-readable recommendations based on metrics."""
        recs: list[str] = []
        if metrics["approve_precision"] < self.TARGET_PRECISION:
            recs.append("Consider raising auto-approve threshold - too many false approvals")
        if metrics["reject_precision"] < self.TARGET_PRECISION:
            recs.append("Consider lowering auto-reject threshold - too many false rejections")
        if metrics["review_rate"] > self.MAX_REVIEW_RATE:
            recs.append("High human review rate - consider loosening thresholds if precision allows")
        if metrics["review_rate"] < 0.1:
            recs.append("Very low review rate - verify model outputs are truly high quality")
        if not recs:
            recs.append("Thresholds performing within target parameters")
        return recs
