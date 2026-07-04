"""
Active learning: improve the system from human corrections.

Collects human feedback and tracks patterns to identify where
the AI model is weakest, guiding future improvement.
"""

from human_in_the_loop.config.settings import get_settings
from human_in_the_loop.models.schemas import FeedbackRecord
from human_in_the_loop.utils.logger import get_logger

logger = get_logger(__name__)


class ActiveLearner:
    """Learns from human corrections to improve AI accuracy over time."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._feedback: list[FeedbackRecord] = []

    def record_feedback(self, feedback: FeedbackRecord) -> None:
        """Record human feedback for learning."""
        self._feedback.append(feedback)
        logger.info("feedback_recorded", was_correct=feedback.was_correct, type=feedback.feedback_type)

    def get_accuracy(self) -> float:
        """Calculate current accuracy from feedback."""
        if not self._feedback:
            return 0.0
        correct = sum(1 for f in self._feedback if f.was_correct)
        return correct / len(self._feedback)

    def get_error_patterns(self) -> dict:
        """Identify common error patterns from corrections."""
        errors = [f for f in self._feedback if not f.was_correct]
        # Group by feedback type
        patterns: dict[str, int] = {}
        for error in errors:
            patterns[error.feedback_type] = patterns.get(error.feedback_type, 0) + 1
        return {"total_errors": len(errors), "by_type": patterns, "accuracy": self.get_accuracy()}

    def should_retrain(self) -> bool:
        """Check if enough feedback has accumulated to warrant retraining."""
        errors = sum(1 for f in self._feedback if not f.was_correct)
        return errors >= self._settings.learning.retrain_threshold

    def get_training_data(self) -> list[FeedbackRecord]:
        """Get correction feedback suitable for fine-tuning."""
        return [f for f in self._feedback if f.corrected_output is not None]

    def get_stats(self) -> dict:
        """Get learning statistics."""
        return {
            "total_feedback": len(self._feedback),
            "accuracy": round(self.get_accuracy(), 3),
            "corrections_available": len(self.get_training_data()),
            "should_retrain": self.should_retrain(),
        }
