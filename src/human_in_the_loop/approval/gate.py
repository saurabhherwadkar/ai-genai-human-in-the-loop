"""
Approval gate: routes items based on confidence thresholds.

Auto-approves high-confidence outputs, auto-rejects very low confidence,
and sends everything in between to human review.
"""

from human_in_the_loop.config.settings import get_settings
from human_in_the_loop.models.schemas import ReviewItem, ReviewStatus
from human_in_the_loop.utils.logger import get_logger

logger = get_logger(__name__)


class ApprovalGate:
    """Routes AI outputs based on confidence thresholds."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def evaluate(self, item: ReviewItem) -> ReviewStatus:
        """
        Determine if an item needs human review based on confidence.

        Args:
            item: The review item with confidence score.

        Returns:
            ReviewStatus indicating routing decision.
        """
        # Auto-approve high confidence
        if item.confidence >= self._settings.approval.auto_approve_above:
            logger.info("auto_approved", item_id=item.item_id, confidence=item.confidence)
            return ReviewStatus.APPROVED

        # Auto-reject very low confidence
        if item.confidence <= self._settings.approval.auto_reject_below:
            logger.info("auto_rejected", item_id=item.item_id, confidence=item.confidence)
            return ReviewStatus.REJECTED

        # Below threshold — needs human review
        if item.confidence < self._settings.approval.confidence_threshold:
            logger.info("sent_to_review", item_id=item.item_id, confidence=item.confidence)
            return ReviewStatus.PENDING

        # Above threshold but below auto-approve — approve
        logger.info("threshold_approved", item_id=item.item_id, confidence=item.confidence)
        return ReviewStatus.APPROVED
