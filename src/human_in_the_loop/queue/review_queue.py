"""
Review queue for managing pending human review items.

Supports priority ordering, assignment, and timeout handling.
"""

import uuid
from datetime import datetime, timezone
from collections import deque

from human_in_the_loop.config.settings import get_settings
from human_in_the_loop.models.schemas import ApprovalDecision, Priority, ReviewItem, ReviewStatus
from human_in_the_loop.utils.logger import get_logger

logger = get_logger(__name__)


class ReviewQueue:
    """Priority queue for items awaiting human review."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._items: dict[str, ReviewItem] = {}
        self._reviewers: list[str] = []
        self._assignment_index: int = 0

    def add_item(self, item: ReviewItem) -> ReviewItem:
        """Add an item to the review queue."""
        if len(self._items) >= self._settings.queue.max_pending_items:
            raise RuntimeError("Review queue is full")
        self._items[item.item_id] = item
        # Auto-assign if reviewers available
        if self._reviewers:
            item.assigned_to = self._assign_reviewer()
        logger.info("item_queued", item_id=item.item_id, priority=item.priority.value)
        return item

    def get_pending(self, reviewer_id: str | None = None) -> list[ReviewItem]:
        """Get pending items, optionally filtered by assignee."""
        items = [i for i in self._items.values() if i.status == ReviewStatus.PENDING]
        if reviewer_id:
            items = [i for i in items if i.assigned_to == reviewer_id]
        # Sort by priority
        priority_order = {Priority.URGENT: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}
        items.sort(key=lambda x: priority_order.get(x.priority, 2))
        return items

    def process_decision(self, decision: ApprovalDecision) -> ReviewItem | None:
        """Process a human decision on a review item."""
        item = self._items.get(decision.item_id)
        if not item:
            return None
        item.status = decision.decision
        item.reviewed_at = datetime.now(timezone.utc)
        item.reviewer_notes = decision.notes
        if decision.correction:
            item.correction = decision.correction
        logger.info("decision_processed", item_id=item.item_id, decision=decision.decision.value)
        return item

    def register_reviewer(self, reviewer_id: str) -> None:
        """Register a human reviewer."""
        if reviewer_id not in self._reviewers:
            self._reviewers.append(reviewer_id)

    def get_stats(self) -> dict:
        """Get queue statistics."""
        pending = sum(1 for i in self._items.values() if i.status == ReviewStatus.PENDING)
        approved = sum(1 for i in self._items.values() if i.status == ReviewStatus.APPROVED)
        rejected = sum(1 for i in self._items.values() if i.status == ReviewStatus.REJECTED)
        return {"total": len(self._items), "pending": pending, "approved": approved, "rejected": rejected, "reviewers": len(self._reviewers)}

    def _assign_reviewer(self) -> str:
        """Assign reviewer using round-robin."""
        if not self._reviewers:
            return ""
        reviewer = self._reviewers[self._assignment_index % len(self._reviewers)]
        self._assignment_index += 1
        return reviewer
