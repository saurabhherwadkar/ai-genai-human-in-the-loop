"""
Escalation manager: routes edge cases to appropriate decision-makers.

Implements multi-tier escalation based on risk scoring, timeout breaches,
confidence uncertainty bands, and domain-specific routing rules.
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from human_in_the_loop.config.settings import get_settings
from human_in_the_loop.models.schemas import Priority, ReviewItem, ReviewStatus
from human_in_the_loop.utils.logger import get_logger

logger = get_logger(__name__)


class EscalationTier(str, Enum):
    """Defines the escalation hierarchy."""
    TIER_1 = "tier_1"  # Standard reviewer
    TIER_2 = "tier_2"  # Senior reviewer / domain expert
    TIER_3 = "tier_3"  # Manager / compliance officer
    EXECUTIVE = "executive"  # Final escalation for critical decisions


class EscalationReason(str, Enum):
    """Why an item was escalated."""
    TIMEOUT = "timeout"
    LOW_AGREEMENT = "low_agreement"
    HIGH_RISK = "high_risk"
    CONFIDENCE_UNCERTAINTY = "confidence_uncertainty"
    REPEATED_REJECTION = "repeated_rejection"
    POLICY_VIOLATION = "policy_violation"
    MANUAL = "manual"


class EscalationRule(BaseModel):
    """A rule that triggers escalation when conditions are met."""
    name: str
    tier: EscalationTier
    reason: EscalationReason
    condition_description: str
    priority_override: Priority | None = None
    auto_apply: bool = True


class EscalationRecord(BaseModel):
    """Tracks an escalation event for audit purposes."""
    item_id: str
    from_tier: EscalationTier
    to_tier: EscalationTier
    reason: EscalationReason
    triggered_by: str = Field(default="system")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context: dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False
    resolution_notes: str = ""


class EscalationManager:
    """
    Manages escalation workflows for items requiring higher-authority review.

    Supports:
    - Multi-tier escalation (reviewer -> senior -> manager -> executive)
    - Risk-based routing with configurable scoring
    - Timeout-triggered auto-escalation
    - Repeated-rejection detection
    - Domain-specific escalation rules
    """

    TIER_ORDER = [EscalationTier.TIER_1, EscalationTier.TIER_2, EscalationTier.TIER_3, EscalationTier.EXECUTIVE]

    def __init__(self) -> None:
        self._settings = get_settings()
        self._rules: list[EscalationRule] = self._default_rules()
        self._records: list[EscalationRecord] = []
        self._tier_assignments: dict[EscalationTier, list[str]] = {
            tier: [] for tier in EscalationTier
        }
        self._item_rejection_count: dict[str, int] = {}
        self._item_tiers: dict[str, EscalationTier] = {}

    def _default_rules(self) -> list[EscalationRule]:
        """Initialize default escalation rules."""
        return [
            EscalationRule(
                name="timeout_escalation",
                tier=EscalationTier.TIER_2,
                reason=EscalationReason.TIMEOUT,
                condition_description="Item pending longer than escalation timeout",
                priority_override=Priority.HIGH,
            ),
            EscalationRule(
                name="high_risk_content",
                tier=EscalationTier.TIER_3,
                reason=EscalationReason.HIGH_RISK,
                condition_description="Item flagged as high-risk based on content analysis",
                priority_override=Priority.URGENT,
            ),
            EscalationRule(
                name="repeated_rejection",
                tier=EscalationTier.TIER_2,
                reason=EscalationReason.REPEATED_REJECTION,
                condition_description="Item rejected multiple times by different reviewers",
            ),
            EscalationRule(
                name="confidence_uncertainty_band",
                tier=EscalationTier.TIER_2,
                reason=EscalationReason.CONFIDENCE_UNCERTAINTY,
                condition_description="Confidence falls in high-uncertainty band (0.4-0.6)",
            ),
            EscalationRule(
                name="policy_violation_detected",
                tier=EscalationTier.TIER_3,
                reason=EscalationReason.POLICY_VIOLATION,
                condition_description="Content may violate organizational policy",
                priority_override=Priority.URGENT,
            ),
        ]

    def register_tier_reviewer(self, reviewer_id: str, tier: EscalationTier) -> None:
        """Register a reviewer at a specific escalation tier."""
        if reviewer_id not in self._tier_assignments[tier]:
            self._tier_assignments[tier].append(reviewer_id)
            logger.info("tier_reviewer_registered", reviewer_id=reviewer_id, tier=tier.value)

    def evaluate_escalation(self, item: ReviewItem) -> EscalationTier | None:
        """
        Evaluate whether an item needs escalation based on configured rules.

        Returns the target tier if escalation is needed, None otherwise.
        """
        current_tier = self._item_tiers.get(item.item_id, EscalationTier.TIER_1)

        # Check timeout-based escalation
        if self._check_timeout_breach(item):
            target = self._next_tier(current_tier)
            if target:
                self._escalate(item, current_tier, target, EscalationReason.TIMEOUT)
                return target

        # Check confidence uncertainty band (0.4-0.6 is highly uncertain)
        if 0.4 <= item.confidence <= 0.6:
            target = EscalationTier.TIER_2
            if self._tier_is_higher(target, current_tier):
                self._escalate(item, current_tier, target, EscalationReason.CONFIDENCE_UNCERTAINTY)
                return target

        # Check repeated rejections
        rejection_count = self._item_rejection_count.get(item.item_id, 0)
        if rejection_count >= 2:
            target = EscalationTier.TIER_2 if rejection_count < 4 else EscalationTier.TIER_3
            if self._tier_is_higher(target, current_tier):
                self._escalate(item, current_tier, target, EscalationReason.REPEATED_REJECTION)
                return target

        return None

    def escalate_manual(self, item: ReviewItem, reason: str, escalated_by: str) -> EscalationRecord:
        """Manually escalate an item to the next tier."""
        current_tier = self._item_tiers.get(item.item_id, EscalationTier.TIER_1)
        target = self._next_tier(current_tier)
        if not target:
            target = EscalationTier.EXECUTIVE

        record = self._escalate(
            item, current_tier, target, EscalationReason.MANUAL,
            triggered_by=escalated_by, context={"reason_text": reason}
        )
        return record

    def escalate_for_risk(self, item: ReviewItem, risk_score: float, risk_factors: list[str]) -> EscalationRecord | None:
        """
        Escalate based on computed risk score.

        Risk score 0-1 where:
        - 0.0-0.3: No escalation needed
        - 0.3-0.6: Tier 2 (senior reviewer)
        - 0.6-0.8: Tier 3 (manager)
        - 0.8-1.0: Executive escalation
        """
        if risk_score < 0.3:
            return None

        current_tier = self._item_tiers.get(item.item_id, EscalationTier.TIER_1)

        if risk_score >= 0.8:
            target = EscalationTier.EXECUTIVE
        elif risk_score >= 0.6:
            target = EscalationTier.TIER_3
        else:
            target = EscalationTier.TIER_2

        if not self._tier_is_higher(target, current_tier):
            return None

        return self._escalate(
            item, current_tier, target, EscalationReason.HIGH_RISK,
            context={"risk_score": risk_score, "risk_factors": risk_factors}
        )

    def record_rejection(self, item_id: str) -> int:
        """Record a rejection event and return the new count."""
        self._item_rejection_count[item_id] = self._item_rejection_count.get(item_id, 0) + 1
        return self._item_rejection_count[item_id]

    def get_escalation_history(self, item_id: str) -> list[EscalationRecord]:
        """Get full escalation history for an item."""
        return [r for r in self._records if r.item_id == item_id]

    def get_tier_workload(self) -> dict[str, dict]:
        """Get current workload distribution across tiers."""
        workload: dict[str, dict] = {}
        for tier in EscalationTier:
            items_at_tier = sum(1 for t in self._item_tiers.values() if t == tier)
            reviewers = len(self._tier_assignments[tier])
            workload[tier.value] = {
                "pending_items": items_at_tier,
                "available_reviewers": reviewers,
                "load_ratio": items_at_tier / max(reviewers, 1),
            }
        return workload

    def resolve_escalation(self, item_id: str, resolution_notes: str) -> bool:
        """Mark an escalation as resolved."""
        for record in reversed(self._records):
            if record.item_id == item_id and not record.resolved:
                record.resolved = True
                record.resolution_notes = resolution_notes
                logger.info("escalation_resolved", item_id=item_id)
                return True
        return False

    def get_stats(self) -> dict:
        """Get escalation statistics."""
        total = len(self._records)
        resolved = sum(1 for r in self._records if r.resolved)
        by_reason: dict[str, int] = {}
        for r in self._records:
            by_reason[r.reason.value] = by_reason.get(r.reason.value, 0) + 1
        return {
            "total_escalations": total,
            "resolved": resolved,
            "pending": total - resolved,
            "by_reason": by_reason,
            "tier_workload": self.get_tier_workload(),
        }

    def _escalate(
        self, item: ReviewItem, from_tier: EscalationTier, to_tier: EscalationTier,
        reason: EscalationReason, triggered_by: str = "system", context: dict | None = None,
    ) -> EscalationRecord:
        """Internal escalation execution."""
        record = EscalationRecord(
            item_id=item.item_id,
            from_tier=from_tier,
            to_tier=to_tier,
            reason=reason,
            triggered_by=triggered_by,
            context=context or {},
        )
        self._records.append(record)
        self._item_tiers[item.item_id] = to_tier
        item.status = ReviewStatus.ESCALATED
        item.priority = self._escalation_priority(to_tier)
        logger.info(
            "item_escalated", item_id=item.item_id,
            from_tier=from_tier.value, to_tier=to_tier.value, reason=reason.value,
        )
        return record

    def _check_timeout_breach(self, item: ReviewItem) -> bool:
        """Check if item has exceeded the escalation timeout."""
        if not item.created_at:
            return False
        elapsed = (datetime.now(timezone.utc) - item.created_at.replace(tzinfo=timezone.utc)).total_seconds()
        return elapsed > self._settings.approval.escalation_after_seconds

    def _next_tier(self, current: EscalationTier) -> EscalationTier | None:
        """Get the next escalation tier above the current one."""
        idx = self.TIER_ORDER.index(current)
        if idx < len(self.TIER_ORDER) - 1:
            return self.TIER_ORDER[idx + 1]
        return None

    def _tier_is_higher(self, target: EscalationTier, current: EscalationTier) -> bool:
        """Check if target tier is higher than current."""
        return self.TIER_ORDER.index(target) > self.TIER_ORDER.index(current)

    def _escalation_priority(self, tier: EscalationTier) -> Priority:
        """Map escalation tier to priority level."""
        mapping = {
            EscalationTier.TIER_1: Priority.MEDIUM,
            EscalationTier.TIER_2: Priority.HIGH,
            EscalationTier.TIER_3: Priority.URGENT,
            EscalationTier.EXECUTIVE: Priority.URGENT,
        }
        return mapping.get(tier, Priority.HIGH)
