"""
Audit trail: complete decision history for compliance and traceability.

Records every significant event in the HITL workflow including submissions,
routing decisions, assignments, reviews, escalations, and corrections.
Supports querying, retention policies, and export for compliance reporting.
"""

import json
import hashlib
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from pathlib import Path

from pydantic import BaseModel, Field

from human_in_the_loop.config.settings import get_settings
from human_in_the_loop.utils.logger import get_logger

logger = get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of auditable events in the HITL workflow."""
    ITEM_SUBMITTED = "item_submitted"
    ROUTING_DECISION = "routing_decision"
    ITEM_QUEUED = "item_queued"
    REVIEWER_ASSIGNED = "reviewer_assigned"
    REVIEW_STARTED = "review_started"
    DECISION_MADE = "decision_made"
    CORRECTION_APPLIED = "correction_applied"
    ESCALATION_TRIGGERED = "escalation_triggered"
    ESCALATION_RESOLVED = "escalation_resolved"
    THRESHOLD_ADJUSTED = "threshold_adjusted"
    FEEDBACK_RECORDED = "feedback_recorded"
    RETRAIN_TRIGGERED = "retrain_triggered"
    ITEM_EXPIRED = "item_expired"
    REVIEWER_REGISTERED = "reviewer_registered"
    POLICY_OVERRIDE = "policy_override"


class AuditEntry(BaseModel):
    """A single audit trail entry with tamper-evident hashing."""
    entry_id: str = Field(default="")
    event_type: AuditEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor: str = Field(default="system", description="Who or what triggered the event")
    item_id: str | None = Field(default=None)
    details: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str = Field(default="")
    entry_hash: str = Field(default="")

    def compute_hash(self, previous_hash: str = "") -> str:
        """Compute a hash for tamper detection using chained hashing."""
        payload = f"{self.event_type.value}|{self.timestamp.isoformat()}|{self.actor}|{self.item_id}|{json.dumps(self.details, sort_keys=True, default=str)}|{previous_hash}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class AuditQuery(BaseModel):
    """Query parameters for searching audit entries."""
    item_id: str | None = None
    event_type: AuditEventType | None = None
    actor: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int = 100
    offset: int = 0


class AuditTrail:
    """
    Immutable audit trail for all HITL workflow decisions.

    Features:
    - Chained hashing for tamper detection
    - Full event history per item
    - Retention policy enforcement
    - Query and export capabilities
    - Compliance-ready reporting
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._entries: list[AuditEntry] = []
        self._entry_counter: int = 0
        self._last_hash: str = "genesis"

    def record(self, event_type: AuditEventType, actor: str = "system",
               item_id: str | None = None, details: dict[str, Any] | None = None) -> AuditEntry:
        """
        Record an audit event with chained integrity hashing.

        Args:
            event_type: The type of event being recorded.
            actor: Who or what triggered the event.
            item_id: The related review item ID, if applicable.
            details: Additional context for the event.

        Returns:
            The created audit entry.
        """
        self._entry_counter += 1
        entry = AuditEntry(
            entry_id=f"audit_{self._entry_counter:06d}",
            event_type=event_type,
            actor=actor,
            item_id=item_id,
            details=details or {},
            previous_hash=self._last_hash,
        )
        entry.entry_hash = entry.compute_hash(self._last_hash)
        self._last_hash = entry.entry_hash
        self._entries.append(entry)

        logger.info(
            "audit_recorded", entry_id=entry.entry_id,
            event_type=event_type.value, item_id=item_id,
        )
        return entry

    def record_submission(self, item_id: str, confidence: float, content_summary: str) -> AuditEntry:
        """Record an item submission event."""
        return self.record(
            AuditEventType.ITEM_SUBMITTED, item_id=item_id,
            details={"confidence": confidence, "content_summary": content_summary},
        )

    def record_routing(self, item_id: str, decision: str, confidence: float, thresholds: dict) -> AuditEntry:
        """Record a routing decision with the thresholds that were applied."""
        return self.record(
            AuditEventType.ROUTING_DECISION, item_id=item_id,
            details={"decision": decision, "confidence": confidence, "thresholds": thresholds},
        )

    def record_review_decision(self, item_id: str, reviewer_id: str, decision: str,
                                notes: str = "", had_correction: bool = False) -> AuditEntry:
        """Record a human review decision."""
        return self.record(
            AuditEventType.DECISION_MADE, actor=reviewer_id, item_id=item_id,
            details={"decision": decision, "notes": notes, "had_correction": had_correction},
        )

    def record_escalation(self, item_id: str, from_tier: str, to_tier: str,
                           reason: str, triggered_by: str = "system") -> AuditEntry:
        """Record an escalation event."""
        return self.record(
            AuditEventType.ESCALATION_TRIGGERED, actor=triggered_by, item_id=item_id,
            details={"from_tier": from_tier, "to_tier": to_tier, "reason": reason},
        )

    def record_threshold_change(self, old_thresholds: dict, new_thresholds: dict, reason: str) -> AuditEntry:
        """Record a threshold adjustment event."""
        return self.record(
            AuditEventType.THRESHOLD_ADJUSTED,
            details={"old": old_thresholds, "new": new_thresholds, "reason": reason},
        )

    def query(self, query: AuditQuery) -> list[AuditEntry]:
        """Query audit entries with filters."""
        results = self._entries

        if query.item_id:
            results = [e for e in results if e.item_id == query.item_id]
        if query.event_type:
            results = [e for e in results if e.event_type == query.event_type]
        if query.actor:
            results = [e for e in results if e.actor == query.actor]
        if query.start_time:
            results = [e for e in results if e.timestamp >= query.start_time]
        if query.end_time:
            results = [e for e in results if e.timestamp <= query.end_time]

        return results[query.offset: query.offset + query.limit]

    def get_item_history(self, item_id: str) -> list[AuditEntry]:
        """Get complete audit history for a specific item."""
        return [e for e in self._entries if e.item_id == item_id]

    def verify_integrity(self) -> dict[str, Any]:
        """
        Verify the integrity of the audit trail by checking the hash chain.

        Returns a report indicating whether the chain is intact.
        """
        if not self._entries:
            return {"valid": True, "entries_checked": 0}

        previous_hash = "genesis"
        broken_at: list[str] = []

        for entry in self._entries:
            expected_hash = entry.compute_hash(previous_hash)
            if entry.entry_hash != expected_hash:
                broken_at.append(entry.entry_id)
            if entry.previous_hash != previous_hash:
                broken_at.append(entry.entry_id)
            previous_hash = entry.entry_hash

        return {
            "valid": len(broken_at) == 0,
            "entries_checked": len(self._entries),
            "integrity_breaks": broken_at,
        }

    def enforce_retention(self) -> int:
        """
        Remove entries older than the configured retention period.

        Returns the number of entries purged.
        """
        retention_days = self._settings.audit.retention_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        original_count = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
        purged = original_count - len(self._entries)
        if purged > 0:
            logger.info("retention_enforced", purged_count=purged, retention_days=retention_days)
        return purged

    def export_for_compliance(self, item_id: str | None = None) -> list[dict]:
        """
        Export audit entries in a compliance-friendly format.

        Suitable for external audit systems and regulatory reporting.
        """
        entries = self.get_item_history(item_id) if item_id else self._entries
        return [
            {
                "id": e.entry_id,
                "type": e.event_type.value,
                "timestamp_utc": e.timestamp.isoformat(),
                "actor": e.actor,
                "item_id": e.item_id,
                "details": e.details,
                "integrity_hash": e.entry_hash,
            }
            for e in entries
        ]

    def get_stats(self) -> dict:
        """Get audit trail statistics."""
        by_type: dict[str, int] = {}
        for entry in self._entries:
            by_type[entry.event_type.value] = by_type.get(entry.event_type.value, 0) + 1

        return {
            "total_entries": len(self._entries),
            "by_event_type": by_type,
            "integrity": self.verify_integrity(),
            "oldest_entry": self._entries[0].timestamp.isoformat() if self._entries else None,
            "newest_entry": self._entries[-1].timestamp.isoformat() if self._entries else None,
        }
