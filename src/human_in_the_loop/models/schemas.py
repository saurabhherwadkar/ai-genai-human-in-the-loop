"""Pydantic schemas for human-in-the-loop."""
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ReviewItem(BaseModel):
    """An item requiring human review."""
    item_id: str = Field(description="Unique item ID")
    content: dict[str, Any] = Field(description="AI-generated content to review")
    context: str = Field(default="", description="Context for the reviewer")
    confidence: float = Field(description="AI confidence score 0-1")
    status: ReviewStatus = Field(default=ReviewStatus.PENDING)
    priority: Priority = Field(default=Priority.MEDIUM)
    assigned_to: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: datetime | None = Field(default=None)
    reviewer_notes: str = Field(default="")
    correction: dict[str, Any] | None = Field(default=None, description="Human correction if rejected")


class ApprovalDecision(BaseModel):
    """Human's decision on a review item."""
    item_id: str = Field(description="Item being reviewed")
    decision: ReviewStatus = Field(description="approved or rejected")
    notes: str = Field(default="")
    correction: dict[str, Any] | None = Field(default=None)
    reviewer_id: str = Field(default="anonymous")


class FeedbackRecord(BaseModel):
    """Record of human feedback for active learning."""
    item_id: str
    original_output: dict[str, Any]
    corrected_output: dict[str, Any] | None
    was_correct: bool
    feedback_type: str = Field(default="correction", description="correction, confirmation, or rejection")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
