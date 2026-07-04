"""FastAPI router for human-in-the-loop endpoints."""
from fastapi import APIRouter, HTTPException
from human_in_the_loop.approval.gate import ApprovalGate
from human_in_the_loop.queue.review_queue import ReviewQueue
from human_in_the_loop.learning.active_learner import ActiveLearner
from human_in_the_loop.models.schemas import ApprovalDecision, FeedbackRecord, ReviewItem

router = APIRouter(prefix="/api/v1/hitl", tags=["human-in-the-loop"])

_gate = ApprovalGate()
_queue = ReviewQueue()
_learner = ActiveLearner()


@router.post("/submit")
async def submit_for_review(item: ReviewItem) -> dict:
    """Submit an AI output for confidence-based routing."""
    status = _gate.evaluate(item)
    item.status = status
    if status.value == "pending":
        _queue.add_item(item)
    return {"item_id": item.item_id, "status": status.value, "confidence": item.confidence}


@router.get("/queue")
async def get_queue(reviewer_id: str | None = None) -> list[dict]:
    """Get pending review items."""
    items = _queue.get_pending(reviewer_id)
    return [i.model_dump() for i in items]


@router.post("/decide")
async def submit_decision(decision: ApprovalDecision) -> dict:
    """Submit a human decision on a review item."""
    item = _queue.process_decision(decision)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    # Record feedback for learning
    feedback = FeedbackRecord(
        item_id=item.item_id, original_output=item.content,
        corrected_output=decision.correction, was_correct=(decision.decision.value == "approved"),
        feedback_type="confirmation" if decision.decision.value == "approved" else "correction",
    )
    _learner.record_feedback(feedback)
    return {"item_id": item.item_id, "status": item.status.value}


@router.post("/reviewers")
async def register_reviewer(reviewer_id: str) -> dict:
    """Register a human reviewer."""
    _queue.register_reviewer(reviewer_id)
    return {"registered": True, "reviewer_id": reviewer_id}


@router.get("/stats")
async def get_stats() -> dict:
    """Get queue and learning statistics."""
    return {"queue": _queue.get_stats(), "learning": _learner.get_stats()}


@router.get("/health")
async def health() -> dict:
    return {"status": "healthy", "service": "human-in-the-loop"}
