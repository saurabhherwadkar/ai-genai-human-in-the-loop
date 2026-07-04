"""Tests for approval gate and queue."""
import pytest
from human_in_the_loop.approval.gate import ApprovalGate
from human_in_the_loop.queue.review_queue import ReviewQueue
from human_in_the_loop.learning.active_learner import ActiveLearner
from human_in_the_loop.models.schemas import ApprovalDecision, FeedbackRecord, ReviewItem, ReviewStatus, Priority


@pytest.fixture
def gate() -> ApprovalGate:
    return ApprovalGate()

@pytest.fixture
def queue() -> ReviewQueue:
    return ReviewQueue()

@pytest.fixture
def learner() -> ActiveLearner:
    return ActiveLearner()


class TestApprovalGate:
    def test_auto_approves_high_confidence(self, gate: ApprovalGate) -> None:
        item = ReviewItem(item_id="i1", content={"text": "hi"}, confidence=0.98)
        assert gate.evaluate(item) == ReviewStatus.APPROVED

    def test_auto_rejects_low_confidence(self, gate: ApprovalGate) -> None:
        item = ReviewItem(item_id="i2", content={"text": "hi"}, confidence=0.1)
        assert gate.evaluate(item) == ReviewStatus.REJECTED

    def test_sends_to_review_below_threshold(self, gate: ApprovalGate) -> None:
        item = ReviewItem(item_id="i3", content={"text": "hi"}, confidence=0.5)
        assert gate.evaluate(item) == ReviewStatus.PENDING


class TestReviewQueue:
    def test_add_and_get_pending(self, queue: ReviewQueue) -> None:
        item = ReviewItem(item_id="q1", content={"text": "test"}, confidence=0.5)
        queue.add_item(item)
        pending = queue.get_pending()
        assert len(pending) == 1

    def test_process_decision(self, queue: ReviewQueue) -> None:
        item = ReviewItem(item_id="q2", content={"text": "test"}, confidence=0.5)
        queue.add_item(item)
        decision = ApprovalDecision(item_id="q2", decision=ReviewStatus.APPROVED, reviewer_id="human1")
        result = queue.process_decision(decision)
        assert result.status == ReviewStatus.APPROVED

    def test_priority_ordering(self, queue: ReviewQueue) -> None:
        queue.add_item(ReviewItem(item_id="low", content={}, confidence=0.5, priority=Priority.LOW))
        queue.add_item(ReviewItem(item_id="urgent", content={}, confidence=0.5, priority=Priority.URGENT))
        pending = queue.get_pending()
        assert pending[0].item_id == "urgent"


class TestActiveLearner:
    def test_record_and_accuracy(self, learner: ActiveLearner) -> None:
        learner.record_feedback(FeedbackRecord(item_id="f1", original_output={}, corrected_output=None, was_correct=True))
        learner.record_feedback(FeedbackRecord(item_id="f2", original_output={}, corrected_output={"fix": "x"}, was_correct=False))
        assert learner.get_accuracy() == 0.5

    def test_should_retrain(self, learner: ActiveLearner) -> None:
        for i in range(50):
            learner.record_feedback(FeedbackRecord(item_id=f"f{i}", original_output={}, corrected_output={}, was_correct=False))
        assert learner.should_retrain() is True
