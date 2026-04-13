"""
单元测试 -- 覆盖核心模块和Agent逻辑。

运行方式：
    cd python/
    python -m pytest tests/ -v
"""

import asyncio
import pytest

from core.event_bus import EventBus, Event, EventType
from core.learner_model import LearnerModel, MasteryLevel
from core.spaced_repetition import SpacedRepetition, ReviewItem
from core.knowledge_graph import KnowledgeGraph, KnowledgeNode, build_sample_math_graph


# ─── EventBus Tests ───


@pytest.mark.asyncio
async def test_event_bus_publish_subscribe():
    bus = EventBus()
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.subscribe(EventType.STUDENT_SUBMISSION, handler)

    event = Event(
        type=EventType.STUDENT_SUBMISSION,
        source="test",
        learner_id="student_1",
        data={"knowledge_id": "arithmetic", "is_correct": True},
    )
    await bus.publish(event)

    assert len(received) == 1
    assert received[0].data["is_correct"] is True


@pytest.mark.asyncio
async def test_event_bus_no_handler():
    bus = EventBus()
    event = Event(
        type=EventType.STUDENT_SUBMISSION,
        source="test",
        learner_id="student_1",
    )
    await bus.publish(event)  # Should not raise


@pytest.mark.asyncio
async def test_event_bus_history():
    bus = EventBus()
    for i in range(5):
        await bus.publish(
            Event(
                type=EventType.STUDENT_SUBMISSION,
                source="test",
                learner_id=f"student_{i % 2}",
            )
        )
    assert len(bus.get_history()) == 5
    assert len(bus.get_history(learner_id="student_0")) == 3


# ─── LearnerModel / BKT Tests ───


def test_bkt_correct_answer_increases_mastery():
    model = LearnerModel("student_1")
    state = model.get_state("arithmetic")
    initial = state.mastery

    state = model.update_mastery("arithmetic", is_correct=True)
    assert state.mastery > initial


def test_bkt_wrong_answer_updates_mastery():
    model = LearnerModel("student_1")
    model.update_mastery("arithmetic", is_correct=True)
    model.update_mastery("arithmetic", is_correct=True)
    state_after_correct = model.get_state("arithmetic").mastery

    model.update_mastery("arithmetic", is_correct=False)
    state_after_wrong = model.get_state("arithmetic").mastery
    assert state_after_wrong < state_after_correct


def test_mastery_level_progression():
    model = LearnerModel("student_1")
    state = model.get_state("arithmetic")
    assert state.level == MasteryLevel.NOT_STARTED

    for _ in range(20):
        model.update_mastery("arithmetic", is_correct=True)

    state = model.get_state("arithmetic")
    assert state.level == MasteryLevel.MASTERED


def test_weak_points():
    model = LearnerModel("student_1")
    model.update_mastery("algebra", is_correct=False)
    model.update_mastery("algebra", is_correct=False)
    model.update_mastery("geometry", is_correct=True)
    model.update_mastery("geometry", is_correct=True)

    weak = model.get_weak_points(threshold=0.5)
    assert any(s.knowledge_id == "algebra" for s in weak)


def test_overall_progress():
    model = LearnerModel("student_1")
    model.update_mastery("a", is_correct=True)
    model.update_mastery("b", is_correct=False)

    progress = model.get_overall_progress()
    assert progress["total_knowledge_points"] == 2
    assert progress["total_attempts"] == 2


# ─── SpacedRepetition / SM-2 Tests ───


def test_sm2_correct_increases_interval():
    sr = SpacedRepetition()
    item = ReviewItem(knowledge_id="test")

    item = sr.review(item, quality=5)
    assert item.interval_days == 1
    assert item.repetition == 1

    item = sr.review(item, quality=5)
    assert item.interval_days == 6
    assert item.repetition == 2

    item = sr.review(item, quality=5)
    assert item.interval_days > 6


def test_sm2_bad_quality_resets():
    sr = SpacedRepetition()
    item = ReviewItem(knowledge_id="test")

    item = sr.review(item, quality=5)
    item = sr.review(item, quality=5)
    item = sr.review(item, quality=5)

    item = sr.review(item, quality=1)
    assert item.repetition == 0
    assert item.interval_days == 1


def test_sm2_ef_adjustment():
    sr = SpacedRepetition()
    item = ReviewItem(knowledge_id="test")
    initial_ef = item.easiness_factor

    item = sr.review(item, quality=5)
    assert item.easiness_factor >= initial_ef

    item2 = ReviewItem(knowledge_id="test2")
    item2 = sr.review(item2, quality=2)
    assert item2.easiness_factor < initial_ef


def test_sm2_due_items():
    sr = SpacedRepetition()
    item1 = ReviewItem(knowledge_id="a")
    item2 = ReviewItem(knowledge_id="b")

    due = sr.get_due_items([item1, item2])
    assert len(due) == 2  # Both are due (next_review = now)


# ─── KnowledgeGraph Tests ───


def test_graph_topological_sort():
    graph = KnowledgeGraph()
    graph.add_node(KnowledgeNode(id="a", name="A"))
    graph.add_node(KnowledgeNode(id="b", name="B", prerequisites=["a"]))
    graph.add_node(KnowledgeNode(id="c", name="C", prerequisites=["a", "b"]))

    order = graph.topological_sort()
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


def test_graph_ready_nodes():
    graph = KnowledgeGraph()
    graph.add_node(KnowledgeNode(id="a", name="A"))
    graph.add_node(KnowledgeNode(id="b", name="B", prerequisites=["a"]))
    graph.add_node(KnowledgeNode(id="c", name="C", prerequisites=["a"]))

    ready = graph.get_ready_nodes(mastered_ids=set())
    assert ready == ["a"]

    ready = graph.get_ready_nodes(mastered_ids={"a"})
    assert set(ready) == {"b", "c"}


def test_graph_learning_path():
    graph = KnowledgeGraph()
    graph.add_node(KnowledgeNode(id="a", name="A"))
    graph.add_node(KnowledgeNode(id="b", name="B", prerequisites=["a"]))
    graph.add_node(KnowledgeNode(id="c", name="C", prerequisites=["b"]))

    path = graph.get_learning_path("c", mastered_ids=set())
    assert path == ["a", "b", "c"]

    path = graph.get_learning_path("c", mastered_ids={"a"})
    assert path == ["b", "c"]


def test_sample_math_graph():
    graph = build_sample_math_graph()
    assert len(graph.nodes) == 20
    order = graph.topological_sort()
    assert order[0] == "arithmetic"
