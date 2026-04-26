"""Tests for rule-based intent classification."""

from src.intent import classify_intent


def test_high_intent_label() -> None:
    label, _ = classify_intent("I will buy this day one")
    assert label == "high_intent"


def test_frustrated_demand_label() -> None:
    label, _ = classify_intent("its been 4 years no news")
    assert label == "frustrated_demand"


def test_nostalgia_reactivation_label() -> None:
    label, _ = classify_intent("KH2 was my childhood")
    assert label == "nostalgia_reactivation"


def test_new_customer_interest_label() -> None:
    label, _ = classify_intent("Marvel would bring new players")
    assert label == "new_customer_interest"


def test_confusion_barrier_label() -> None:
    label, _ = classify_intent("the story is too confusing")
    assert label == "confusion_barrier"
