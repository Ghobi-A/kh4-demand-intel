"""Tests for rule-based intent classification."""

from src.intent import classify_intent


def test_generic_excited_is_general_discussion() -> None:
    label, _ = classify_intent("I love Kingdom Hearts so much")
    assert label == "general_discussion"


def test_cant_wait_is_general_discussion() -> None:
    label, _ = classify_intent("I can't wait for KH4")
    assert label == "general_discussion"




def test_regression_marvel_star_wars_feature_speculation_stays_general_discussion() -> None:
    label, _ = classify_intent(
        "Would be cool if KH4 adds Marvel team-up attacks and Star Wars ship combat"
    )
    assert label == "general_discussion"


def test_regression_love_kingdom_hearts_stays_general_discussion() -> None:
    label, _ = classify_intent("I love Kingdom Hearts so much")
    assert label == "general_discussion"

def test_high_intent_bought_series() -> None:
    label, _ = classify_intent("I've bought this series 3 times")
    assert label == "high_intent"


def test_high_intent_buying_every_copy() -> None:
    label, _ = classify_intent("I will be buying every copy of KH4")
    assert label == "high_intent"


def test_new_customer_access_signal() -> None:
    label, _ = classify_intent("Now that it's on Steam I can finally play")
    assert label == "new_customer_interest"


def test_star_wars_world_is_general_discussion() -> None:
    label, _ = classify_intent("I want a Star Wars world in KH4")
    assert label == "general_discussion"


def test_world_speculation_is_general_discussion() -> None:
    label, _ = classify_intent("I hope Kingdom Hearts 4 has Star Wars and Marvel worlds")
    assert label == "general_discussion"


def test_confusion_barrier_order_question() -> None:
    label, _ = classify_intent("What order do I play these games in?")
    assert label == "confusion_barrier"


def test_confusion_barrier_story_confusion() -> None:
    label, _ = classify_intent("This story is confusing as hell")
    assert label == "confusion_barrier"


def test_frustrated_demand_where_is_kh4() -> None:
    label, _ = classify_intent("Where is KH4?")
    assert label == "frustrated_demand"


def test_frustrated_demand_no_trailer() -> None:
    label, _ = classify_intent("No trailer in 4 years")
    assert label == "frustrated_demand"


def test_expectation_decay_dont_care_anymore() -> None:
    label, _ = classify_intent("I don't care anymore until they show something")
    assert label == "expectation_decay"


def test_expectation_decay_not_buying() -> None:
    label, _ = classify_intent("I'm not buying it")
    assert label == "expectation_decay"


def test_content_drought_fatigue_copium() -> None:
    label, _ = classify_intent("Gimme dat copium")
    assert label == "content_drought_fatigue"


def test_content_drought_fatigue_take_anything() -> None:
    label, _ = classify_intent("At this point I'll take anything")
    assert label == "content_drought_fatigue"


def test_nostalgia_reactivation_childhood() -> None:
    label, _ = classify_intent("This game was my childhood")
    assert label == "nostalgia_reactivation"


def test_nostalgia_reactivation_ps2_memory() -> None:
    label, _ = classify_intent("I remember playing KH2 on PS2")
    assert label == "nostalgia_reactivation"


def test_creator_praise_is_general_discussion() -> None:
    label, _ = classify_intent("Great video, another banger")
    assert label == "general_discussion"


def test_priority_expectation_decay_over_high_intent() -> None:
    label, _ = classify_intent("I was buying day one but now I don't care anymore")
    assert label == "expectation_decay"
