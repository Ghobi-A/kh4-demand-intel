"""Headless Streamlit startup check: every tab renders without an exception."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")


def test_dashboard_renders_every_tab_without_exceptions() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()

    assert not app.exception, [str(e.value) for e in app.exception]
    # The per-tab guard reports failures as st.error; none should fire.
    assert not [e.value for e in app.error if "could not be rendered" in str(e.value)]
    labels = [tab.label for tab in app.tabs]
    # Events is hidden in a checkout without the local event outputs.
    assert len(labels) == 11 and "Events" not in labels, labels
    assert app.tabs[0].label == "Overview"
    overview_text = " ".join(str(m.value) for m in app.tabs[0].markdown)
    assert "Most defensible findings" in " ".join(str(h.value) for h in app.subheader)
    assert "Not Square Enix data" in " ".join(str(c.value) for c in app.caption)
    assert "Sentiment is not behavioural intent" in overview_text


def test_live_inference_uses_a_trained_model_not_the_rule_fallback(tmp_path) -> None:
    """Regression: a deployed checkout has no artefacts and silently used rules."""
    from src.ml import inference

    inference._LIVE_MODEL.clear()
    model_dir, info = inference.ensure_live_model(artifact_dir=tmp_path / "absent")
    try:
        assert model_dir is not None, info
        assert info["source"] == "trained_this_session"
        result = inference.predict_text("Pre-ordering the second it is out", artifact_dir=model_dir)
        assert result["model_name"] != "rules"
        assert 0.0 <= result["actionable_probability"] <= 1.0
    finally:
        inference._LIVE_MODEL.clear()


def test_live_inference_banner_states_the_supervised_method() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180).run()
    banners = [str(s.value) for s in app.success]
    assert any("Active inference method: supervised two-stage" in b for b in banners), banners
    assert not any("rule baseline (fallback)" in str(w.value) for w in app.warning)


def test_comparison_and_error_filters_survive_interaction() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()
    app.radio(key="comparison_protocol").set_value("Near-duplicate linked groups (classical models)").run()
    assert "embedding" not in app.selectbox(key="paired_reference").options
    app.selectbox(key="paired_reference").select("char_word_svm")
    app.selectbox(key="paired_metric").select("weighted_pr_auc")
    app.select_slider(key="review_budget").set_value(60)
    app.selectbox(key="error_model").select("char_word_svm")
    app.selectbox(key="error_kind").select("false_negative")
    app.run()
    assert not app.exception
    assert not [e.value for e in app.error if "could not be rendered" in str(e.value)]
    assert any("assumption sensitivity" in str(w.value) for w in app.warning)
