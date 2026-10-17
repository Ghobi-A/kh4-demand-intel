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
    assert len(app.tabs) == 12
    assert app.tabs[0].label == "Overview"
    overview_text = " ".join(str(m.value) for m in app.tabs[0].markdown)
    assert "Most defensible findings" in " ".join(str(h.value) for h in app.subheader)
    assert "Not Square Enix data" in " ".join(str(c.value) for c in app.caption)
    assert "Sentiment is not behavioural intent" in overview_text
