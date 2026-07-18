"""Supervised ML layer for KH4 demand intelligence.

This package is separable from the heuristic scoring layer in
``src/score.py``. It distinguishes four concerns:

- classification (rule baseline + supervised models)
- probability estimation (calibration)
- ranking (actionable-signal prioritisation)
- decision-support scoring (existing heuristic layer, unchanged)

All results produced from the current 200-row audit corpus are
development-preliminary until the corpus is re-audited and expanded.
"""

EVALUATION_STATUS_DEVELOPMENT = "development_preliminary"
EVALUATION_STATUS_FINAL = "final_held_out"
