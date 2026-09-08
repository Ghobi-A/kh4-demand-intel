"""Canonical timestamp handling for KH4 signal data.

Every signal carries the *creation* time supplied by the source API — never
the scrape time. This module is the single place where raw timestamp values
(ISO strings, epoch seconds, datetimes) become a timezone-aware UTC series,
where invalid values are reported rather than silently coerced, and where
missing values stay missing rather than being fabricated.

It also provides the temporal-coverage audit that the forecasting layer uses
to decide how much model complexity the observed history can actually
support.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Serialised form used everywhere a timestamp is written back to CSV.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

STATUS_VALID = "valid"
STATUS_MISSING = "missing"
STATUS_INVALID = "invalid"

# Epoch-zero values are a well-known fabrication artefact: an API response
# without ``created_utc`` used to default to 0, producing a 1970 timestamp
# that looks real. Anything in this window is treated as invalid, not as data.
_EPOCH_ZERO_CUTOFF = pd.Timestamp("1971-01-01", tz="UTC")

# Content cannot have been created meaningfully in the future; a small margin
# absorbs clock skew between the source API and this machine.
_FUTURE_MARGIN = pd.Timedelta(days=1)


@dataclass
class TimestampReport:
    """Counts describing what happened while parsing a timestamp column."""

    total: int = 0
    valid: int = 0
    missing: int = 0
    invalid: int = 0
    epoch_zero_fabricated: int = 0
    future: int = 0
    naive_rejected: int = 0
    earliest: str | None = None
    latest: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def log(self, context: str = "timestamps") -> None:
        LOGGER.info(
            "%s: %s rows — %s valid, %s missing, %s invalid "
            "(epoch-zero %s, future %s, tz-naive rejected %s)",
            context,
            self.total,
            self.valid,
            self.missing,
            self.invalid,
            self.epoch_zero_fabricated,
            self.future,
            self.naive_rejected,
        )


def _coerce_numeric_epochs(raw: pd.Series) -> pd.Series:
    """Return a datetime series for values that look like epoch seconds."""
    numeric = pd.to_numeric(raw, errors="coerce")
    return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")


_TZ_SUFFIX = re.compile(r"(Z|z|[+-]\d{2}:?\d{2})$")


def _is_timezone_naive(value) -> bool:
    """Return True when a raw value carries no timezone information at all."""
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return value.tzinfo is None
    text = str(value).strip()
    if not text:
        return False
    return _TZ_SUFFIX.search(text) is None


def period_start(values, freq: str = "W-MON") -> pd.Series:
    """Return the start of the period each timestamp falls in, in UTC.

    Weekly periods start on Monday. Computing the boundary directly keeps the
    series timezone-aware; converting through pandas ``Period`` silently drops
    the timezone, which would shift every bucket for non-UTC inputs.
    """
    series = pd.Series(values)
    if not pd.api.types.is_datetime64_any_dtype(series):
        series, _ = parse_timestamps(series)
    if series.empty:
        return series
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    else:
        series = series.dt.tz_convert("UTC")
    normalised = series.dt.normalize()
    if freq.upper().startswith("W"):
        return normalised - pd.to_timedelta(normalised.dt.dayofweek, unit="D")
    if freq.upper().startswith("D"):
        return normalised
    raise ValueError(f"Unsupported frequency {freq!r}; use 'W-MON' or 'D'")


def period_range(start: pd.Timestamp, end: pd.Timestamp, freq: str = "W-MON") -> pd.DatetimeIndex:
    """Complete, gap-free index of period starts covering ``start``..``end``."""
    step = "7D" if freq.upper().startswith("W") else "1D"
    return pd.date_range(start=start, end=end, freq=step, tz="UTC")


def parse_timestamps(
    values,
    assume_utc: bool = False,
) -> tuple[pd.Series, TimestampReport]:
    """Parse raw timestamp values into a tz-aware UTC series plus a report.

    Accepts ISO-8601 strings, epoch seconds (numeric or numeric-as-string) and
    datetime objects. Missing values stay missing (``NaT``). Values that cannot
    be parsed, that fall in the epoch-zero window, or that lie in the future are
    reported as invalid and returned as ``NaT`` — they are never repaired by
    substituting another column such as the scrape time.

    Timezone-naive inputs are rejected as invalid unless ``assume_utc`` is set,
    because guessing an offset silently shifts every downstream weekly bucket.
    """
    raw = pd.Series(values).reset_index(drop=True)
    report = TimestampReport(total=int(len(raw)))
    if raw.empty:
        return pd.Series([], dtype="datetime64[ns, UTC]"), report

    missing_mask = raw.isna()
    if raw.dtype == object:
        blank = raw.astype("string").str.strip().isin(["", "nan", "None", "NaT"])
        missing_mask = missing_mask | blank.fillna(False)

    numeric_like = pd.to_numeric(raw, errors="coerce").notna() & ~missing_mask

    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")

    if numeric_like.any():
        parsed.loc[numeric_like] = _coerce_numeric_epochs(raw[numeric_like])

    textual = ~numeric_like & ~missing_mask
    naive_mask = pd.Series(False, index=raw.index)
    if textual.any():
        subset = raw[textual]
        naive_mask.loc[subset.index] = subset.map(_is_timezone_naive)
        parsed.loc[textual] = pd.to_datetime(subset, errors="coerce", utc=True, format="mixed")

    if not assume_utc and naive_mask.any():
        report.naive_rejected = int(naive_mask.sum())
        parsed.loc[naive_mask] = pd.NaT

    unparsed = parsed.isna() & ~missing_mask

    epoch_zero = parsed.notna() & (parsed < _EPOCH_ZERO_CUTOFF)
    report.epoch_zero_fabricated = int(epoch_zero.sum())
    parsed.loc[epoch_zero] = pd.NaT

    horizon = pd.Timestamp.now(tz="UTC") + _FUTURE_MARGIN
    future = parsed.notna() & (parsed > horizon)
    report.future = int(future.sum())
    parsed.loc[future] = pd.NaT

    report.missing = int(missing_mask.sum())
    report.invalid = int(unparsed.sum()) + report.epoch_zero_fabricated + report.future
    report.valid = int(parsed.notna().sum())
    if report.valid:
        report.earliest = parsed.min().isoformat()
        report.latest = parsed.max().isoformat()

    return parsed, report


def timestamp_status(parsed: pd.Series, raw=None) -> pd.Series:
    """Label each row ``valid`` / ``missing`` / ``invalid``.

    A row is ``missing`` when the source supplied nothing at all and
    ``invalid`` when it supplied something unusable — the distinction matters
    for data quality reporting and for deciding whether re-fetching helps.
    """
    parsed = pd.Series(parsed).reset_index(drop=True)
    status = pd.Series(STATUS_VALID, index=parsed.index, dtype="object")
    status[parsed.isna()] = STATUS_MISSING
    if raw is not None:
        raw_series = pd.Series(raw).reset_index(drop=True)
        supplied = raw_series.notna()
        if raw_series.dtype == object:
            blank = raw_series.astype("string").str.strip().isin(["", "nan", "None", "NaT"])
            supplied = supplied & ~blank.fillna(False)
        status[parsed.isna() & supplied] = STATUS_INVALID
    return status


def serialise_timestamps(values) -> pd.Series:
    """Serialise timestamps deterministically as ``YYYY-MM-DDTHH:MM:SS+00:00``.

    Missing values become an empty string so a round-trip through CSV keeps
    them missing rather than turning them into the literal text ``NaT``.
    """
    series = pd.Series(values)
    if series.empty:
        return pd.Series([], dtype="object")
    if not pd.api.types.is_datetime64_any_dtype(series):
        series, _ = parse_timestamps(series)
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    else:
        series = series.dt.tz_convert("UTC")
    formatted = series.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return formatted.where(series.notna(), "")


def _quantiles(counts: pd.Series) -> dict:
    if counts.empty:
        return {}
    return {
        "min": float(counts.min()),
        "p25": float(counts.quantile(0.25)),
        "median": float(counts.median()),
        "p75": float(counts.quantile(0.75)),
        "max": float(counts.max()),
        "mean": float(counts.mean()),
    }


def _gap_runs(counts: pd.Series, min_gap: int) -> list[dict]:
    """Return runs of consecutive empty periods of at least ``min_gap``."""
    gaps: list[dict] = []
    run_start: pd.Timestamp | None = None
    run_length = 0
    previous: pd.Timestamp | None = None
    for period, value in counts.items():
        if value == 0:
            if run_start is None:
                run_start = period
                run_length = 0
            run_length += 1
            previous = period
        else:
            if run_start is not None and run_length >= min_gap:
                gaps.append(
                    {
                        "start": run_start.isoformat(),
                        "end": previous.isoformat() if previous is not None else None,
                        "periods": run_length,
                    }
                )
            run_start = None
            run_length = 0
    if run_start is not None and run_length >= min_gap:
        gaps.append(
            {
                "start": run_start.isoformat(),
                "end": previous.isoformat() if previous is not None else None,
                "periods": run_length,
            }
        )
    return gaps


@dataclass
class CoverageAudit:
    """Measured temporal information available for modelling."""

    earliest: str | None = None
    latest: str | None = None
    observed_periods: int = 0
    non_empty_periods: int = 0
    rows_with_timestamp: int = 0
    rows_without_timestamp: int = 0
    freq: str = "W-MON"
    comments_per_period: dict = field(default_factory=dict)
    source_coverage: dict = field(default_factory=dict)
    gaps: list = field(default_factory=list)
    usable_event_periods: int = 0
    event_details: list = field(default_factory=list)
    longest_dense_run: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def audit_temporal_coverage(
    df: pd.DataFrame,
    freq: str = "W-MON",
    timestamp_column: str = "timestamp",
    events: pd.DataFrame | None = None,
    min_event_periods: int = 2,
    min_gap: int = 2,
) -> CoverageAudit:
    """Measure how much temporal information the dataset actually contains.

    The forecasting layer calls this *before* choosing a model so that model
    complexity follows the observed history rather than the other way round.
    """
    parsed, _ = parse_timestamps(df[timestamp_column]) if timestamp_column in df.columns else (
        pd.Series([], dtype="datetime64[ns, UTC]"),
        TimestampReport(),
    )
    audit = CoverageAudit(freq=freq)
    audit.rows_without_timestamp = int(parsed.isna().sum())
    valid = parsed.dropna()
    audit.rows_with_timestamp = int(len(valid))
    if valid.empty:
        return audit

    audit.earliest = valid.min().isoformat()
    audit.latest = valid.max().isoformat()

    periods = period_start(valid, freq)
    counts = periods.value_counts().sort_index()
    full_index = period_range(counts.index.min(), counts.index.max(), freq)
    counts = counts.reindex(full_index, fill_value=0)

    audit.observed_periods = int(len(counts))
    audit.non_empty_periods = int((counts > 0).sum())
    audit.comments_per_period = _quantiles(counts)
    audit.gaps = _gap_runs(counts, min_gap=min_gap)

    non_empty = (counts > 0).astype(int).tolist()
    longest = current = 0
    for value in non_empty:
        current = current + 1 if value else 0
        longest = max(longest, current)
    audit.longest_dense_run = longest

    if "source" in df.columns:
        sources = df["source"].reset_index(drop=True)[parsed.notna().values]
        frame = pd.DataFrame({"period": periods.values, "source": sources.values})
        by_source = frame.groupby("source")["period"].nunique().to_dict()
        audit.source_coverage = {str(k): int(v) for k, v in by_source.items()}

    if events is not None and not events.empty and "event_date" in events.columns:
        event_dates = pd.to_datetime(events["event_date"], utc=True, errors="coerce").dropna()
        details = []
        for _, row in events.iterrows():
            event_date = pd.to_datetime(row.get("event_date"), utc=True, errors="coerce")
            if pd.isna(event_date):
                continue
            before = int((counts.index < event_date).sum())
            after = int((counts.index >= event_date).sum())
            usable = before >= min_event_periods and after >= min_event_periods
            details.append(
                {
                    "event_name": str(row.get("event_name", "")),
                    "event_date": event_date.isoformat(),
                    "periods_before": before,
                    "periods_after": after,
                    "usable": usable,
                }
            )
        audit.event_details = details
        audit.usable_event_periods = int(sum(1 for d in details if d["usable"]))
        del event_dates

    return audit


def write_coverage_report(audit: CoverageAudit, output_dir: Path, prefix: str = "temporal_coverage") -> tuple[Path, Path]:
    """Write the coverage audit as JSON plus a short Markdown summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    md_path = output_dir / f"{prefix}.md"

    payload = audit.to_dict()
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    per_period = audit.comments_per_period
    lines = [
        "# Temporal coverage audit",
        "",
        "Measured from the observed data before any model was chosen.",
        "",
        f"- Frequency: `{audit.freq}`",
        f"- Earliest observation: {audit.earliest or 'n/a'}",
        f"- Latest observation: {audit.latest or 'n/a'}",
        f"- Observed periods: {audit.observed_periods}",
        f"- Non-empty periods: {audit.non_empty_periods}",
        f"- Longest consecutive non-empty run: {audit.longest_dense_run}",
        f"- Rows with a usable timestamp: {audit.rows_with_timestamp}",
        f"- Rows without a usable timestamp: {audit.rows_without_timestamp}",
        f"- Usable event periods: {audit.usable_event_periods}",
        "",
    ]
    if per_period:
        lines += [
            "## Comments per period",
            "",
            "| Statistic | Value |",
            "|---|---|",
        ]
        for key in ["min", "p25", "median", "mean", "p75", "max"]:
            if key in per_period:
                lines.append(f"| {key} | {per_period[key]:.2f} |")
        lines.append("")
    if audit.source_coverage:
        lines += ["## Periods covered per source", "", "| Source | Periods |", "|---|---|"]
        for source, value in sorted(audit.source_coverage.items()):
            lines.append(f"| {source} | {value} |")
        lines.append("")
    if audit.gaps:
        lines += ["## Gaps (runs of empty periods)", "", "| Start | End | Periods |", "|---|---|---|"]
        for gap in audit.gaps:
            lines.append(f"| {gap['start'][:10]} | {str(gap['end'])[:10]} | {gap['periods']} |")
        lines.append("")
    if audit.event_details:
        lines += [
            "## Event usability",
            "",
            "| Event | Date | Periods before | Periods after | Usable |",
            "|---|---|---|---|---|",
        ]
        for detail in audit.event_details:
            lines.append(
                f"| {detail['event_name']} | {detail['event_date'][:10]} | "
                f"{detail['periods_before']} | {detail['periods_after']} | "
                f"{'yes' if detail['usable'] else 'no'} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines))
    return json_path, md_path


def complexity_tier(non_empty_periods: int) -> str:
    """Choose Bayesian model complexity from measured history length.

    The ladder is fixed in advance so the choice is a property of the data,
    not of the result: under 30 usable periods only an intercept, a linear
    trend and event effects are identifiable.
    """
    if non_empty_periods < 30:
        return "minimal"
    if non_empty_periods <= 60:
        return "moderate"
    return "rich"


__all__ = [
    "CoverageAudit",
    "STATUS_INVALID",
    "STATUS_MISSING",
    "STATUS_VALID",
    "TimestampReport",
    "audit_temporal_coverage",
    "complexity_tier",
    "parse_timestamps",
    "period_range",
    "period_start",
    "serialise_timestamps",
    "timestamp_status",
    "write_coverage_report",
]
