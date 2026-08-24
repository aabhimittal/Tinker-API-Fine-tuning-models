"""Distribution drift between corpora - the check that explains a "mystery" regression.

Validation asks "is this dataset well-formed?". Drift asks the harder,
production question: "is this dataset *the same shape* as the one the model was
tuned on?". A refreshed corpus that is 3x longer per example, or whose answers
suddenly all start with "Sure!", will train and evaluate cleanly while quietly
changing the model.

Implemented here, dependency-free and deterministic:

- :func:`psi` - Population Stability Index, the standard for "has this feature
  moved?" with the usual 0.1 / 0.25 action thresholds.
- :func:`js_divergence` - symmetric, bounded (0..1 in bits), safe on disjoint
  supports where KL is infinite.
- :func:`ks_statistic` - distribution-free two-sample test statistic for
  continuous features (lengths, rewards, scores).
- :func:`compare_distributions` / :func:`compare_datasets` - the report:
  per-feature drift over prompt/response length, turn count, role structure,
  vocabulary and response prefixes, graded to the same severities the dataset
  validator uses.

Bins are shared between baseline and candidate (quantile bins from the
baseline), zero counts are epsilon-smoothed, and every statistic is finite even
for empty overlaps - a drift check that raises is a drift check nobody runs.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from tinker_finetune.data.validation import Severity
from tinker_finetune.models.schemas import ChatExample, Role

__all__ = [
    "EPSILON",
    "psi",
    "js_divergence",
    "ks_statistic",
    "quantile_bins",
    "bucketize",
    "FeatureDrift",
    "DriftReport",
    "compare_distributions",
    "compare_categorical",
    "compare_datasets",
]

EPSILON = 1e-6
PSI_WARN = 0.1
PSI_ERROR = 0.25
_WORD = re.compile(r"[a-z0-9']+")


def _clean(values: Iterable[float], *, name: str = "values") -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value ({value})")
        out.append(value)
    return out


def quantile_bins(values: Sequence[float], num_bins: int = 10) -> list[float]:
    """Interior bin edges from the baseline's quantiles.

    Equal-width bins are useless for token lengths (one bin holds 99% of the
    mass); quantile bins put the resolution where the data is. Degenerate
    baselines - all one value, fewer rows than bins - collapse to fewer edges
    rather than producing duplicates that would create empty bins.
    """
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    ordered = sorted(_clean(values, name="baseline"))
    if not ordered:
        raise ValueError("cannot build bins from an empty baseline")
    uniques = sorted(set(ordered))
    if len(uniques) == 1:
        # A constant baseline gets a single edge at that value, so a candidate
        # that moves off the constant still lands in a different bucket.
        return [uniques[0]]
    # More bins than distinct values only manufactures empty bins, which inflate
    # PSI; cap the resolution at what the baseline can actually support.
    num_bins = min(num_bins, len(uniques))
    edges: list[float] = []
    for i in range(1, num_bins):
        pos = i * (len(ordered) - 1) / num_bins
        low = math.floor(pos)
        high = min(low + 1, len(ordered) - 1)
        edge = ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
        if not edges or edge > edges[-1]:
            edges.append(edge)
    return edges


def bucketize(values: Iterable[float], edges: Sequence[float]) -> list[int]:
    """Count values into ``len(edges) + 1`` buckets defined by interior ``edges``."""
    counts = [0] * (len(edges) + 1)
    for value in _clean(values, name="values"):
        index = 0
        for edge in edges:
            if value <= edge:
                break
            index += 1
        counts[index] += 1
    return counts


def _proportions(counts: Sequence[int]) -> list[float]:
    total = sum(counts)
    if total <= 0:
        return [0.0] * len(counts)
    return [max(c / total, EPSILON) for c in counts]


def psi(baseline: Sequence[int], candidate: Sequence[int]) -> float:
    """Population Stability Index over aligned bin counts.

    ``sum((a - b) * ln(a / b))``. Symmetric, zero for identical distributions,
    and finite here because empty bins are floored at :data:`EPSILON`.
    """
    if len(baseline) != len(candidate):
        raise ValueError("bin counts must have the same length")
    if not baseline:
        return 0.0
    if sum(baseline) == 0 or sum(candidate) == 0:
        return float("inf") if sum(baseline) != sum(candidate) else 0.0
    b, c = _proportions(baseline), _proportions(candidate)
    return sum((ci - bi) * math.log(ci / bi) for bi, ci in zip(b, c, strict=True))


def js_divergence(baseline: Sequence[int], candidate: Sequence[int]) -> float:
    """Jensen-Shannon divergence in bits: 0 (identical) to 1 (disjoint supports)."""
    if len(baseline) != len(candidate):
        raise ValueError("bin counts must have the same length")
    if not baseline or sum(baseline) == 0 or sum(candidate) == 0:
        return 0.0 if sum(baseline) == sum(candidate) else 1.0
    b, c = _proportions(baseline), _proportions(candidate)
    bt, ct = sum(b), sum(c)
    b = [x / bt for x in b]
    c = [x / ct for x in c]

    def kl(p: Sequence[float], q: Sequence[float]) -> float:
        return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q, strict=True) if pi > 0)

    m = [(bi + ci) / 2 for bi, ci in zip(b, c, strict=True)]
    return max(0.0, min(1.0, 0.5 * kl(b, m) + 0.5 * kl(c, m)))


def ks_statistic(baseline: Sequence[float], candidate: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap), 0..1."""
    a = sorted(_clean(baseline, name="baseline"))
    b = sorted(_clean(candidate, name="candidate"))
    if not a or not b:
        return 0.0 if not a and not b else 1.0
    i = j = 0
    best = 0.0
    while i < len(a) and j < len(b):
        value = min(a[i], b[j])
        while i < len(a) and a[i] <= value:
            i += 1
        while j < len(b) and b[j] <= value:
            j += 1
        best = max(best, abs(i / len(a) - j / len(b)))
    return best


@dataclass
class FeatureDrift:
    """Drift for one named feature, already graded."""

    feature: str
    psi: float
    jsd: float
    ks: float | None
    severity: Severity
    baseline_summary: dict[str, float] = field(default_factory=dict)
    candidate_summary: dict[str, float] = field(default_factory=dict)
    detail: str = ""

    @property
    def drifted(self) -> bool:
        return self.severity is not Severity.info


@dataclass
class DriftReport:
    features: list[FeatureDrift] = field(default_factory=list)
    baseline_rows: int = 0
    candidate_rows: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def severity(self) -> Severity:
        if any(f.severity is Severity.error for f in self.features):
            return Severity.error
        if any(f.severity is Severity.warning for f in self.features):
            return Severity.warning
        return Severity.info

    @property
    def ok(self) -> bool:
        return self.severity is not Severity.error

    def drifted(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.drifted]

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "baseline_rows": self.baseline_rows,
            "candidate_rows": self.candidate_rows,
            "notes": list(self.notes),
            "features": [
                {
                    "feature": f.feature,
                    "psi": round(f.psi, 6),
                    "jsd": round(f.jsd, 6),
                    "ks": None if f.ks is None else round(f.ks, 6),
                    "severity": f.severity.value,
                    "detail": f.detail,
                    "baseline": f.baseline_summary,
                    "candidate": f.candidate_summary,
                }
                for f in self.features
            ],
        }

    def render(self) -> str:
        lines = [f"drift {self.severity.value}: {self.baseline_rows} baseline vs {self.candidate_rows} candidate rows"]
        lines += [f"  [{f.severity.value}] {f.feature}: psi={f.psi:.3f} jsd={f.jsd:.3f} {f.detail}".rstrip() for f in self.features]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _grade(value: float, warn: float, error: float) -> Severity:
    if value >= error:
        return Severity.error
    if value >= warn:
        return Severity.warning
    return Severity.info


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    return {
        "n": float(len(ordered)),
        "mean": round(mean, 4),
        "p50": round(ordered[len(ordered) // 2], 4),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
        "max": round(ordered[-1], 4),
    }


def compare_distributions(
    feature: str,
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    num_bins: int = 10,
    warn: float = PSI_WARN,
    error: float = PSI_ERROR,
) -> FeatureDrift:
    """Numeric drift for one feature (lengths, rewards, scores)."""
    base = _clean(baseline, name="baseline")
    cand = _clean(candidate, name="candidate")
    if not base or not cand:
        return FeatureDrift(
            feature=feature,
            psi=0.0,
            jsd=0.0,
            ks=None,
            severity=Severity.warning,
            baseline_summary=_summary(base),
            candidate_summary=_summary(cand),
            detail="one side is empty; drift is undefined",
        )
    edges = quantile_bins(base, num_bins)
    b_counts, c_counts = bucketize(base, edges), bucketize(cand, edges)
    score = psi(b_counts, c_counts)
    drift = FeatureDrift(
        feature=feature,
        psi=score,
        jsd=js_divergence(b_counts, c_counts),
        ks=ks_statistic(base, cand),
        severity=_grade(score, warn, error),
        baseline_summary=_summary(base),
        candidate_summary=_summary(cand),
    )
    if min(len(base), len(cand)) < 30:
        # PSI on tiny samples is mostly sampling noise; say so instead of alarming.
        drift.detail = "small sample (<30 rows); treat as indicative only"
        if drift.severity is Severity.error:
            drift.severity = Severity.warning
    return drift


def compare_categorical(
    feature: str,
    baseline: Iterable[str],
    candidate: Iterable[str],
    *,
    top_k: int = 50,
    warn: float = PSI_WARN,
    error: float = PSI_ERROR,
) -> FeatureDrift:
    """Drift over a categorical/vocabulary feature, using the baseline's top-k plus 'other'."""
    b_counts_all = Counter(baseline)
    c_counts_all = Counter(candidate)
    if not b_counts_all or not c_counts_all:
        return FeatureDrift(
            feature=feature,
            psi=0.0,
            jsd=0.0,
            ks=None,
            severity=Severity.warning,
            detail="one side is empty; drift is undefined",
        )
    keys = [k for k, _ in b_counts_all.most_common(top_k)]
    b_vec = [b_counts_all[k] for k in keys] + [sum(b_counts_all.values()) - sum(b_counts_all[k] for k in keys)]
    c_vec = [c_counts_all[k] for k in keys] + [sum(c_counts_all.values()) - sum(c_counts_all[k] for k in keys)]
    score = psi(b_vec, c_vec)
    unseen = sum(count for key, count in c_counts_all.items() if key not in b_counts_all)
    oov = unseen / max(1, sum(c_counts_all.values()))
    return FeatureDrift(
        feature=feature,
        psi=score,
        jsd=js_divergence(b_vec, c_vec),
        ks=None,
        severity=_grade(score, warn, error),
        baseline_summary={"distinct": float(len(b_counts_all))},
        candidate_summary={"distinct": float(len(c_counts_all)), "unseen_share": round(oov, 4)},
        detail=f"{oov:.1%} of candidate mass is unseen in baseline" if oov > 0.05 else "",
    )


def _features(examples: Sequence[ChatExample]) -> dict[str, list[float]]:
    prompt_chars: list[float] = []
    answer_chars: list[float] = []
    turns: list[float] = []
    for example in examples:
        msgs = example.messages
        prompt = " ".join(m.content or "" for m in msgs if m.role is not Role.assistant)
        answer = " ".join(m.content or "" for m in msgs if m.role is Role.assistant)
        prompt_chars.append(float(len(prompt)))
        answer_chars.append(float(len(answer)))
        turns.append(float(len(msgs)))
    return {"prompt_chars": prompt_chars, "answer_chars": answer_chars, "turns": turns}


def _first_words(examples: Sequence[ChatExample], words: int = 2) -> list[str]:
    out: list[str] = []
    for example in examples:
        answer = next((m.content or "" for m in example.messages if m.role is Role.assistant), "")
        tokens = _WORD.findall(answer.lower())[:words]
        out.append(" ".join(tokens) if tokens else "<empty>")
    return out


def _vocab(examples: Sequence[ChatExample], *, cap: int = 200) -> list[str]:
    out: list[str] = []
    for example in examples:
        for message in example.messages:
            out.extend(_WORD.findall((message.content or "").lower())[:cap])
    return out


def compare_datasets(
    baseline: Sequence[ChatExample],
    candidate: Sequence[ChatExample],
    *,
    num_bins: int = 10,
    top_k: int = 50,
) -> DriftReport:
    """Full drift report between two chat corpora (e.g. last month's vs this month's).

    Covers length, structure, vocabulary and answer-prefix drift - the last one
    catches style collapse ("Sure! Here's...") that length statistics miss.
    """
    report = DriftReport(baseline_rows=len(baseline), candidate_rows=len(candidate))
    if not baseline or not candidate:
        report.notes.append("one corpus is empty; nothing to compare")
        report.features.append(
            FeatureDrift(
                feature="corpus",
                psi=0.0,
                jsd=0.0,
                ks=None,
                severity=Severity.error,
                detail="empty corpus",
            )
        )
        return report

    b_feats, c_feats = _features(baseline), _features(candidate)
    for name in ("prompt_chars", "answer_chars", "turns"):
        report.features.append(compare_distributions(name, b_feats[name], c_feats[name], num_bins=num_bins))
    report.features.append(compare_categorical("answer_prefix", _first_words(baseline), _first_words(candidate), top_k=top_k))
    report.features.append(compare_categorical("vocabulary", _vocab(baseline), _vocab(candidate), top_k=top_k))

    ratio = len(candidate) / len(baseline)
    if ratio < 0.5 or ratio > 2.0:
        report.notes.append(f"corpus size changed {ratio:.2f}x; drift statistics compare shape, not volume")
    return report
