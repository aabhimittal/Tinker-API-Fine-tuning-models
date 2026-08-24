"""Pre-flight dataset validation: the checks that save a fine-tuning budget.

Training is the expensive part of the loop, so every defect that can be caught
before the first ``forward_backward`` should be. This module implements the
checks that bite real production corpora:

- **PII leakage** (emails, phones, national ids, Luhn-valid card numbers,
  cloud/API credentials, private keys) with optional redaction.
- **Duplicates and near-duplicates** via a deterministic MinHash/LSH index, so
  a 100k-row corpus does not need 5e9 pairwise comparisons.
- **Train/eval contamination** - eval rows that also appear in train, exactly
  or near-exactly, which silently inflates held-out metrics.
- **Label conflicts** - the same prompt paired with contradictory answers.
- **Silent truncation** - examples whose assistant tokens fall outside
  ``max_seq_len`` and therefore contribute *zero* gradient signal.
- **Context overflow** against the base model's real context window.
- **Unicode hygiene** - lone surrogates, bidi overrides (trojan-source style),
  zero-width joiners, replacement characters, NFC mismatches.
- **Structural problems** - empty targets, misplaced system turns, consecutive
  same-role turns, response collapse (one answer dominating the corpus).

Findings are graded (``info`` / ``warning`` / ``error``) and aggregated, so a
report over a million rows stays a page long.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from tinker_finetune.data.templating import build_supervised_datum
from tinker_finetune.data.tokenization import Tokenizer, build_tokenizer
from tinker_finetune.models.registry import get_model, is_supported
from tinker_finetune.models.schemas import ChatExample, Message, Role

__all__ = [
    "Severity",
    "Finding",
    "ValidationReport",
    "DatasetValidationError",
    "validate_dataset",
    "redact_pii",
    "redact_examples",
    "NearDuplicateIndex",
    "PII_PATTERNS",
]


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


_ORDER = {Severity.info: 0, Severity.warning: 1, Severity.error: 2}


@dataclass(frozen=True)
class Finding:
    """One aggregated problem class, not one problem occurrence."""

    code: str
    severity: Severity
    message: str
    count: int = 1
    examples: tuple[int, ...] = ()  # up to a handful of offending row indices

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "count": self.count,
            "examples": list(self.examples),
        }


class DatasetValidationError(ValueError):
    """Raised by :meth:`ValidationReport.raise_for_severity`."""

    def __init__(self, findings: Sequence[Finding]) -> None:
        self.findings = list(findings)
        joined = "; ".join(f"[{f.code}] {f.message}" for f in findings)
        super().__init__(f"Dataset validation failed: {joined}")


@dataclass
class ValidationReport:
    path: str | None = None
    num_examples: int = 0
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    # -- queries ---------------------------------------------------------
    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.error]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.warning]

    @property
    def ok(self) -> bool:
        return not self.errors

    def by_code(self, code: str) -> Finding | None:
        return next((f for f in self.findings if f.code == code), None)

    def has(self, code: str) -> bool:
        return self.by_code(code) is not None

    def at_or_above(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if _ORDER[f.severity] >= _ORDER[severity]]

    def raise_for_severity(self, severity: Severity = Severity.error) -> None:
        """Raise if any finding is at or above ``severity`` (CI gate)."""
        bad = self.at_or_above(severity)
        if bad:
            raise DatasetValidationError(bad)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "num_examples": self.num_examples,
            "ok": self.ok,
            "findings": [f.to_dict() for f in self.findings],
            "stats": self.stats,
        }

    def summary(self) -> str:
        head = f"{self.num_examples} examples: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        lines = [head] + [
            f"  {f.severity.value:<7} {f.code:<24} {f.message}" for f in self.findings
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PII / secret detection
# ---------------------------------------------------------------------------
# Ordered most-specific first: a private key block should not be matched as a
# generic token, and a card number should not be reported as a phone number.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("api_key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ \-]?){12,18}\d\b")),  # Luhn-filtered below
    ("ssn", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    ("phone", re.compile(r"(?<![\d\-])(?:\+\d{1,3}[ \-]?)?(?:\(\d{3}\)|\d{3})[ \-]\d{3}[ \-]\d{4}(?![\d\-])")),
    ("ipv4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b")),
]

_LUHN_ONLY = {"credit_card"}


def _luhn_ok(digits: str) -> bool:
    """Standard mod-10 check; rejects the many 16-digit non-card numbers."""
    ds = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(ds) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return ``(redacted_text, kinds_found)``.

    Redaction is placeholder-based (``[REDACTED:email]``) rather than deletion
    so the surrounding sentence structure - which the model is learning -
    survives intact.
    """
    kinds: list[str] = []
    out = text
    for kind, pattern in PII_PATTERNS:
        def _sub(m: re.Match[str], _kind: str = kind) -> str:
            if _kind in _LUHN_ONLY and not _luhn_ok(m.group(0)):
                return m.group(0)
            kinds.append(_kind)
            return f"[REDACTED:{_kind}]"

        out = pattern.sub(_sub, out)
    return out, kinds


def redact_examples(examples: Iterable[ChatExample]) -> tuple[list[ChatExample], Counter]:
    """Redact PII across every message; returns new examples + per-kind counts."""
    counts: Counter = Counter()
    cleaned: list[ChatExample] = []
    for ex in examples:
        msgs = []
        for m in ex.messages:
            text, kinds = redact_pii(m.content)
            counts.update(kinds)
            msgs.append(Message(role=m.role, content=text))
        cleaned.append(ChatExample(messages=msgs))
    return cleaned, counts


# ---------------------------------------------------------------------------
# Deterministic MinHash / LSH near-duplicate index
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Casefold + NFKC + whitespace-collapse: the identity used for dupes."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _shingles(text: str, k: int = 5) -> set[str]:
    norm = _normalize(text)
    if len(norm) <= k:
        return {norm} if norm else set()
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


def _h(value: str, salt: int) -> int:
    # blake2b, not builtin hash(): PYTHONHASHSEED randomisation would make
    # duplicate detection non-reproducible across processes.
    # surrogatepass: corpora scraped from the wild do contain lone surrogates,
    # and hashing must not be the thing that crashes the validator.
    return int.from_bytes(
        hashlib.blake2b(
            value.encode("utf-8", "surrogatepass"), digest_size=8, salt=salt.to_bytes(2, "little")
        ).digest(),
        "big",
    )


class NearDuplicateIndex:
    """MinHash + banded LSH. Sub-quadratic candidate generation.

    ``num_perm`` signatures split into ``bands`` bands; two items are candidate
    duplicates when any band matches, and the exact Jaccard of their shingle
    sets is then compared against ``threshold``.
    """

    def __init__(self, *, threshold: float = 0.85, num_perm: int = 32, bands: int = 8, k: int = 5) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1].")
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands.")
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands = bands
        self.k = k
        self._buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
        self._shingles: dict[int, set[str]] = {}
        self._exact: dict[str, int] = {}

    def _signature(self, shingles: set[str]) -> list[int]:
        if not shingles:
            return [0] * self.num_perm
        return [min(_h(s, p) for s in shingles) for p in range(self.num_perm)]

    def _band_keys(self, sig: Sequence[int]) -> list[tuple[int, bytes]]:
        rows = self.num_perm // self.bands
        return [
            (b, hashlib.blake2b(
                b"".join(x.to_bytes(8, "big") for x in sig[b * rows : (b + 1) * rows]),
                digest_size=8,
            ).digest())
            for b in range(self.bands)
        ]

    def query(self, text: str) -> list[tuple[int, float]]:
        """Return ``(item_id, jaccard)`` for indexed items similar to ``text``."""
        norm = _normalize(text)
        if not norm:
            return []
        exact = self._exact.get(norm)
        if exact is not None:
            return [(exact, 1.0)]
        sh = _shingles(text, self.k)
        if not sh:
            return []
        seen: set[int] = set()
        hits: list[tuple[int, float]] = []
        for key in self._band_keys(self._signature(sh)):
            for other in self._buckets.get(key, ()):
                if other in seen:
                    continue
                seen.add(other)
                other_sh = self._shingles[other]
                union = len(sh | other_sh)
                jac = len(sh & other_sh) / union if union else 0.0
                if jac >= self.threshold:
                    hits.append((other, jac))
        return sorted(hits, key=lambda t: -t[1])

    def add(self, item_id: int, text: str) -> None:
        norm = _normalize(text)
        if norm:
            self._exact.setdefault(norm, item_id)
        sh = _shingles(text, self.k)
        self._shingles[item_id] = sh
        if not sh:
            return
        for key in self._band_keys(self._signature(sh)):
            self._buckets[key].append(item_id)

    def add_all(self, texts: Iterable[str], *, start: int = 0) -> None:
        for i, t in enumerate(texts, start=start):
            self.add(i, t)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = re.compile(r"[​-‍⁠﻿]")
_BIDI = re.compile(r"[‪-‮⁦-⁩]")
_MAX_EXAMPLES_PER_FINDING = 5


def _prompt_text(ex: ChatExample) -> str:
    """Everything up to the first assistant turn - the example's 'question'."""
    parts = []
    for m in ex.messages:
        if m.role is Role.assistant:
            break
        parts.append(m.content)
    return "\n".join(parts)


def _response_text(ex: ChatExample) -> str:
    return "\n".join(m.content for m in ex.messages if m.role is Role.assistant)


def validate_dataset(
    examples: Sequence[ChatExample],
    *,
    path: str | Path | None = None,
    tokenizer: Tokenizer | None = None,
    base_model: str | None = None,
    max_seq_len: int | None = None,
    eval_examples: Sequence[ChatExample] | None = None,
    near_duplicate_threshold: float = 0.85,
    min_examples: int = 8,
) -> ValidationReport:
    """Run every check and return a graded, aggregated report.

    Nothing here raises on bad data: the point is to report *all* problems in
    one pass. Use :meth:`ValidationReport.raise_for_severity` to gate a
    pipeline on the result.
    """
    report = ValidationReport(path=str(path) if path else None, num_examples=len(examples))
    findings: list[Finding] = []

    def add(code: str, severity: Severity, message: str, rows: Sequence[int] = (), count: int | None = None) -> None:
        findings.append(
            Finding(
                code=code,
                severity=severity,
                message=message,
                count=count if count is not None else max(len(rows), 1),
                examples=tuple(rows[:_MAX_EXAMPLES_PER_FINDING]),
            )
        )

    if not examples:
        add("empty_dataset", Severity.error, "Dataset contains no examples.")
        report.findings = findings
        return report

    if len(examples) < min_examples:
        add(
            "tiny_dataset",
            Severity.warning,
            f"Only {len(examples)} examples (< {min_examples}); LoRA runs on corpora this small "
            "usually memorise rather than generalise.",
        )

    if tokenizer is None:
        tokenizer = build_tokenizer(base_model or "byte", prefer_hf=False)

    context_length = None
    if base_model and is_supported(base_model):
        context_length = get_model(base_model).context_length

    # --- per-example structural / unicode / tokenisation checks ---------
    empty_target: list[int] = []
    zero_supervised: list[int] = []
    orphan_answer: list[int] = []
    untokenizable: list[int] = []
    overflow: list[int] = []
    truncated: list[int] = []
    bad_system: list[int] = []
    consecutive_role: list[int] = []
    control_chars: list[int] = []
    zero_width: list[int] = []
    bidi: list[int] = []
    surrogates: list[int] = []
    replacement: list[int] = []
    nfc_mismatch: list[int] = []
    trailing_user: list[int] = []
    pii_rows: list[int] = []
    pii_kinds: Counter = Counter()

    seq_lens: list[int] = []
    sup_lens: list[int] = []

    effective_max = max_seq_len or context_length or 1_000_000

    for i, ex in enumerate(examples):
        texts = [m.content for m in ex.messages]

        if not _response_text(ex).strip():
            empty_target.append(i)

        for j, m in enumerate(ex.messages):
            if m.role is Role.system and j != 0:
                bad_system.append(i)
                break
        for a, b in zip(ex.messages, ex.messages[1:], strict=False):
            if a.role is b.role and a.role in (Role.assistant, Role.user):
                consecutive_role.append(i)
                break
        if ex.messages[-1].role is Role.user:
            trailing_user.append(i)

        blob = "\n".join(texts)
        if _CONTROL.search(blob):
            control_chars.append(i)
        if _ZERO_WIDTH.search(blob):
            zero_width.append(i)
        if _BIDI.search(blob):
            bidi.append(i)
        if any(0xD800 <= ord(c) <= 0xDFFF for c in blob):
            surrogates.append(i)
        if "�" in blob:
            replacement.append(i)
        if unicodedata.normalize("NFC", blob) != blob:
            nfc_mismatch.append(i)

        _, kinds = redact_pii(blob)
        if kinds:
            pii_rows.append(i)
            pii_kinds.update(kinds)

        # Tokenisation: full length vs. supervision surviving truncation. A
        # corpus that cannot be tokenised at all must still produce a report.
        try:
            full = build_supervised_datum(ex, tokenizer, max_seq_len=10_000_000)
        except (UnicodeEncodeError, UnicodeDecodeError, ValueError):
            untokenizable.append(i)
            continue
        seq_lens.append(len(full.input_tokens))
        sup_lens.append(full.num_supervised_tokens)
        if context_length and len(full.input_tokens) > context_length:
            overflow.append(i)
        if len(full.input_tokens) > effective_max:
            truncated.append(i)
            clipped = build_supervised_datum(ex, tokenizer, max_seq_len=effective_max)
            if clipped.num_supervised_tokens == 0:
                zero_supervised.append(i)
            elif all(w > 0 for w in clipped.weights):
                # Truncation keeps the tail, so the answer survives but its
                # question does not: the model learns to emit it unprompted.
                orphan_answer.append(i)
        elif full.num_supervised_tokens == 0:
            zero_supervised.append(i)

    if empty_target:
        add("empty_assistant_turn", Severity.error,
            f"{len(empty_target)} example(s) have a blank assistant response - zero learning signal.",
            empty_target)
    if zero_supervised:
        add("zero_supervised_tokens", Severity.error,
            f"{len(zero_supervised)} example(s) contribute no supervised tokens at "
            f"max_seq_len={effective_max}; the assistant turn is entirely truncated away.",
            zero_supervised)
    if orphan_answer:
        add("prompt_truncated_away", Severity.error,
            f"{len(orphan_answer)} example(s) lose their entire prompt to truncation at "
            f"max_seq_len={effective_max}; only the unconditioned answer would be trained on.",
            orphan_answer)
    if untokenizable:
        add("untokenizable_example", Severity.error,
            f"{len(untokenizable)} example(s) could not be tokenised (invalid encoding); "
            "they were excluded from the token statistics below.", untokenizable)
    if overflow:
        add("context_overflow", Severity.error,
            f"{len(overflow)} example(s) exceed the {context_length}-token context window of {base_model}.",
            overflow)
    if truncated and not overflow:
        add("truncation", Severity.warning,
            f"{len(truncated)} example(s) exceed max_seq_len={effective_max} and will be truncated.",
            truncated)
    if bad_system:
        add("system_turn_not_first", Severity.warning,
            f"{len(bad_system)} example(s) place a system turn after the conversation start.",
            bad_system)
    if consecutive_role:
        add("consecutive_same_role", Severity.warning,
            f"{len(consecutive_role)} example(s) contain two consecutive turns from the same role.",
            consecutive_role)
    if trailing_user:
        add("trailing_user_turn", Severity.info,
            f"{len(trailing_user)} example(s) end on a user turn; the trailing prompt is unsupervised.",
            trailing_user)
    if control_chars:
        add("control_characters", Severity.warning,
            f"{len(control_chars)} example(s) contain raw control characters.", control_chars)
    if zero_width:
        add("zero_width_characters", Severity.warning,
            f"{len(zero_width)} example(s) contain zero-width/invisible characters.", zero_width)
    if bidi:
        add("bidi_override", Severity.error,
            f"{len(bidi)} example(s) contain bidirectional override characters "
            "(trojan-source risk: rendered text differs from the tokens trained on).", bidi)
    if surrogates:
        add("lone_surrogate", Severity.error,
            f"{len(surrogates)} example(s) contain lone surrogate code points; these break UTF-8 encoding.",
            surrogates)
    if replacement:
        add("replacement_character", Severity.warning,
            f"{len(replacement)} example(s) contain U+FFFD - a sign of an earlier mis-decode.", replacement)
    if nfc_mismatch:
        add("unicode_not_nfc", Severity.info,
            f"{len(nfc_mismatch)} example(s) are not NFC-normalised; identical-looking strings will "
            "tokenise differently.", nfc_mismatch)
    if pii_rows:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(pii_kinds.items()))
        add("pii_detected", Severity.error,
            f"{len(pii_rows)} example(s) contain probable PII/secrets ({kinds}). "
            "Redact before training - fine-tuned weights memorise rare strings.", pii_rows)

    # --- duplicates, near-duplicates, label conflicts -------------------
    index = NearDuplicateIndex(threshold=near_duplicate_threshold)
    exact_prompts: dict[str, int] = {}
    exact_pairs: dict[str, int] = {}
    duplicates: list[int] = []
    near_dupes: list[int] = []
    conflicts: list[int] = []

    for i, ex in enumerate(examples):
        prompt = _prompt_text(ex)
        response = _response_text(ex)
        pair_key = f"{_normalize(prompt)}\x00{_normalize(response)}"
        prompt_key = _normalize(prompt)

        if pair_key in exact_pairs:
            duplicates.append(i)
        else:
            exact_pairs[pair_key] = i
            hits = index.query(prompt + "\n" + response)
            if hits:
                near_dupes.append(i)

        if prompt_key and prompt_key in exact_prompts:
            if pair_key not in exact_pairs or exact_pairs[pair_key] == i:
                conflicts.append(i)
        else:
            exact_prompts[prompt_key] = i
        index.add(i, prompt + "\n" + response)

    if duplicates:
        pct = 100 * len(duplicates) / len(examples)
        add("exact_duplicates", Severity.warning if pct < 20 else Severity.error,
            f"{len(duplicates)} duplicate example(s) ({pct:.1f}% of the corpus); duplicates act as "
            "an undeclared upweighting.", duplicates)
    if near_dupes:
        add("near_duplicates", Severity.warning,
            f"{len(near_dupes)} example(s) are >={near_duplicate_threshold:.0%} similar to an earlier example.",
            near_dupes)
    if conflicts:
        add("label_conflict", Severity.warning,
            f"{len(conflicts)} prompt(s) appear with conflicting assistant responses; the model is "
            "being asked to fit contradictory targets.", conflicts)

    # --- response collapse / imbalance ----------------------------------
    response_counts = Counter(_normalize(_response_text(ex)) for ex in examples)
    top_resp, top_n = response_counts.most_common(1)[0]
    if len(examples) >= 4 and top_n / len(examples) >= 0.5:
        add("response_collapse", Severity.warning,
            f"A single response accounts for {100 * top_n / len(examples):.0f}% of the corpus "
            f"({top_n}/{len(examples)}); expect degenerate outputs.", count=top_n)

    # --- train/eval contamination ---------------------------------------
    if eval_examples:
        contaminated: list[int] = []
        for i, ex in enumerate(eval_examples):
            if index.query(_prompt_text(ex) + "\n" + _response_text(ex)):
                contaminated.append(i)
        if contaminated:
            pct = 100 * len(contaminated) / len(eval_examples)
            add("eval_contamination", Severity.error,
                f"{len(contaminated)}/{len(eval_examples)} eval example(s) ({pct:.1f}%) also appear in "
                "train; held-out metrics will be optimistic.", contaminated)

    # --- length distribution --------------------------------------------
    ordered = sorted(seq_lens)
    p50 = ordered[len(ordered) // 2] if ordered else 0
    p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))] if ordered else 0
    if p50 > 0 and p99 / p50 >= 20:
        add("length_outliers", Severity.info,
            f"Sequence lengths are heavily skewed (p50={p50}, p99={p99}); padding waste and "
            "OOM risk concentrate in the tail.")

    report.stats = {
        "total_tokens": sum(seq_lens),
        "supervised_tokens": sum(sup_lens),
        "supervised_ratio": round(sum(sup_lens) / max(sum(seq_lens), 1), 4),
        "seq_len_p50": p50,
        "seq_len_p99": p99,
        "seq_len_max": max(seq_lens) if seq_lens else 0,
        "unique_prompts": len(exact_prompts),
        "unique_responses": len(response_counts),
        "duplicate_rate": round(len(duplicates) / len(examples), 4),
        "pii_kinds": dict(pii_kinds),
        "most_common_response_share": round(top_n / len(examples), 4),
    }
    report.findings = sorted(findings, key=lambda f: (-_ORDER[f.severity], f.code))
    return report
