"""Kanji engine: stroke-order validation, radical mnemonics, contextual readings.

Three learner-facing capabilities
---------------------------------
1. **Stroke-order validation (the "AR canvas" backend).**  A stroke is a
   polyline of sampled points.  We validate each drawn stroke against the
   reference stroke by (a) start/end proximity, (b) direction cosine, and
   (c) path deviation.  Drawing *backwards* yields `direction_reversed`, which
   is what triggers the haptic buzz in the client.

2. **Radical mnemonics.**  Kanji are decomposed into radicals; the story is
   composed from the radical hints so learners never rote-memorise shapes.

3. **Contextual readings only.**  `readings_for_word()` returns the on'yomi /
   kun'yomi actually used inside a given vocabulary word by matching the kanji's
   known reading stems against the word's furigana — the app never asks you to
   recite a reading list detached from vocabulary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..data import load_kanji, load_radicals

Point = tuple[float, float]      # normalised 0..1 canvas coordinates


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit(v: Point) -> Point:
    m = math.hypot(*v) or 1e-9
    return (v[0] / m, v[1] / m)


def direction(stroke: list[Point]) -> Point:
    """Net direction as the vector from first to last point."""
    if len(stroke) < 2:
        return (1.0, 0.0)
    return _unit((stroke[-1][0] - stroke[0][0], stroke[-1][1] - stroke[0][1]))


def arc_direction(stroke: list[Point]) -> Point:
    """Direction along the *sweep* of the stroke (robust for curves)."""
    if len(stroke) < 3:
        return direction(stroke)
    k = max(1, len(stroke) // 4)
    return _unit((stroke[-1][0] - stroke[-1 - k][0], stroke[-1][1] - stroke[-1 - k][1]))


def mean_deviation(a: list[Point], b: list[Point]) -> float:
    """Average distance between resampled paths (shape similarity)."""
    ra, rb = resample(a, 24), resample(b, 24)
    return sum(dist(pa, pb) for pa, pb in zip(ra, rb)) / len(ra)


def resample(points: list[Point], n: int) -> list[Point]:
    if len(points) == 1:
        return [points[0]] * n
    total = sum(dist(points[i], points[i + 1]) for i in range(len(points) - 1))
    if total == 0:
        return [points[0]] * n
    step, out, acc, cur = total / (n - 1), [points[0]], 0.0, points[0]
    for p in points[1:]:
        seg = dist(cur, p)
        while acc + seg >= step and seg > 0:
            t = (step - acc) / seg
            cur = (cur[0] + (p[0] - cur[0]) * t, cur[1] + (p[1] - cur[1]) * t)
            out.append(cur)
            seg = dist(cur, p)
            acc = 0.0
        acc += seg
        cur = p
    while len(out) < n:
        out.append(points[-1])
    return out[:n]


# ---------------------------------------------------------------------------
# Stroke validation
# ---------------------------------------------------------------------------

@dataclass
class StrokeVerdict:
    index: int
    ok: bool
    score: float                     # 0..1
    reasons: list[str] = field(default_factory=list)

    @property
    def haptic(self) -> str | None:
        """What the client should do: 'reverse-buzz' | 'soft-tap' | None."""
        if self.ok:
            return None
        if any(r == "direction_reversed" for r in self.reasons):
            return "reverse-buzz"
        return "soft-tap"


def validate_stroke(index: int, drawn: list[Point], reference: list[Point],
                    *, start_tol: float = 0.18, end_tol: float = 0.22,
                    reverse_cos: float = -0.35) -> StrokeVerdict:
    """Compare one user-drawn stroke with its reference polyline."""
    if len(drawn) < 2:
        return StrokeVerdict(index, False, 0.0, ["too_short"])
    reasons: list[str] = []

    d_start = dist(drawn[0], reference[0])
    d_end = dist(drawn[-1], reference[-1])
    cos = sum(a * b for a, b in zip(direction(drawn), direction(reference)))

    if cos < reverse_cos:
        reasons.append("direction_reversed")
    if d_start > start_tol:
        reasons.append("wrong_start")
    if d_end > end_tol:
        reasons.append("wrong_end")
    dev = mean_deviation(drawn, reference)
    if dev > 0.16:
        reasons.append("path_drift")

    # composite score: geometry closeness weighted by correctness of order
    score = max(0.0, 1.0 - (0.35 * min(1.0, d_start / start_tol)
                            + 0.25 * min(1.0, d_end / end_tol)
                            + 0.40 * min(1.0, dev / 0.16)))
    if "direction_reversed" in reasons:
        score *= 0.4
    ok = not reasons
    return StrokeVerdict(index, ok, round(score, 3), reasons)


def validate_writing(drawn: list[list[Point]], reference: list[list[Point]]) -> dict:
    """Full-character check: strokes must be drawn in order and count match."""
    verdicts = []
    if len(drawn) != len(reference):
        verdicts.append(StrokeVerdict(-1, False, 0.0,
                                      [f"stroke_count_expected_{len(reference)}"
                                       f"_got_{len(drawn)}"]))
    for i, ref in enumerate(reference):
        if i >= len(drawn):
            verdicts.append(StrokeVerdict(i, False, 0.0, ["missing_stroke"]))
            continue
        verdicts.append(validate_stroke(i, drawn[i], ref))
    scores = [v.score for v in verdicts]
    return {
        "verdicts": [{"index": v.index, "ok": v.ok, "score": v.score,
                      "reasons": v.reasons, "haptic": v.haptic} for v in verdicts],
        "passed": all(v.ok for v in verdicts),
        "averageScore": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "strokeCount": len(reference),
    }


# ---------------------------------------------------------------------------
# Radicals & mnemonic composition
# ---------------------------------------------------------------------------

class KanjiDatabase:
    def __init__(self, kanji: list[dict] | None = None, radicals: dict | None = None):
        self.kanji = kanji if kanji is not None else load_kanji()
        self.radicals = radicals if radicals is not None else load_radicals()
        self._by_char = {k["character"]: k for k in self.kanji}

    def get(self, char: str) -> dict | None:
        return self._by_char.get(char)

    def search(self, level: str | None = None, grade: int | None = None,
                limit: int = 50) -> list[dict]:
        out = []
        for k in self.kanji:
            if level and k.get("jlpt") != level:
                continue
            if grade is not None and k.get("grade") != grade:
                continue
            out.append(k)
            if len(out) >= limit:
                break
        return out

    # --- mnemonics ---
    def radical_breakdown(self, char: str) -> list[dict]:
        k = self._by_char.get(char)
        if not k:
            return []
        rows = []
        for rid in k.get("radicals", []):
            rad = self.radicals.get(rid, {})
            rows.append({"id": rid, "name": rad.get("name", "?"),
                         "meaning": rad.get("meaning", "?"), "hint": rad.get("hint", "")})
        return rows

    def mnemonic_story(self, char: str) -> dict:
        """Compose the stored story plus the radical-derived scaffold."""
        k = self._by_char.get(char)
        if not k:
            raise KeyError(f"kanji {char!r} not in database")
        parts = self.radical_breakdown(char)
        scaffold = " + ".join(f"{p['name']} ({p['meaning']})" for p in parts) or "—"
        return {
            "character": char,
            "story": k.get("mnemonic", ""),
            "radicals": parts,
            "scaffold": scaffold,
            "components": k.get("components", []),
            "strokeCount": k["strokeCount"],
        }

    # --- contextual readings (never taught in isolation) ---
    def readings_in_context(self, char: str, word: str, furigana: str) -> dict:
        """Which reading of `char` is used inside `word`?

        Matches the furigana against the kanji's on'yomi/kun'yomi stems.
        """
        k = self._by_char.get(char)
        if not k:
            raise KeyError(f"kanji {char!r} not in database")
        from .kana import to_hiragana
        fu = to_hiragana(furigana)
        onyomi_hits, kunyomi_hits = [], []
        for r in k.get("onyomi", []):
            base = to_hiragana(r.lower())
            if base and base[0] in fu.replace("っ", "").replace("ー", ""):
                onyomi_hits.append(r)
            if base and base in fu:
                onyomi_hits.append(r)
        for r in k.get("kunyomi", []):
            stem = r.split(".")[0].split("-").strip(".-")
            if stem and stem in fu:
                kunyomi_hits.append(r)
        kind = ("onyomi" if onyomi_hits and not kunyomi_hits else
                "kunyomi" if kunyomi_hits and not onyomi_hits else
                "mixed" if onyomi_hits and kunyomi_hits else "unknown")
        return {
            "character": char,
            "word": word,
            "furigana": furigana,
            "readingKind": kind,
            "candidatesOn": sorted(set(onyomi_hits)),
            "candidatesKun": sorted(set(kunyomi_hits)),
            "explanation": (
                "Compound/loan-style reading (音読み) — typical inside multi-kanji words."
                if kind == "onyomi" else
                "Native reading (訓読み) — typical when the kanji stands alone or carries okurigana."
                if kind == "kunyomi" else
                "Ambiguous in this context; both layers appear in the word."
                if kind == "mixed" else
                "Irregular/jukujiku reading — memorise through the whole word, not the parts."
            ),
        }
