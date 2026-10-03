"""Pitch-accent engine.

Model of Tokyo-standard (共通語) pitch accent
--------------------------------------------
Japanese is a *high/low* pitch language, not a stress language.  Each word has
an "accent nucleus" position `n` (0 <= n <= moraCount):

  n = 0            平板型 heiban      : H rises after mora 1 and never falls
                                        (even if followed by a particle)
  n = 1            頭高型 atamadaka   : mora 1 is H, everything after is L
  n = moraCount    尾高型 odaka       : falls on the last mora; the *particle*
                                        after it drops to L
  1 < n < moraCount 中高型 nakadaka  : falls inside the word

The canonical contour therefore includes a *following particle* slot so that
odaka words are distinguishable from heiban ones:

  hashi(N1) ga -> L H L     橋が (bridge)
  hashi(N2) ga -> H L L     箸が (chopsticks)

This module provides
  * pattern generation / classification
  * minimal-pair lookup (the killer feature most apps lack)
  * downsampled synthetic F0 curves for visualisation
  * scoring of a *measured* F0 curve against the target contour, producing
    human-readable coaching feedback ("your pitch dropped too early on 'shi'").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..data import load_pitch_entries
from .kana import segment_morae

# Reference F0 levels (Hz) for the synthetic reference contour.
LOW_HZ = 150.0
HIGH_HZ = 235.0


def classify_accent_type(accent_location: int, mora_count: int) -> str:
    """Return one of heiban / atamadaka / nakadaka / odaka."""
    if mora_count <= 0:
        raise ValueError("mora_count must be positive")
    if not (0 <= accent_location <= mora_count):
        raise ValueError(
            f"accent location {accent_location} out of range for {mora_count} morae"
        )
    if accent_location == 0:
        return "heiban"
    if accent_location == 1:
        # For a 1-mora word the fall is on the only mora: still head-high.
        return "atamadaka"
    if accent_location == mora_count:
        return "odaka"
    return "nakadaka"


def pitch_pattern(accent_location: int, mora_count: int, with_particle: bool = True) -> str:
    """High/Low string over the word's morae (+ optional particle mora).

    Rule: mora i (1-indexed) is H iff (i > 1) and (i <= accent or accent == 0),
    except that an accented mora itself stays H and the *next* mora falls.
    """
    seq: list[str] = []
    for i in range(1, mora_count + 1):
        if i == 1:
            seq.append("H" if accent_location == 1 else "L")
        elif accent_location == 0:
            seq.append("H")
        elif i <= accent_location:
            seq.append("H")
        else:
            seq.append("L")
    if with_particle:
        # Particle is H only when the word is heiban or unaccented-before-drop.
        seq.append("H" if accent_location == 0 else "L")
    return "".join(seq)


def f0_curve(accent_location: int, mora_count: int, samples_per_mora: int = 8,
              with_particle: bool = True) -> list[float]:
    """Synthetic native-speaker F0 contour used as the visualiser baseline."""
    pattern = pitch_pattern(accent_location, mora_count, with_particle)
    out: list[float] = []
    for idx, sym in enumerate(pattern):
        target = HIGH_HZ if sym == "H" else LOW_HZ
        prev = out[-1] if out else LOW_HZ
        for s in range(samples_per_mora):
            # short glide between morae makes the chart look like real F0
            t = s / samples_per_mora
            v = prev + (target - prev) * min(1.0, t * 2.2)
            out.append(round(v, 2))
        out[-1] = target
    return out


@dataclass
class PitchTarget:
    kana: str
    morae: list[str]
    accent_location: int
    mora_count: int
    accent_type: str
    pattern: str                 # incl. particle slot
    meaning: str = ""

    @property
    def romaji(self) -> str:
        from .kana import mora_to_romaji
        return "-".join(mora_to_romaji(m) or "Q" for m in self.morae)

    def contour(self, samples_per_mora: int = 8) -> list[float]:
        return f0_curve(self.accent_location, self.mora_count, samples_per_mora)

    def as_dict(self) -> dict:
        return {
            "kana": self.kana,
            "morae": self.morae,
            "romaji": self.romaji,
            "accentLocation": self.accent_location,
            "moraCount": self.mora_count,
            "accentType": self.accent_type,
            "pattern": self.pattern,
            "meaning": self.meaning,
            "referenceF0": self.contour(),
        }


class PitchDictionary:
    """Lookup + minimal-pair index over the bundled pitch dataset."""

    def __init__(self, entries: list[dict] | None = None):
        self._entries = entries if entries is not None else load_pitch_entries()
        self._index: dict[str, list[dict]] = {}
        for e in self._entries:
            self._index.setdefault(e["kana"], []).append(e)

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, kana: str) -> list[PitchTarget]:
        kana = "".join(segment_morae(kana))
        hits = self._index.get(kana, [])
        targets = []
        for e in hits:
            n = e["moraCount"]
            loc = e["accentLocation"]
            targets.append(PitchTarget(
                kana=e["kana"],
                morae=segment_morae(e["kana"]),
                accent_location=loc,
                mora_count=n,
                accent_type=classify_accent_type(loc, n),
                pattern=pitch_pattern(loc, n),
                meaning=e.get("meaning", ""),
            ))
        return targets

    def minimal_pairs(self, kana: str) -> list[tuple[PitchTarget, PitchTarget]]:
        """All pairs of readings sharing the same kana but differing in accent."""
        ts = self.lookup(kana)
        pairs = []
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                if ts[i].accent_location != ts[j].accent_location:
                    pairs.append((ts[i], ts[j]))
        return pairs

    def all_minimal_pair_kana(self) -> list[str]:
        seen = []
        for kana, group in self._index.items():
            if len({g["accentLocation"] for g in group}) > 1:
                seen.append(kana)
        return sorted(seen)


# ---------------------------------------------------------------------------
# Scoring a learner's measured F0 contour against the target
# ---------------------------------------------------------------------------

@dataclass
class MoraVerdict:
    mora: str
    romaji: str
    expected: str
    observed: str
    ok: bool


@dataclass
class PitchScore:
    accuracy: float                 # 0..1 fraction of correctly matched morae
    drop_position_expected: int
    drop_position_observed: int | None
    verdicts: list[MoraVerdict]
    feedback: list[str]
    grade: str                      # "native-like" | "good" | "fair" | "needs work"


def _observed_levels(f0: list[float], mora_count: int, threshold_hz: float = 190.0) -> list[str]:
    """Average each mora window of the measured contour and bin into H/L."""
    if not f0:
        return ["L"] * mora_count
    win = max(1, len(f0) // mora_count)
    levels = []
    for i in range(mora_count):
        chunk = f0[i * win:(i + 1) * win] or f0[-1:]
        mean = sum(chunk) / len(chunk)
        levels.append("H" if mean >= threshold_hz else "L")
    return levels


def _first_drop(levels: list[str]) -> int | None:
    for i in range(1, len(levels)):
        if levels[i] == "L" and levels[i - 1] == "H":
            return i          # 1-indexed mora where the fall happens
    return None


def score_performance(target: PitchTarget, measured_f0: list[float]) -> PitchScore:
    """Compare a recorded F0 track against the native reference contour."""
    from .kana import mora_to_romaji

    obs = _observed_levels(measured_f0, target.mora_count)
    exp = target.pattern[:target.mora_count]           # ignore particle slot here
    verdicts = [
        MoraVerdict(m, mora_to_romaji(m) or "?", e, o, e == o)
        for m, e, o in zip(target.morae, exp, obs)
    ]
    correct = sum(1 for v in verdicts if v.ok)
    accuracy = correct / len(verdicts) if verdicts else 0.0

    exp_drop = target.accent_location if target.accent_location else None
    obs_drop = _first_drop(obs)

    feedback: list[str] = []
    if accuracy >= 0.999:
        grade = "native-like"
        feedback.append("ピッチの高低が完全に一致しています。Excellent — the contour matches a native speaker.")
    else:
        grade = "good" if accuracy >= 0.75 else ("fair" if accuracy >= 0.5 else "needs work")
        if obs_drop is None and exp_drop is not None:
            feedback.append(
                f"You never dropped your pitch. 「{target.kana}」 is {target.accent_type}: "
                f"the fall should come right after mora {exp_drop} "
                f"({mora_to_romaji(target.morae[exp_drop - 1])})."
            )
        elif obs_drop is not None and exp_drop is None:
            feedback.append(
                f"Your pitch fell after mora {obs_drop}, but 「{target.kana}」 is 平板型 (heiban) — "
                "keep it high all the way through, even onto the particle."
            )
        elif obs_drop is not None and exp_drop is not None and obs_drop != exp_drop:
            side = "too early" if obs_drop < exp_drop else "too late"
            where = mora_to_romaji(target.morae[max(0, obs_drop - 1)]) or "?"
            feedback.append(
                f"Your pitch dropped {side} — it fell on '{where}' (mora {obs_drop}) "
                f"but should fall after mora {exp_drop}."
            )
        bad = [v for v in verdicts if not v.ok]
        for v in bad[:2]:
            if abs((exp.index(v.expected) if v.expected in exp else 0)) >= 0:
                feedback.append(
                    f"Mora {verdicts.index(v) + 1} 「{v.mora}」 ({v.romaji}): expected {v.expected}, you sang {v.observed}."
                )
    return PitchScore(accuracy, exp_drop or 0, obs_drop, verdicts, feedback, grade)
