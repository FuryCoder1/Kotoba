"""Gamification — RPG journey, streaks, yokai companion, coins.

Journey (RPG Story Mode)
------------------------
A linear map of Japan. Each region has a syllabus (JLPT level + grammar
focus). A region is cleared when the learner's SRS stats show enough mature
cards AND roleplay/grammar milestones are met; reaching Osaka unlocks a
Kansai-ben mini-module.

Companion (Virtual Yokai)
-------------------------
Evolution is driven by *quality of study*, not just clicks: stage =
f(streak, SRS retention, reviews completed). Neglect decays the bond so the
pet visibly wilts if SRS reviews are skipped (retention < 0.7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

REGIONS = [
    {"id": "hokkaido", "name": "Hokkaido Village", "jp": "北海道の村",
     "level": "N5", "focus": ["です・ます", "は vs が", "counters"],
     "requireMatureCards": 10, "requireRoleplays": 0},
    {"id": "sendai", "name": "Sendai Coast", "jp": "仙台",
     "level": "N5", "focus": ["て form sequencing", "location verbs あります/います"],
     "requireMatureCards": 25, "requireRoleplays": 1},
    {"id": "tokyo", "name": "Tokyo Metro", "jp": "東京",
     "level": "N4", "focus": ["に vs で", "potential form", "te-iru states"],
     "requireMatureCards": 45, "requireRoleplays": 2},
    {"id": "kyoto", "name": "Kyoto Old Town", "jp": "京都",
     "level": "N3", "focus": ["keigo triad", "passive/causative"],
     "requireMatureCards": 70, "requireRoleplays": 3},
    {"id": "osaka", "name": "Osaka Dotonbori", "jp": "大阪",
     "level": "N3", "focus": ["volitional", "sentence-final よ/ね/ぜ"],
     "requireMatureCards": 95, "requireRoleplays": 4},
]

KANSAI_BEN = {
    "unlockedAt": "osaka",
    "notes": "When you reach Osaka a mini-module activates:",
    "lessons": [
        {"standard": "大丈夫です", "kansai": "ほな、またな！→ See you!", "note": "ほな = では"},
        {"standard": "そうです", "kansai": "そうや / やな", "note": "copula や"},
        {"standard": "ありがとう", "kansai": "おおきに", "note": "distinct thanks"},
        {"standard": "高い", "kansai": "つえー", "note": " vowel shift"},
        {"standard": "〜ています", "kansai": "〜とる", "note": "te-iru reduction"},
        {"standard": "なぜ", "kansai": "なんでやねん", "note": "the famous tsukkomi"},
    ],
}

YOKAI_STAGES = {
    "shiba_inu": ["Mochi Pup", "Travelling Shiba", "Shogun Shiba", "Celestial Dog (Inugami)"],
    "kitsune": ["Kit Pup", "Torch Fox", "Miko's Fox", "Celestial Kitsune"],
    "tanuki": ["Puddle Tanuki", "Leaf Trickster", "Kettle Tanuki", "Shukuku Deity"],
}


@dataclass
class JourneyState:
    current_region: int = 0                     # index into REGIONS
    cleared: list[str] = field(default_factory=list)
    roleplays_passed: int = 0                   # sessions with grade >= 70
    coins: int = 0

    def as_dict(self) -> dict:
        cur = REGIONS[self.current_region]
        return {
            "currentRegion": cur,
            "regionIndex": self.current_region,
            "cleared": self.cleared,
            "roleplaysPassed": self.roleplays_passed,
            "coins": self.coins,
            "kansaiBenUnlocked": "osaka" in self.cleared,
            "mapProgress": round(self.current_region / (len(REGIONS) - 1), 2),
        }


def update_journey(state: JourneyState, srs_stats: dict) -> tuple[JourneyState, list[str]]:
    """Advance regions while requirements hold. Returns (state, events)."""
    events: list[str] = []
    mature = srs_stats.get("mature", 0)
    while state.current_region < len(REGIONS):
        r = REGIONS[state.current_region]
        if mature >= r["requireMatureCards"] and state.roleplays_passed >= r["requireRoleplays"]:
            if r["id"] not in state.cleared:
                state.cleared.append(r["id"])
                state.coins += 50
                events.append(f"Cleared {r['name']} (+50 coins)")
            state.current_region += 1
            if state.current_region < len(REGIONS):
                events.append(f"Travelled to {REGIONS[state.current_region]['name']}")
            if r["id"] == "osaka":
                events.append("🎌 Kansai-ben mini-module unlocked!")
        else:
            break
    if state.current_region >= len(REGIONS):
        state.current_region = len(REGIONS) - 1
    return state, events


# ---------------------------------------------------------------------------
# Streaks
# ---------------------------------------------------------------------------

def register_study_day(history: list[str], today: str | None = None) -> dict:
    """history = ISO dates studied. Returns updated history + streak info."""
    today = today or date.today().isoformat()
    hist = sorted(set(history))
    if not hist or hist[-1] != today:
        hist.append(today)
    streak = 1
    d = date.fromisoformat(hist[-1])
    for prev in reversed(hist[:-1]):
        if date.fromisoformat(prev) == d - timedelta(days=1):
            streak += 1
            d = date.fromisoformat(prev)
        else:
            break
    broke = False
    if len(hist) >= 2:
        gap = (date.fromisoformat(hist[-1]) - date.fromisoformat(hist[-2])).days
        broke = gap > 1
    return {"history": hist, "streak": streak,
            "longestStreak": _longest(hist), "streakJustBrokeYesterday": broke}


def _longest(hist: list[str]) -> int:
    if not hist:
        return 0
    best = run = 1
    for i in range(1, len(hist)):
        if (date.fromisoformat(hist[i]) - date.fromisoformat(hist[i-1])).days == 1:
            run += 1; best = max(best, run)
        else:
            run = 1
    return best


# ---------------------------------------------------------------------------
# Companion
# ---------------------------------------------------------------------------

def companion_status(species: str, streak: int, reviews_total: int,
                     retention: float) -> dict:
    stages = YOKAI_STAGES.get(species)
    if stages is None:
        raise KeyError(f"unknown companion '{species}' "
                       f"(choose from {sorted(YOKAI_STAGES)})")
    score = (min(streak, 60) / 60) * 0.4 + \
            (min(reviews_total, 500) / 500) * 0.35 + \
            max(0.0, min(retention, 1.0)) * 0.25
    idx = 0 if score < 0.25 else 1 if score < 0.5 else 2 if score < 0.75 else 3
    if retention < 0.7:
        mood = "wilting 😟 — your SRS backlog is hurting; review to heal the bond"
    elif retention > 0.9 and streak >= 7:
        mood = "thriving ✨ — feeding on perfect reviews"
    else:
        mood = "content 🙂"
    next_at = [0.25, 0.5, 0.75, 1.01][idx]
    return {
        "species": species,
        "name": stages[idx],
        "stage": idx + 1,
        "maxStage": len(stages),
        "growthScore": round(score, 3),
        "mood": mood,
        "nextEvolutionAt": None if idx == len(stages) - 1 else next_at,
        "evolveHint": ("Keep the streak and finish every review today."
                       if idx < 3 else "Fully evolved — legendary companion."),
    }
