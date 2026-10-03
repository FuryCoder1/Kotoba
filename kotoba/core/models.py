"""Shared domain types and small utilities used across Kotoba engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Rating(int, Enum):
    """SM-2 style answer quality. 0-4; >=3 counts as a correct recall."""
    AGAIN = 0
    HARD = 1
    GOOD = 2      # mapped to SM-2 quality 3
    EASY = 3      # mapped to SM-2 quality 4

    @property
    def sm2(self) -> int:
        return {Rating.AGAIN: 1, Rating.HARD: 2, Rating.GOOD: 3, Rating.EASY: 5}[self]


@dataclass
class Token:
    """A morpheme-level token produced by the grammar deconstructor."""
    surface: str
    reading: str
    lemma: str
    pos: str
    particle: str | None = None
    depends_on: int = -1          # index of head token, -1 = root
    role: str | None = None       # filled in by the deconstructor (e.g. "topic")

    def as_dict(self) -> dict:
        return {
            "surface": self.surface,
            "reading": self.reading,
            "lemma": self.lemma,
            "pos": self.pos,
            "particle": self.particle,
            "dependsOn": self.depends_on,
            "role": self.role,
        }


@dataclass
class Sentence:
    id: str
    text: str
    romaji: str
    translation: str
    level: str
    furigana: list[dict] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)


@dataclass
class WordEntry:
    id: str
    word: str
    reading: str
    romaji: str
    glosses: list[str]
    pos: list[str]
    pitch_accent: int
    pitch_pattern: str
    level: str
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "WordEntry":
        return cls(
            id=d["id"],
            word=d["word"],
            reading=d["reading"],
            romaji=d.get("romaji", ""),
            glosses=list(d.get("glosses", [])),
            pos=list(d.get("pos", [])),
            pitch_accent=int(d.get("pitchAccent", 0)),
            pitch_pattern=d.get("pitchPattern", ""),
            level=d.get("level", "N5"),
            tags=list(d.get("tags", [])),
        )


def mora_count(kana: str) -> int:
    """Count morae in a hiragana/katakana string (tsu/long vowels count)."""
    from .kana import count_morae
    return count_morae(kana)
