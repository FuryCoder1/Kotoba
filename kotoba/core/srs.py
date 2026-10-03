"""Spaced-repetition scheduler (SM-2) with Leitner boxes and leaky buckets.

Why three mechanisms in one engine?
-----------------------------------
* **SM-2 intervals** give per-card adaptive scheduling — the classic Anki-like
  behaviour users expect from an SRS.
* **Leitner boxes** provide a coarse, *explainable* mastery ladder ("box 4/6")
  that drives the Yokai companion's growth stages.
* **Forgetting curve / leaky bucket** models the probability that a card is
  still retrievable right now (`retention`).  The AI Sensei uses it to decide
  *when* to re-test a weakness, and the story mode uses it to gate level-ups.

The implementation is pure-python and deterministic given a clock, so it can be
unit-tested without mocks and shipped to a Flutter/WatermelonDB client as-is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .models import Rating, utcnow

# Stability of a memory trace after the first successful recall (days).
BASE_STABILITY_DAYS = 1.0
# How much easier each subsequent successful recall becomes (SM-2 ease floor).
MIN_EASE, MAX_EASE = 1.3, 3.0
DEFAULT_EASE = 2.5
BOXES = 6


@dataclass
class Card:
    """One reviewable item (a vocabulary word, kanji, or grammar point)."""
    id: str
    front: str
    back: str
    tags: list[str] = field(default_factory=list)
    # --- scheduling state ---
    reps: int = 0                  # consecutive successful recalls
    lapses: int = 0                # total failures
    interval_days: float = 0.0     # current SM-2 interval
    ease: float = DEFAULT_EASE
    box: int = 1                   # Leitner box 1..BOXES
    stability_days: float = BASE_STABILITY_DAYS
    due: datetime | None = None
    last_reviewed: datetime | None = None
    created: datetime = field(default_factory=utcnow)

    def is_due(self, now: datetime | None = None) -> bool:
        now = now or utcnow()
        return self.due is None or self.due <= now

    @property
    def new(self) -> bool:
        return self.reps == 0 and self.last_reviewed is None

    def retention(self, now: datetime | None = None) -> float:
        """Exponential forgetting-curve estimate P(recall now)."""
        now = now or utcnow()
        anchor = self.last_reviewed or self.created
        elapsed = max(0.0, (now - anchor).total_seconds() / 86400.0)
        if elapsed == 0:
            return 1.0
        return math.exp(-elapsed / max(0.25, self.stability_days))

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("due", "last_reviewed", "created"):
            if isinstance(d[k], datetime):
                d[k] = d[k].isoformat()
        d["retention"] = round(self.retention(), 4)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Card":
        dt = lambda v: datetime.fromisoformat(v) if v else None
        return cls(
            id=d["id"], front=d["front"], back=d["back"], tags=list(d.get("tags", [])),
            reps=int(d.get("reps", 0)), lapses=int(d.get("lapses", 0)),
            interval_days=float(d.get("interval_days", 0.0)),
            ease=float(d.get("ease", DEFAULT_EASE)),
            box=int(d.get("box", 1)),
            stability_days=float(d.get("stability_days", BASE_STABILITY_DAYS)),
            due=dt(d.get("due")), last_reviewed=dt(d.get("last_reviewed")),
            created=dt(d.get("created")) or utcnow(),
        )


@dataclass
class ReviewResult:
    card_id: str
    rating: Rating
    previous_interval: float
    interval_days: float
    ease: float
    box: int
    reps: int
    lapses: int
    due: datetime
    retention_after: float


class Scheduler:
    """Owns a deck of cards and applies the SM-2 + Leitner update rules."""

    def __init__(self, cards: Iterable[Card] | None = None):
        self.cards: dict[str, Card] = {}
        for c in cards or []:
            self.add(c)

    # ---- deck management ------------------------------------------------
    def add(self, card: Card) -> Card:
        if card.id in self.cards:
            raise ValueError(f"duplicate card id {card.id}")
        self.cards[card.id] = card
        return card

    def remove(self, card_id: str) -> None:
        self.cards.pop(card_id, None)

    def __len__(self) -> int:
        return len(self.cards)

    def due_cards(self, now: datetime | None = None, limit: int | None = None) -> list[Card]:
        now = now or utcnow()
        due = [c for c in self.cards.values() if c.is_due(now)]
        # teach newest-first when nothing is overdue; otherwise lowest retention first
        due.sort(key=lambda c: (c.retention(now), c.due or c.created))
        return due[:limit] if limit else due

    def stats(self, now: datetime | None = None) -> dict:
        now = now or utcnow()
        by_box: dict[int, int] = {b: 0 for b in range(1, BOXES + 1)}
        for c in self.cards.values():
            by_box[c.box] = by_box.get(c.box, 0) + 1
        mature = [c for c in self.cards.values() if c.interval_days >= 21]
        return {
            "total": len(self.cards),
            "newCards": sum(1 for c in self.cards.values() if c.new),
            "dueToday": len(self.due_cards(now)),
            "mature": len(mature),
            "young": len(self.cards) - len(mature),
            "byBox": by_box,
            "meanRetention": round(
                sum(c.retention(now) for c in self.cards.values()) / len(self.cards), 4
            ) if self.cards else 1.0,
        }

    # ---- the core algorithm --------------------------------------------
    def review(self, card_id: str, rating: Rating, now: datetime | None = None) -> ReviewResult:
        card = self.cards[card_id]
        now = now or utcnow()
        q = rating.sm2
        prev_interval = card.interval_days

        if q < 3:                                  # failure
            card.lapses += 1
            card.reps = 0
            card.box = 1
            card.interval_days = 0.0               # relearn same day
            card.ease = max(MIN_EASE, card.ease - 0.20)
            card.stability_days = max(0.25, card.stability_days * 0.4)
            due = now + timedelta(minutes=10)      # immediate re-test
        else:                                      # success
            card.reps += 1
            if rating is Rating.HARD:              # SM-2 quality 2
                card.ease = max(MIN_EASE, card.ease - 0.15)
                card.interval_days = max(1.0, prev_interval * 1.2)
            elif rating is Rating.GOOD:            # quality 3
                card.interval_days = (1.0 if card.reps == 1
                                      else 6.0 if card.reps == 2
                                      else prev_interval * card.ease)
            else:                                  # EASY -> quality 5
                card.ease = min(MAX_EASE, card.ease + 0.15)
                card.interval_days = (2.0 if card.reps == 1
                                      else 8.0 if card.reps == 2
                                      else prev_interval * card.ease * 1.3)
            card.interval_days = round(card.interval_days, 2)
            card.box = min(BOXES, card.box + 1)
            # FSRS-flavoured stability growth proportional to difficulty handled
            card.stability_days = card.stability_days * (1.0 + 0.55 * card.ease)
            due = now + timedelta(days=card.interval_days)

        card.last_reviewed = now
        card.due = due
        return ReviewResult(
            card_id=card.id, rating=rating, previous_interval=prev_interval,
            interval_days=card.interval_days, ease=round(card.ease, 3), box=card.box,
            reps=card.reps, lapses=card.lapses, due=due,
            retention_after=round(card.retention(now), 4),
        )

    # ---- persistence ----------------------------------------------------
    def export(self) -> list[dict]:
        return [c.to_dict() for c in self.cards.values()]

    @classmethod
    def load(cls, payload: list[dict]) -> "Scheduler":
        return cls(Card.from_dict(d) for d in payload)
