"""Roleplay engine — scenario state machine driven by data/scenarios.json.

A session walks the learner through an NPC conversation:
  opening line -> learner turn -> goal check + Keigo rubric scoring ->
  scripted NPC reply (regex `modelResponses`) -> next goal.

Grading per turn (0..100):
  * 55%  – goal coverage (required phrases present)
  * 30%  – Keigo rubric (required patterns earn, penalties deduct)
  * 15%  – register analysis from kotoba.core.keigo

The transcript, missed goals and keigo violations are returned so the AI
Sensei can log them into its mistake memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..data import load_scenarios
from .keigo import analyze


@dataclass
class TurnResult:
    npc_reply: str
    goal_progress: list[dict]           # all goals with done flag
    goals_completed_this_turn: list[str]
    keigo_score: float                  # 0..1 rubric satisfaction
    rubric_hits: list[str]
    rubric_penalties: list[str]            # rule + reason strings deducted
    register_verdict: str
    turn_grade: int                     # 0..100
    session_grade: int
    finished: bool

    def as_dict(self) -> dict:
        return {
            "npcReply": self.npc_reply,
            "goalProgress": self.goal_progress,
            "goalsCompletedThisTurn": self.goals_completed_this_turn,
            "keigoScore": round(self.keigo_score, 3),
            "rubricHits": self.rubric_hits,
            "rubricPenalties": self.rubric_penalties,
            "registerVerdict": self.register_verdict,
            "turnGrade": self.turn_grade,
            "sessionGrade": self.session_grade,
            "finished": self.finished,
        }


class RoleplaySession:
    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.goals_done: set[str] = set()
        self.transcript: list[dict] = []
        self.turn_scores: list[int] = []
        self._compiled_penalties = [
            (p["rule"], re.compile(p["pattern"]), p["points"], p["why"])
            for p in scenario.get("keigoRubric", {}).get("penalties", [])
        ]
        self._compiled_required = [
            (p["rule"], re.compile(p["pattern"]), p["points"], p["why"])
            for p in scenario.get("keigoRubric", {}).get("required", [])
        ]
        self.opening = scenario.get("openingLine", "こんにちは！")
        self.transcript.append({"speaker": "npc", "text": self.opening})

    # -- helpers ----------------------------------------------------------
    def _match_npc(self, user_text: str) -> str:
        for r in self.scenario.get("modelResponses", []):
            try:
                if re.search(r["match"], user_text):
                    return r["reply"]
            except re.error:
                continue
        fallbacks = [
            "なるほど。他にご入用ですか？",
            "かしこまりました。ほかに何かお手伝いしましょうか？",
        ]
        return fallbacks[len(self.transcript) % 2]

    def _score_keigo(self, text: str) -> tuple[float, list[str], list[str]]:
        hits, penalties = [], []
        earned = possible = 0.0
        for rule, rx, pts, why in self._compiled_required:
            possible += pts
            if rx.search(text):
                earned += pts
                hits.append(rule)
        pen = 0.0
        for rule, rx, pts, why in self._compiled_penalties:
            if rx.search(text):
                pen += abs(pts)
                penalties.append(f"{rule}: {why}")
        base = earned / possible if possible else 1.0
        score = max(0.0, min(1.0, base - pen / max(possible, 1)))
        return score, hits, penalties

    # -- public API --------------------------------------------------------
    def progress(self) -> list[dict]:
        return [
            {"id": g["id"], "text": g["text"],
             "requiredPhrases": g["requiredPhrases"],
             "done": g["id"] in self.goals_done}
            for g in self.scenario.get("goals", [])
        ]

    @property
    def finished(self) -> bool:
        return all(g["id"] in self.goals_done for g in self.scenario.get("goals", []))

    @property
    def session_grade(self) -> int:
        return round(sum(self.turn_scores) / len(self.turn_scores)) if self.turn_scores else 0

    def submit_turn(self, user_text: str) -> TurnResult:
        user_text = user_text.strip()
        self.transcript.append({"speaker": "user", "text": user_text})

        completed_now = []
        for g in self.scenario.get("goals", []):
            if g["id"] in self.goals_done:
                continue
            if all(ph in user_text for ph in g["requiredPhrases"]):
                self.goals_done.add(g["id"])
                completed_now.append(g["id"])

        total_goals = len(self.scenario.get("goals", [])) or 1
        coverage = len(self.goals_done) / total_goals
        keigo_score, hits, penalties = self._score_keigo(user_text)
        reg = analyze(user_text)

        grade = round(100 * (0.55 * coverage + 0.30 * keigo_score
                             + 0.15 * reg.formality_score))
        grade = max(0, min(100, grade))
        self.turn_scores.append(grade)

        reply = self._match_npc(user_text)
        self.transcript.append({"speaker": "npc", "text": reply})

        return TurnResult(
            npc_reply=reply,
            goal_progress=self.progress(),
            goals_completed_this_turn=completed_now,
            keigo_score=keigo_score,
            rubric_hits=hits,
            rubric_penalties=penalties,
            register_verdict=reg.verdict,
            turn_grade=grade,
            session_grade=self.session_grade,
            finished=self.finished,
        )

    def summary(self) -> dict:
        missed = [g["text"] for g in self.scenario.get("goals", [])
                  if g["id"] not in self.goals_done]
        return {
            "scenario": self.scenario["id"],
            "title": self.scenario["title"],
            "grade": self.session_grade,
            "goalsCompleted": len(self.goals_done),
            "goalsTotal": len(self.scenario.get("goals", [])),
            "missedGoals": missed,
            "transcript": self.transcript,
            "finished": self.finished,
        }


def list_scenarios() -> list[dict]:
    return [{"id": s["id"], "title": s["title"], "location": s["location"],
             "level": s["level"], "description": s["description"],
             "goals": len(s.get("goals", []))} for s in load_scenarios()]


def start(scenario_id: str) -> RoleplaySession:
    for s in load_scenarios():
        if s["id"] == scenario_id:
            return RoleplaySession(s)
    raise KeyError(f"unknown scenario '{scenario_id}'")
