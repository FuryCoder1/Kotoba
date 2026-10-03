"""AI Sensei — personalised tutor with mistake memory.

The LLM glue (Whisper STT + GPT-style generation) lives server-side; this
module is the *deterministic brain* it calls:

  * MistakeMemory   – counts learner errors by category (particle confusion,
                      keigo register slips, pitch drops, wrong kanji reading).
                      Persisted as JSON; in production mirrored to pgvector so
                      an LLM can retrieve "what does *this* user keep doing?"
  * quiz_for()      – generates targeted multiple-choice items from the corpus
                      centred on the learner's weakest category
  * answer()        – routes a free-text English question to the right engine:
                        - particle contrast -> grammar.find_contrast
                        - politeness/keigo  -> keigo.analyze / register notes
                        - word meaning/pitch-> lexicon lookup w/ accent info
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from ..data import load_lexicon
from . import grammar, keigo, pitch
from .grammar import load_corpus


CATEGORIES = ("particle-wa-ga", "particle-ni-de", "particle-other",
              "keigo-register", "pitch-accent", "kanji-reading")


@dataclass
class MistakeMemory:
    counts: Counter = field(default_factory=Counter)
    examples: dict = field(default_factory=dict)          # category -> list of snippets
    roleplay_grades: list[int] = field(default_factory=list)

    def log(self, category: str, detail: str = "") -> None:
        assert category in CATEGORIES, f"unknown category {category}"
        self.counts[category] += 1
        if detail:
            self.examples.setdefault(category, []).append(detail)
            self.examples[category] = self.examples[category][-5:]

    def log_roleplay(self, grade: int, violations: list[str]) -> None:
        self.roleplay_grades.append(grade)
        for v in violations:
            self.log("keigo-register", v)

    @property
    def weakest(self) -> str | None:
        if not self.counts:
            return None
        return self.counts.most_common(1)[0][0]

    def profile(self) -> dict:
        total = sum(self.counts.values())
        return {
            "totalMistakes": total,
            "counts": {c: self.counts.get(c, 0) for c in CATEGORIES},
            "weakestCategory": self.weakest,
            "recentExamples": self.examples,
            "avgRoleplayGrade": (round(sum(self.roleplay_grades)
                                       / len(self.roleplay_grades))
                                 if self.roleplay_grades else None),
        }

    def export(self) -> str:
        return json.dumps({"counts": dict(self.counts),
                           "examples": self.examples,
                           "roleplayGrades": self.roleplay_grades},
                          ensure_ascii=False)

    @classmethod
    def import_json(cls, blob: str) -> "MistakeMemory":
        d = json.loads(blob)
        m = cls(counts=Counter(d.get("counts", {})),
                examples=d.get("examples", {}),
                roleplay_grades=d.get("roleplayGrades", []))
        return m


# ---------------------------------------------------------------------------
# Targeted quiz generation
# ---------------------------------------------------------------------------

_QUIZ_TEMPLATES = {
    "particle-wa-ga": {
        "blanks": ["私＿＿学生です。", "猫＿＿います。"],
        "options": ["は", "が"],
        "answers": {"私＿＿学生です。": "は", "猫＿＿います。": "が"},
        "why": "は marks the known topic; が introduces new information or the subject itself.",
    },
    "particle-ni-de": {
        "blanks": ["学校＿＿日本語を勉強します。", "公園＿＿会います。"],
        "options": ["に", "で"],
        "answers": {"学校＿＿日本語を勉強します。": "で", "公園＿＿会います。": "に"},
        "why": "で hosts a voluntary action; に pins existence/arrival/meeting locations.",
    },
    "particle-other": {
        "blanks": ["東京＿＿大阪まで行きます。", "これ＿＿本です。"],
        "options": ["から", "は"],
        "answers": {"東京＿＿大阪まで行きます。": "から", "これ＿＿本です。": "は"},
        "why": "から marks the starting point (pairs with まで).",
    },
    "keigo-register": {
        "blanks": ["店員さんに「ビール＿＿ください」と言います。",
                   "先生に質問する時、最も丁寧なのは？"],
        "options": ["を", "恐れ入りますが、伺ってもよろしいでしょうか"],
        "answers": {"店員さんに「ビール＿＿ください」と言います。": "を",
                    "先生に質問する時、最も丁寧なのは？": "恐れ入りますが、伺ってもよろしいでしょうか"},
        "why": "Quotation slots still need を; softener + humble 伺う raises register.",
    },
    "pitch-accent": {
        "blanks": ["「箸」(chopsticks) のピッチ型は？",
                   "「橋」(bridge) のピッチ型は？"],
        "options": ["LH+L (②)", "HL+L (①)"],
        "answers": {"「箸」(chopsticks) のピッチ型は？": "LH+L (②)",
                    "「橋」(bridge) のピッチ型は？": "HL+L (①)"},
        "why": "Both are はし; only the accent nucleus differs — pitch distinguishes meaning.",
    },
    "kanji-reading": {
        "blanks": ["「水曜日」の読みは？", "「山」の音読みは？"],
        "options": ["すいようび", "サン"],
        "answers": {"「水曜日」の読みは？": "すいようび", "「山」の音読みは？": "サン"},
        "why": "Onyomi appears in compound words like 水曜日; kunyomi stands alone as 山やま.",
    },
}


def quiz_for(memory: MistakeMemory, limit: int = 4) -> dict:
    cat = memory.weakest or "particle-wa-ga"
    tpl = _QUIZ_TEMPLATES[cat]
    items = []
    for blank in tpl["blanks"][:limit]:
        items.append({
            "prompt": blank,
            "options": tpl["options"],
            "answer": tpl["answers"][blank],
            "category": cat,
        })
    return {
        "targetCategory": cat,
        "rationale": f"You have logged {memory.counts.get(cat, 0)} mistakes here — drilling this first.",
        "teachingNote": tpl["why"],
        "items": items,
    }


# ---------------------------------------------------------------------------
# Free-text answering
# ---------------------------------------------------------------------------

_LEX_INDEX: dict[str, dict] | None = None


def _lex_index() -> dict[str, dict]:
    global _LEX_INDEX
    if _LEX_INDEX is None:
        idx = {}
        for e in load_lexicon():
            idx[e["word"]] = e
            idx[e["reading"]] = e
            idx[e["romaji"].lower()] = e
        _LEX_INDEX = idx
    return _LEX_INDEX


def answer(question: str, memory: MistakeMemory | None = None) -> dict:
    """Route an English/Japanese learner question to the right engine."""
    q = question.strip()
    # 1) particle contrast question?
    contrast = grammar.find_contrast(q)
    if contrast and re.search(r"\b(why|vs|instead|rather)\b.*\b(ni|de|wa|ga|wo|o|no|to|e|kara|made|mo)\b|\b(ni|de|wa|ga)\b.*(instead|or|vs)", q, re.I):
        note = ""
        if memory and memory.counts.get("particle-wa-ga", 0) + memory.counts.get("particle-ni-de", 0) > 2:
            note = " (This is one of your recurring weak spots — I've queued a drill.)"
        return {"engine": "grammar", "response": contrast + note,
                "suggestedQuiz": quiz_for(memory)["items"] if memory and memory.weakest else []}
    # 2) keigo question?
    if re.search(r"keigo|polite|honorific|humble|respect|ていねい|敬語", q, re.I):
        reg = keigo.expected_register(q)
        return {"engine": "keigo",
                "response": (f"For that context aim for {reg}. Teineigo (ます・です) is the floor; "
                             "use kenjougo (申します/参ります/いたします) to lower yourself and "
                             "sonkeigo (おっしゃる/いらっしゃる) ONLY for the other person."),
                "expectedRegister": reg}
    # 3) vocabulary / pitch lookup: quoted or bare romaji/kana/kanji token
    tokens = re.findall(r"[一-龥぀-ゟ゠-ヿ]+|[A-Za-z]{2,}", q)
    for t in tokens:
        hit = _lex_index().get(t) or _lex_index().get(t.lower())
        if hit:
            targets = pitch.PitchDictionary().lookup(hit["reading"])
            pinfo = ""
            if targets:
                tg = targets[0]
                pinfo = f" Pitch: {tg.pattern} ({tg.accent_type}); say it with a fall after mora {tg.accent_location or 'none'}."
            pairs = pitch.PitchDictionary().minimal_pairs(hit["reading"])
            extra = ""
            if pairs:
                extra = f" ⚠️ Minimal pair! Same kana, different pitch changes meaning: {[ (p.meaning, q2.meaning) for p, q2 in pairs ]}"
            return {"engine": "lexicon",
                    "response": f"{hit['word']}（{hit['reading']}）: {', '.join(hit['glosses'])}.{pinfo}{extra}",
                    "entry": hit}
    # 4) fallback: general study advice from profile
    prof = memory.profile() if memory else None
    tip = ""
    if prof and prof["weakestCategory"]:
        tip = f" Based on your log, focus on {prof['weakestCategory'].replace('-', ' ')} today."
    return {"engine": "fallback",
            "response": ("I couldn't map that to a specific rule yet. Try asking about a "
                         "particle ('why ni instead of de?'), a word ('what is hashi?'), "
                         "or politeness ('how do I complain politely?')." + tip)}
