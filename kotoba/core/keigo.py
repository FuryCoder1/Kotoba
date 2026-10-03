"""Keigo (敬語) engine — register detection and politeness feedback.

Japanese politeness is a three-way system plus plain speech:

  * 丁寧語 teineigo      – polite forms (desu/masu ~です/ます)
  * 尊敬語 sonkeigo      – honorific *elevating* the other party
                          (おっしゃる, いらっしゃる, なさる, ご覧になる…)
  * 謙譲語 kenjougo      – humble *lowering* oneself for the other party
                          (申す, 参る, いたす, いただく, お~する, ~てあげる→差し上げる…)
  * ビジネス／クッション言葉 – softeners used before requests/complaints
                          (恐れ入りますが, 申し訳ありませんが, ちょっと…)

The grader answers two questions the blueprint cares about:
  1. Is this utterance at the right register for an unfamiliar shop clerk?
  2. Did the learner commit the classic error of applying sonkeigo to
     their OWN action (e.g. *ビールをいらっしゃいます)?
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Pattern banks (ordered: more specific first when scanning)
# ---------------------------------------------------------------------------

SONKEIGO = [
    r"おっしゃっ?て?",            # 仰る (say)
    r"いらっしゃい(ます|ませ|なさい)?",  # honourable "be/go/come" + welcome interjection
    r"なさっ?て?",                # なさる (do)
    r"ご覧(?:に)?(?:なる|なり)",   # see
    r"お(?:掛け|かけ)(?:ください|になります)",
    r"召し上が(?:ります|れ)",      # eat (honorific)
]

KENJOUGO = [
    r"申(?:します|して|した)",     # 言う -> humble
    r"参(?:ります|って|った)",     # 行く/来る -> humble
    r"いた(?:します|して|した)",   # する -> humble
    r"いただ(?:きます|いて|いた)", # もらう/食べる -> humble
    r"お(?:[ぁ-んァ-ンー]{1,6}?)(?:します|して|した)",  # お~する productive humble (lazy, no clause-crossing)
    r"拝見(?:します|致しました)?",
    r"承知(?:しました|いた(?:します|して))",
    r"伺(?:います|う|った|って)",   # 聞く/訪ねる -> humble
]

TEINEIGO = [
    r"[ぁ-んァ-ンーa-zA-Z一-龥]{1,6}(?:ます|ません|ました|ませんでした)",
    r"(?:です|でした|じゃないですか|でしょうか)",
    r"(?:ください|下さい|お願いします|お願いいたします|お願いできますか)",
]

CASUAL = [
    r"くれよ|くれる|ちょうだい|ねえ|おい",
    r"[だぞぜ]$|だよ$|なの$",
    r"うん。|yeah",
    r"なさいよ",
]

SOFTENERS = [
    r"恐れ入りますが",
    r"申し訳(?:ありません|ございません)が?",
    r"(?:ちょっと|少し)(?:、|,)?",
    r"すみませんが",
    r"あの(?:う)?、?",
    r"お手数ですが",
]

# Sonkeigo mistakenly applied to the speaker's own action.
SELF_HONORIFIC_ERRORS = [
    r"(?:私|わたし|ワタシ)[はも]?\s*(?:ビール|食べ物)?\s*(?:を)?いらっしゃいま",
    r"いらっしゃいます(?:ビール|コーヒー|メニュー)",
    r"おっしゃ(?:います|る)(?:私|ぼく|僕|わたし)",
    r"(?:私|僕|ぼく)が?おっしゃ",
    r"(?:召し上がります|お召し上がり)(?:た)?(?:私)",
]

QUICK_PHRASES = {
    "sumimasen": "すみません",
    "onegaishimasu": "お願いします",
    "arigatou gozaimasu": "ありがとうございます",
    "itadakimasu": "いただきます",
    "gochisousama": "ごちそうさまでした",
}


@dataclass
class KeigoAnalysis:
    text: str
    registers: list[str] = field(default_factory=list)   # e.g. ["teineigo","kenjougo"]
    softeners: list[str] = field(default_factory=list)
    self_honorific_errors: list[str] = field(default_factory=list)
    casual_hits: list[str] = field(default_factory=list)
    formality_score: float = 0.0                          # 0..1
    verdict: str = ""

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "registers": self.registers,
            "softeners": self.softeners,
            "selfHonorificErrors": self.self_honorific_errors,
            "casualHits": self.casual_hits,
            "formalityScore": round(self.formality_score, 3),
            "verdict": self.verdict,
        }


def _find_all(patterns: list[str], text: str) -> list[str]:
    hits: list[str] = []
    for p in patterns:
        for m in re.finditer(p, text):
            frag = m.group(0).strip()
            if frag and frag not in hits:
                hits.append(frag)
    return hits


def analyze(text: str) -> KeigoAnalysis:
    """Classify one learner utterance into keigo registers with feedback."""
    a = KeigoAnalysis(text=text)
    a.self_honorific_errors = _find_all(SELF_HONORIFIC_ERRORS, text)
    a.casual_hits = _find_all(CASUAL, text)
    a.softeners = _find_all(SOFTENERS, text)

    if _find_all(SONKEIGO, text):
        a.registers.append("sonkeigo")
    if _find_all(KENJOUGO, text):
        a.registers.append("kenjougo")
    if _find_all(TEINEIGO, text):
        a.registers.append("teineigo")

    score = 0.0
    if "teineigo" in a.registers:
        score += 0.45
    if "kenjougo" in a.registers:
        score += 0.30
    if "sonkeigo" in a.registers and not a.self_honorific_errors:
        score += 0.15
    score += min(0.10, 0.05 * len(a.softeners))
    score -= 0.35 * len(a.casual_hits)
    score -= 0.40 * len(a.self_honorific_errors)
    a.formality_score = max(0.0, min(1.0, score))

    if a.self_honorific_errors:
        a.verdict = (
            f"⚠️ Sonkeigo (尊敬語) was applied to your own action "
            f"({a.self_honorific_errors[0]}). Honorifics describe the *other* "
            f"person's behaviour; lower yourself with kenjougo instead "
            f"(申します / 参ります / いたします)."
        )
    elif a.casual_hits and "teineigo" not in a.registers:
        a.verdict = (
            f"Casual speech ({a.casual_hits[0]}) toward an unfamiliar clerk "
            f"reads as rude. Use です・ます form + ください."
        )
    elif a.formality_score >= 0.75:
        a.verdict = "Excellent register control — natural customer-level keigo."
    elif "teineigo" in a.registers:
        a.verdict = "Polite enough for daily service encounters; add softeners or humble forms to level up."
    else:
        a.verdict = "No polite markers detected — this would sound blunt in a shop."
    return a


def expected_register(context: str) -> str:
    """Tiny heuristic used by Sensei/roleplay to pick a target register."""
    c = context.lower()
    if any(k in c for k in ("shop", "clerk", "counter", "station", "hotel", "izakaya")):
        return "teineigo+kenjougo"
    if any(k in c for k in ("boss", "professor", "senior", "customer")):
        return "sonkeigo-aware teineigo"
    if any(k in c for k in ("friend", "family", "casual")):
        return "plain/casual"
    return "teineigo"
