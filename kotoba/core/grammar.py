"""Grammar Deconstructor — the particle/predicate dependency engine.

Given a sentence (from the corpus, or produced by the client's tokenizer), we
build a *dependency map* that the mobile UI can draw as arcs over the text:

    私 は   東京 の 学校 で 日本語 を 勉強します
    └──wa──┘          └no┘  └de┘      └wo┘    └root
       │                 │     │         │
     topic            poss  location   object

Rules encoded here (the "contextual grammar" pain point):

1. A case particle governs the **noun phrase immediately to its left** and
   attaches to that NP's head; the whole PP then modifies the predicate or the
   noun selected by the particle's subcategorisation frame.
2. `wa` is a *topic* marker: it overrides the grammatical case of its host
   (`ga -> wa`, `wo -> wa`) and scopes over the whole clause.
3. `ga` marks the **nominative subject**; with potential/agentive predicates it
   can also mark the object (分かる, 好きです) — we detect that pattern.
4. `ni` vs `de`: `ni` = goal/target/time/indirect object/existence-locative;
   `de` = action-location/instrument/means/limit.  We disambiguate using the
   semantic class of the predicate (existence verbs vs action verbs).
5. `no` = genitive/modifier linking two nouns.
6. `e` (へ) = directional goal, softer than `ni`.
7. Conjunction/quotative particles (`to`, `ya`, `ka`, `yo`, `ne`) attach to the
   preceding constituent without re-scoping the predicate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..data import load_sentences
from .models import Sentence, Token

# --------------------------------------------------------------------------
# Particle knowledge base
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ParticleSpec:
    surface: str
    label: str                  # human-readable role
    explains: str               # one-line learner explanation
    selects: str                # what it marks on its left
    scope: str                  # "predicate" | "noun" | "clause"
    contrast_with: tuple[str, ...] = ()

SURFACE_ROMAN = {
    "は": "wa", "が": "ga", "を": "wo", "に": "ni", "で": "de", "の": "no",
    "へ": "e", "と": "to", "から": "kara", "まで": "made", "も": "mo",
    "か": "ka", "よ": "yo", "ね": "ne",
}

_PARTICLE_SPECS = [
    ParticleSpec("は", "topic",
                 "Marks the topic — what the sentence is about. Replaces が/を when topicalising.",
                 "noun phrase", "clause", ("が",)),
    ParticleSpec("が", "subject",
                 "Marks the grammatical subject (new information), or the object of potentials/likes.",
                 "noun phrase", "predicate", ("は",)),
    ParticleSpec("を", "object",
                 "Marks the direct object of a transitive verb (written 'wo' in romaji).",
                 "noun phrase", "predicate"),
    ParticleSpec("に", "target",
                 "Goal / recipient / specific time / location of existence or arrival.",
                 "noun phrase", "predicate", ("で",)),
    ParticleSpec("で", "location-action",
                 "Where an action happens, the means/instrument used, or a scope/limit.",
                 "noun phrase", "predicate", ("に",)),
    ParticleSpec("の", "possessive",
                 "Links two nouns: possession, modification, apposition ('a book of Japanese').",
                 "noun", "noun"),
    ParticleSpec("へ", "direction",
                 "Direction of movement — softer and more poetic than に.",
                 "noun phrase", "predicate", ("に",)),
    ParticleSpec("と", "companion-quote",
                 "'and'/'with' (list, accompaniment) or quotative 'that … said'.",
                 "noun phrase", "predicate"),
    ParticleSpec("から", "source",
                 "Starting point in space or time ('from', 'since').",
                 "noun phrase", "predicate", ("まで",)),
    ParticleSpec("まで", "limit",
                 "Endpoint in space or time ('until', 'as far as').",
                 "noun phrase", "predicate", ("から",)),
    ParticleSpec("も", "also",
                 "Replaces the case particle to mean 'also/too' while keeping its case role.",
                 "noun phrase", "clause"),
    ParticleSpec("か", "question",
                 "Sentence-final question marker, or 'or' between alternatives.",
                 "clause", "clause"),
    ParticleSpec("よ", "assertion",
                 "Adds emphasis/new information to the listener.",
                 "clause", "clause"),
    ParticleSpec("ね", "confirmation",
                 "Seeks agreement or softens ('isn't it?', 'right?').",
                 "clause", "clause"),
]

# Primary key is the *romaji* case label that the tokenizer emits in
# `Token.particle` (wa/ga/wo/ni/de/no/e/to/kara/made/mo/ka/yo/ne), so lookups
# hit directly.  SURFACES maps a written particle back to that label for
# English-language learner questions ("why に and not で?").
PARTICLES: dict[str, ParticleSpec] = {SURFACE_ROMAN[p.surface]: p for p in _PARTICLE_SPECS}
SURFACES: dict[str, str] = dict(SURFACE_ROMAN)

# Verb classes used for the に/で disambiguation
EXISTENCE_VERBS = ("います", "いる", "あります", "ある", "住んでいます", "住む")
MOTION_VERBS = ("行きます", "行く", "来ます", "来る", "着きます", "着く", "帰ります", "帰る", "向かいます")
TRANSITIVE_HINTS = ("食べ", "見", "読", "買", "書", "勉強", "使い", "教", "くれ", "もら")


def _is_verb(token: Token) -> bool:
    return token.pos in ("verb", "auxiliary verb", "copula") or token.pos.startswith("verb")


def annotate(tokens: list[Token]) -> list[Token]:
    """Assign a syntactic `role` to every token using the rules above."""
    n = len(tokens)
    root_idx = next((i for i, t in enumerate(tokens) if t.depends_on == -1), n - 1)
    predicate = tokens[root_idx]

    out = [Token(**{k: getattr(t, k) for k in
                    ("surface", "reading", "lemma", "pos", "particle", "depends_on")})
           for t in tokens]
    out[root_idx].role = "predicate(root)"

    # First pass: content words get provisional roles from their head.
    for i, t in enumerate(out):
        if t.particle is None:
            # content word: its role comes from what it depends on
            head = out[t.depends_on] if 0 <= t.depends_on < n else None
            if head is not None and head.particle in ("の",):
                t.role = "modifier(noun)"
            elif i == root_idx:
                t.role = "predicate(root)"
            elif head is not None and _is_verb(head):
                t.role = "complement"
            else:
                t.role = "content"
            continue

        key = t.particle if t.particle in PARTICLES else SURFACES.get(t.particle)
        spec = PARTICLES.get(key)
        host_idx = t.depends_on if 0 <= t.depends_on < n else max(0, i - 1)
        host = out[host_idx]
        role = spec.label if spec else t.particle

        # に/で refinement against the predicate class
        if key == "ni" and any(predicate.surface.startswith(v[:3]) for v in MOTION_VERBS):
            role = "goal(direction)"
            host.role = "goal-noun"
        elif key == "ni" and any(v[:3] in predicate.surface for v in EXISTENCE_VERBS):
            role = "existence-location"
            host.role = "location-noun"
        elif key == "de":
            role = "location-action"
            host.role = _de_role(host, predicate)
        elif key == "ga" and _potential_object(tokens, i, host_idx, predicate):
            role = "object(potential/like)"
            host.role = "object-noun"
        elif key == "to" and _conditional_to(out, i, host_idx, out):
            role = "condition"
            host.role = "conditional-clause"
        else:
            host.role = _host_role(spec, host, predicate)
        t.role = role

    # Second pass: propagate each particle's semantic label onto its host NP so
    # the UI can colour nouns by case role even when annotate order varies.
    for i, t in enumerate(out):
        if t.particle is None:
            continue
        key = t.particle if t.particle in PARTICLES else SURFACES.get(t.particle)
        spec = PARTICLES.get(key)
        host = out[t.depends_on] if 0 <= t.depends_on < n else None
        if host is None or spec is None:
            continue
        if host.role in (None, "content", "complement", "modifier(noun)"):
            host.role = _host_role(spec, host, predicate)

    # temporal nouns that sit bare (昨日, 明日) act as adverbials
    for i, t in enumerate(out):
        if t.pos == "noun" and t.role is None and any(
                w in t.surface for w in ("今日", "明日", "昨日", "毎日", "毎週")):
            t.role = "time(adverbial)"
    for t in out:
        if t.role is None:
            t.role = "content"
    return out


def _de_role(host: Token, predicate: Token) -> str:
    """で after a place noun = venue of the action; elsewhere it is a means."""
    PLACE_SUFFIX = ("駅", "所", "家", "国", "市", "町", "村", "学校", "会社", "店",
                    "大阪", "東京", "京都", "北海道", "中", "前", "後ろ")
    if any(k in host.surface for k in PLACE_SUFFIX) or host.pos in ("proper noun", "place"):
        return "action-location"
    return "means-instrument"


def _conditional_to(tokens: list[Token], part_i: int, host_i: int, out: list[Token]) -> bool:
    """と used as 'if/when' (attaches to a verb/adjective), not 'and/with'.

    Conditional と takes a *predicative* host; companion/list と takes a noun.
    We additionally require that something downstream still depends on the
    particle (the apodosis), so sentence-final quotative と is not mislabelled.
    """
    host = out[host_i]
    if host.pos in ("noun", "pronoun", "proper noun", "determiner"):
        return False
    return host.pos == "verb" or host.pos == "adjective" or any(
        o.depends_on == part_i for o in out)


def _potential_object(tokens, part_idx, host_idx, predicate) -> bool:
    """が marking the *object* of a potential form / likes / needs."""
    surf = predicate.surface
    return any(k in surf for k in ("分かります", "分かる", "できます", "できる",
                                   "好き", "ほしい", "必要", "上手", "得意"))


def _host_role(spec: ParticleSpec | None, host: Token, predicate: Token) -> str:
    if spec is None:
        return "content"
    mapping = {
        "topic": "topic",
        "subject": "subject",
        "object": "direct-object",
        "target": "indirect-object/time",
        "location-action": "location(instrument)",
        "possessive": "possessed-noun",
        "direction": "goal",
        "companion-quote": "companion/quoted-content",
        "source": "source",
        "limit": "limit",
        "also": "focus-particle(host)",
    }
    return mapping.get(spec.label, "dependent")


# --------------------------------------------------------------------------
# Corpus + public API
# --------------------------------------------------------------------------

def load_corpus() -> list[Sentence]:
    corpus = []
    for s in load_sentences():
        toks = [Token(surface=t["surface"], reading=t["reading"], lemma=t["lemma"],
                      pos=t["pos"], particle=t.get("particle"),
                      depends_on=int(t.get("dependsOn", -1)))
                for t in s["tokens"]]
        corpus.append(Sentence(id=s["id"], text=s["text"], romaji=s["romaji"],
                               translation=s["translation"], level=s["level"],
                               furigana=list(s.get("furigana", [])), tokens=toks))
    return corpus


def deconstruct(sentence: Sentence) -> dict:
    """Return the visual-map payload consumed by the Flutter overlay."""
    annotated = annotate(sentence.tokens)
    arcs = []
    for i, t in enumerate(annotated):
        if t.particle and 0 <= t.depends_on < len(annotated):
            key = t.particle if t.particle in PARTICLES else SURFACES.get(t.particle, t.particle)
            spec = PARTICLES.get(key)
            src = t.depends_on
            if key == "no":
                # The genitive links two nouns: draw from the *modifier*
                # (the noun that depends on this particle's host) when present.
                mods = [j for j, o in enumerate(annotated)
                        if o.depends_on == t.depends_on and o.particle is None]
                if mods:
                    src = mods[0]
            arcs.append({
                "from": src, "to": i,
                "particle": key,
                "surface": t.surface,
                "label": spec.label if spec else key,
                "headSurface": annotated[t.depends_on].surface,
            })
    return {
        "id": sentence.id,
        "text": sentence.text,
        "romaji": sentence.romaji,
        "translation": sentence.translation,
        "level": sentence.level,
        "furigana": sentence.furigana,
        "tokens": [t.as_dict() for t in annotated],
        "arcs": arcs,
        "particleNotes": [
            {"particle": a["particle"], "surface": a["surface"],
             "explanation": PARTICLES[a["particle"]].explains}
            for a in arcs if a["particle"] in PARTICLES
        ],
    }


ROMAN_TO_PARTICLE = {
    "wa": "は", "ha": "は", "ga": "が", "ka": "か", "o": "を", "wo": "を",
    "ni": "に", "de": "で", "no": "の", "e": "へ", "to": "と",
    "kara": "から", "made": "まで", "mo": "も", "yo": "よ", "ne": "ね",
}


def _roman_to_particle(roman: str) -> str | None:
    return ROMAN_TO_PARTICLE.get(roman)


def explain(particle: str) -> str:
    key = particle if particle in PARTICLES else SURFACES.get(particle)
    spec = PARTICLES.get(key)
    if not spec:
        return f"{particle} is not in the particle knowledge base yet."
    contrast = ""
    if spec.contrast_with:
        others = "、".join(spec.contrast_with)
        contrast = f" Often confused with {others}."
    return f"{spec.surface}（{spec.label}）: {spec.explains}{contrast}"


def find_contrast(question: str) -> str | None:
    """Map a learner's English question onto a particle contrast note.

    e.g. 'why did we use ni instead of de here?' -> the に vs で explanation.
    """
    q = question.lower()
    # allow "instead of", "vs", "or", "rather than" between the two particles
    m = re.findall(r"\b(ni|de|wa|ga|wo|o|no|e|to|kara|made|mo|ka)\b"
                   r"(?:\s+(?:instead\s+of|rather\s+than|vs\.?|or|instead))?\s*(?:of\s+)?"
                   r"\b(ni|de|wa|ga|wo|o|no|e|to|kara|made|mo|ka)\b", q)
    for a, b in m:
        pa, pb = _roman_to_particle(a), _roman_to_particle(b)
        if pa and pb and pa != pb:
            return f"{explain(pa)}   ⟷   {explain(pb)}"
    singles = [w for w in re.findall(r"\b(ni|de|wa|ga|wo|o|no|e|to|kara|made|mo|ka)\b", q)]
    if singles:
        return explain(_roman_to_particle(singles[0]))
    return None
