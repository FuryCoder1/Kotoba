"""Media Sync — import anime/drama subtitles and mine vocabulary.

Accepts SRT or WebVTT text (the format every ripper/streamer subtitle export
uses), parses cues, tokenises the Japanese with the reading engine, filters
against the learner's known vocabulary, and produces a per-episode word list
ranked by frequency — ready to become Kotoba flashcards.
"""
from __future__ import annotations

import re
from collections import Counter

from .reading import annotate_text, tokenize_japanese

# cue header: "12\n00:01:23,456 --> 00:01:25,100" (SRT) or with '.' (VTT)
_CUE_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+)?\s*\n?"
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})[^\n]*\n"
    r"((?:(?!\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->).+\n?)+)")


def parse_subtitles(text: str) -> list[dict]:
    """Return [{start,end,text}] from SRT/VTT content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # strip VTT header/NOTE blocks
    text = re.sub(r"WEBVTT.*?\n\n", "", text, flags=re.S)
    text = re.sub(r"NOTE[^\n]*\n(?:[^\n]*\n)*?\n", "", text)
    cues = []
    for m in _CUE_RE.finditer(text):
        raw = " ".join(l.strip() for l in m.group(3).strip().splitlines())
        raw = re.sub(r"<[^>]+>", "", raw)          # styling tags
        if not raw:
            continue
        cues.append({"start": m.group(1), "end": m.group(2), "text": raw})
    if not cues:                                   # fallback: plain lines
        cues = [{"start": "00:00:00.000", "end": "00:00:00.000", "text": ln.strip()}
                for ln in text.splitlines() if ln.strip()]
    return cues


def _has_japanese(s: str) -> bool:
    return bool(re.search(r"[぀-ゟ゠-ヿ一-龥]", s))


def extract_vocabulary(subtitle_text: str,
                       known_readings: list[str] | None = None,
                       min_level: str | None = None) -> dict:
    """Mine an episode's subtitles for new vocabulary.

    Returns frequency-ranked word list + furigana-marked transcript lines.
    """
    known = set(known_readings or [])
    cues = [c for c in parse_subtitles(subtitle_text) if _has_japanese(c["text"])]
    counter: Counter[str] = Counter()
    meta: dict[str, dict] = {}
    lines = []
    for c in cues:
        ann = annotate_text(c["text"], mode="unknown-only", known_readings=list(known))
        lines.append({"time": c["start"], "furiganaText": ann["readable"],
                      "coverage": ann["coverage"]})
        for seg in tokenize_japanese(c["text"]):
            if seg.kind == "kanji-word" and seg.entry:
                e = seg.entry
                if e["reading"] in known:
                    continue
                counter[e["word"]] += 1
                meta[e["word"]] = {
                    "wordId": e["id"], "word": e["word"], "reading": e["reading"],
                    "romaji": e.get("romaji", ""), "glosses": e["glosses"],
                    "pitchPattern": e.get("pitchPattern", ""), "level": e.get("level", ""),
                }
    words = [meta[w] | {"frequency": n} for w, n in counter.most_common()]
    if min_level:                                    # filter JLPT ceiling N5<N4<...
        order = ["N5", "N4", "N3", "N2", "N1"]
        cap = order.index(min_level) if min_level in order else 99
        words = [w for w in words if w["level"] in order[:cap + 1]]
    return {
        "cueCount": len(cues),
        "newWordCount": len(words),
        "words": words,
        "transcriptLines": lines[:50],
        "flashcards": [dict(w, source="media-sync") for w in words],
    }
