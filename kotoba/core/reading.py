"""Immersive Reading Engine — smart furigana, OCR pipeline stub, shadowing.

Three responsibilities from the blueprint:

1. Smart Furigana (``annotate_text``): longest-match kanji segmentation over
   the bundled lexicon; furigana can be toggled globally, per-word, or set to
   "show only words you don't know yet" (``known_readings``) — exactly how a
   real reader's mental dictionary works.

2. Manga OCR (``ocr_image``): in production this delegates to PaddleOCR /
   ML Kit; here we expose the *contract* (image -> blocks of text) plus
   ``process_ocr_text`` which takes extracted Japanese text and produces the
   tap-to-add flashcard payload (word + reading + gloss + pitch pattern).

3. Listening & Shadowing (``shadowing_plan``): waveform-comparison scoring is
   delegated to the pitch engine; this module builds the practice plan around
   a corpus sentence — speed ladder 0.75x -> 1.0x -> 1.25x and pause points at
   each particle/clause boundary taken from the grammar deconstructor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..data import load_lexicon
from .grammar import load_corpus, deconstruct
from .models import WordEntry


# ---------------------------------------------------------------------------
# 1. Longest-match tokenizer + smart furigana
# ---------------------------------------------------------------------------

_LEXICON: list[WordEntry] | None = None


def _lexicon() -> list[WordEntry]:
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = [WordEntry.from_dict(d) for d in load_lexicon()]
    return _LEXICON


def _surface_forms(entry: WordEntry) -> list[str]:
    forms = {entry.word}
    # strip okurigana: 勉強する -> also match 勉強
    m = re.match(r"^([一-龥]+?)(?:[ぁ-ん]+)$", entry.word)
    if m:
        forms.add(m.group(1))
    return [f for f in forms if f and not f.isdigit()]


@dataclass
class Segment:
    text: str
    kind: str                       # "kanji-word" | "kana" | "punct" | "unknown-kanji"
    entry: dict | None = None       # lexicon hit (as plain dict)
    show_furigana: bool = False
    furigana: str | None = None

    def as_dict(self) -> dict:
        d = {"text": self.text, "kind": self.kind,
             "furigana": self.furigana if self.show_furigana else None}
        if self.entry:
            d["entry"] = {"id": self.entry["id"], "reading": self.entry["reading"],
                          "romaji": self.entry.get("romaji", ""),
                          "glosses": self.entry["glosses"],
                          "pitchPattern": self.entry.get("pitchPattern", ""),
                          "level": self.entry.get("level", "")}
        return d


def tokenize_japanese(text: str) -> list[Segment]:
    """Greedy longest-match segmentation against the lexicon."""
    index: dict[str, WordEntry] = {}
    for e in _lexicon():
        for form in _surface_forms(e):
            cur = index.get(form)
            if cur is None or len(e.glosses[0]) >= len(cur.glosses[0]):
                index.setdefault(form, e)
    segments: list[Segment] = []
    i, n = 0, len(text)
    while i < n:
        matched = None
        for L in range(min(6, n - i), 0, -1):     # max word length 6 chars
            chunk = text[i:i + L]
            if chunk in index:
                matched = (chunk, index[chunk])
                break
        if matched:
            chunk, entry = matched
            segments.append(Segment(chunk, "kanji-word", entry=entry.__dict__
                                    if not isinstance(entry, dict) else entry))
            i += len(chunk)
        elif re.match(r"[぀-ゟ゠-ヿ]", text[i]):
            m = re.match(r"[぀-ゟ゠-ヿー]+", text[i:])
            seg = m.group(0)
            segments.append(Segment(seg, "kana"))
            i += len(seg)
        elif re.match(r"[、。！？・「」『』（）…〜\s,.!?]", text[i]):
            m = re.match(r"[、。！？・「」『』（）…〜\s,.!?]+", text[i:])
            segments.append(Segment(m.group(0), "punct"))
            i += len(m.group(0))
        else:
            m = re.match(r"[一-龥]+", text[i:])
            if m:
                segments.append(Segment(m.group(0), "unknown-kanji"))
                i += len(m.group(0))
            else:
                m = re.match(r".+?(?=[぀-ゟ゠-ヿ、。！？・「」『』（）…〜一-龥]|$)", text[i:])
                seg = m.group(0) or text[i]
                segments.append(Segment(seg, "kana" if seg.isascii() else "punct"))
                i += len(seg)
    return segments


def annotate_text(text: str, mode: str = "all",
                  known_readings: list[str] | None = None,
                  reveal: list[str] | None = None) -> dict:
    """Smart furigana render.

    mode: "all" | "off" | "unknown-only" (with learner's known vocabulary list)
    reveal: per-word override — surfaces listed here always show furigana
            regardless of mode (the UI's tap-to-reveal toggle).
    """
    known = set(known_readings or [])
    forced = set(reveal or [])
    segs = tokenize_japanese(text)
    unknown_words = []
    for s in segs:
        if s.kind == "kanji-word" and s.entry:
            reading = s.entry["reading"]
            s.furigana = reading
            if mode == "all":
                s.show_furigana = True
            elif mode == "unknown-only":
                s.show_furigana = reading not in known
                if not s.show_furigana:
                    if s.text in forced:
                        s.show_furigana = True
                    else:
                        continue
                else:
                    unknown_words.append(s.text)
            else:  # "off"
                s.show_furigana = False
            if s.text in forced:
                s.show_furigana = True
        elif s.kind == "unknown-kanji":
            s.furigana = "?"          # not in lexicon yet -> candidate for OCR review
            if mode != "off":
                s.show_furigana = True
            unknown_words.append(s.text)
    readable = "".join(
        f"{s.text}[{s.furigana}]" if (s.show_furigana and s.furigana) else s.text
        for s in segs)
    return {
        "mode": mode,
        "segments": [s.as_dict() for s in segs],
        "readable": readable,
        "coverage": round(sum(1 for s in segs if s.kind in ("kanji-word", "kana"))
                          / max(1, len([s for s in segs if s.kind != "punct"])), 3),
        "unknownWords": sorted(set(unknown_words)),
    }


# ---------------------------------------------------------------------------
# 2. Manga OCR contract + flashcard extraction
# ---------------------------------------------------------------------------

@dataclass
class OcrBlock:
    text: str
    bbox: tuple[int, int, int, int]      # x, y, w, h
    confidence: float


def ocr_image(image_bytes: bytes | None = None, *, mock_blocks: list[dict] | None = None):
    """Production hook: route to PaddleOCR/ML Kit. In dev, pass mock_blocks.

    Returns list[OcrBlock]; raises RuntimeError if neither backend nor mocks.
    """
    if mock_blocks is not None:
        return [OcrBlock(b["text"], tuple(b.get("bbox", (0, 0, 0, 0))),
                         b.get("confidence", 1.0)) for b in mock_blocks]
    raise RuntimeError(
        "No OCR backend available. Wire this to PaddleOCR (server) or "
        "Google ML Kit (on-device) and pass image_bytes.")


def extract_vocabulary(blocks: list[OcrBlock] | list[dict]) -> dict:
    """Turn OCR'd manga text into tap-to-add flashcard candidates."""
    lines = [b.text if isinstance(b, OcrBlock) else b["text"] for b in blocks]
    joined = "\n".join(lines)
    ann = annotate_text(joined.replace("\n", ""), mode="all")
    cards = []
    seen = set()
    for s in ann["segments"]:
        if s["kind"] == "kanji-word" and "entry" in s:
            e = s["entry"]
            if e["id"] in seen:
                continue
            seen.add(e["id"])
            cards.append({
                "wordId": e["id"], "word": s["text"], "reading": e["reading"],
                "romaji": e["romaji"], "glosses": e["glosses"],
                "pitchPattern": e["pitchPattern"], "level": e["level"],
                "source": "manga-ocr",
            })
    return {"rawText": joined, "cards": cards,
            "unmatchedKanji": ann["unknownWords"]}


# ---------------------------------------------------------------------------
# 3. Listening & Shadowing plans
# ---------------------------------------------------------------------------

def shadowing_plan(sentence_id: str) -> dict:
    """Build a shadowing session for a corpus sentence.

    Pause points come from the grammar deconstructor's clause boundaries
    (particle arcs + predicate heads); speed ladder keeps pitch constant by
    time-stretching (WSOLA) in the audio layer — metadata only here.
    """
    sentence = next((s for s in load_corpus() if s.id == sentence_id), None)
    if sentence is None:
        raise KeyError(f"no sentence '{sentence_id}'")
    dec = deconstruct(sentence)
    chunks, current = [], ""
    for t in dec["tokens"]:
        current += t["surface"]
        if t["particle"] in ("て", "で", "が", "から", "ので", "と") or \
           t["pos"] in ("verb", "adjective"):
            chunks.append({"text": current,
                           "roleHint": t.get("role"),
                           "pauseAfterMs": 350})
            current = ""
    if current:
        chunks.append({"text": current, "roleHint": None, "pauseAfterMs": 0})
    morae = sum(len(re.findall(r"[぀-ゟ゠-ヿー]", c["text"])) for c in chunks)
    return {
        "sentenceId": sentence.id,
        "text": sentence.text,
        "translation": sentence.translation,
        "level": sentence.level,
        "speedLadder": [0.75, 1.0, 1.25],
        "chunks": chunks,
        "estimatedMora": morae,
        "nativeAudioUrl": f"/audio/{sentence.id}_native.mp3",   # Azure/ElevenLabs TTS slot
        "scoring": "compare recorded F0 contour per chunk via kotoba.core.pitch.score_performance",
    }
