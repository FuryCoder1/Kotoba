"""Bundled seed datasets for Kotoba.

All linguistic data ships as JSON so the engines stay pure-python and testable.
Sources are hand-curated subsets modelled on:
  * JMDict (v2) EN-JL dictionary  -> lexicon.json  (fields renamed to camelCase)
  * NHK / JDBN pitch-accent DB    -> pitch.json
  * KANJIDIC2 + Heisig-style mnemonics -> kanji.json
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def _load(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def load_lexicon() -> list[dict]:
    """Word entries with reading, glosses, part-of-speech and pitch number."""
    return _load("lexicon.json")


def load_pitch_entries() -> list[dict]:
    """Raw pitch-accent entries: {kana, moraCount, accentLocation, pattern}."""
    return _load("pitch.json")


def load_kanji() -> list[dict]:
    """Kanji with strokes, on/kun readings, radicals and mnemonic story."""
    return _load("kanji.json")


def load_radicals() -> dict[str, dict]:
    """Radical id -> {name, meaning, hint} used to compose kanji mnemonics."""
    return _load("radicals.json")


def load_sentences() -> list[dict]:
    """Graded example sentences used by the reading engine & deconstructor."""
    return _load("sentences.json")


def load_scenarios() -> list[dict]:
    """Roleplay scenarios (izakaya, station, ...) with keigo rubrics."""
    return _load("scenarios.json")
