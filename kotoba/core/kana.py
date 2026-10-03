"""Kana utilities: mora segmentation, dakoron normalisation, romaji.

Japanese pitch accent is defined over *morae*, not syllables, so getting
segmentation right is the single most important prerequisite for the pitch
engine.  Rules implemented here follow standard phonotactics:

*拗音 (ゃ/ゅ/ょ) fuse with the preceding kana -> one mora  (きょう = kyoo = 2 morae)
*長音符 ー extends the vowel                -> one mora   (こうこう = 4 morae)
*促音 っ / ッ is its own mora               (きっぷ = ki-p-pu = 3 morae)
*撥音 ん is its own mora                    (にほん = ni-ho-n = 3 morae)
"""
from __future__ import annotations

import unicodedata

_SMALL = set("ゃゅょャュョ")
_LONG = set("ー〜")
_GEM_OR_N = set("っッんン")

# Hiragana <-> katakana offset (0x60)
def to_hiragana(kana: str) -> str:
    out = []
    for ch in kana:
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6:      # katakana range
            out.append(chr(o - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def to_katakana(kana: str) -> str:
    out = []
    for ch in kana:
        o = ord(ch)
        if 0x3041 <= o <= 0x3096:      # hiragana range
            out.append(chr(o + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def strip_diacritics_kana(kana: str) -> str:
    """Normalise dakoron/handakuten forms to their base kana.

    が -> か, ぱ -> は, etc.  Used for dictionary lookups where the pitch
    dictionary stores unvoiced bases alongside voiced entries.
    """
    decomposed = unicodedata.normalize("NFD", kana)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped)


def segment_morae(kana: str) -> list[str]:
    """Split a kana string into morae.

    >>> segment_morae("がっこう")
    ['が', 'っ', 'こ', 'う']
    >>> segment_morae("きょう")
    ['きょ', 'う']
    >>> segment_morae("とうきょう")
    ['と', 'う', 'きょ', 'う']
    """
    kana = to_hiragana(kana)
    morae: list[str] = []
    i, n = 0, len(kana)
    while i < n:
        ch = kana[i]
        if ch in _GEM_OR_N or ch in _LONG:
            morae.append(ch)
            i += 1
            continue
        nxt = kana[i + 1] if i + 1 < n else ""
        if nxt in _SMALL:                      # digraph mora (拗音)
            morae.append(ch + nxt)
            i += 2
        else:
            morae.append(ch)
            i += 1
    return morae


def count_morae(kana: str) -> int:
    return len(segment_morae(kana))


_HIRA_VOWELS = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "wo", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo",
    "っ": "", "ー": "-", "、": ", ", "。": ". ",
}

_KOMBI = {
    "き": "ky", "し": "sh", "ち": "ch", "に": "ny", "ひ": "hy",
    "み": "my", "り": "ry", "ぎ": "gy", "び": "by", "ぴ": "py",
    "じ": "j",
}


def mora_to_romaji(mora: str) -> str:
    if len(mora) == 2 and mora[1] in _SMALL and mora[0] in _KOMBI:
        return _KOMBI[mora[0]] + _HIRA_VOWELS[mora[1]]
    if len(mora) == 2 and mora[1] in _SMALL:
        return _HIRA_VOWELS.get(mora[0], mora[0]) + _HIRA_VOWELS[mora[1]][1:]
    return _HIRA_VOWELS.get(mora, mora)


def kana_to_romaji(kana: str) -> str:
    return "".join(mora_to_romaji(m) for m in segment_morae(kana))
