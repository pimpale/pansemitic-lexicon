#!/usr/bin/env python3
"""Proto-Semitic root computation for a matched Arabic↔Hebrew pair.

The root is computed *after* matching, when both words are in hand, by reducing
each side to its consonantal radicals and running the existing joint
reconstruction over just those radicals.  Doing it jointly (rather than mapping
one language's root tag onto the other) is what makes the result a real
Proto-Semitic root: the reconstruction reconciles the regular sound
correspondences — Arabic ث / Hebrew שׁ → PS *θ, Arabic ش / Hebrew שׂ → the PS
lateral, the het/ḥet and ʕayin/ġayin merges, etc. — using Arabic's more
conservative phonology to break Hebrew's mergers.

Radical sources, in order of preference:
  1. The Wiktionary root *tag* (a category like "…belonging to the root ق د م").
     Authoritative and complete even for weak (ق-و-م) and geminate (ر-ب-ب)
     roots, where the surface form hides a radical.  Hebrew tags are read from
     the *script*, which preserves distinctions (שׁ vs שׂ, ט vs ת) that the
     pronunciation merges.
  2. Failing a tag, the surface form's bare consonant skeleton (caller-supplied
     romanization).  Best-effort — affixing nominal templates can leak an affix
     consonant; verb binyan gemination is collapsed.

Unlike a correspondence-graph over root tags (which over-merges by transitive
chaining), each pair's root is computed independently from its own radicals, so
unrelated roots never fuse.
"""

from __future__ import annotations

import re

from reconstruction import (
    ArabicWord,
    Consonant,
    HebrewWord,
    PansemiticWord,
    Phoneme,
    ReconstructionError,
    reconstruct_from_words,
)

# Arabic letter → IPA radical (conservative; Arabic keeps the PS inventory).
_AR_LETTER_IPA: dict[str, str] = {
    "ء": "ʔ", "أ": "ʔ", "إ": "ʔ", "ؤ": "ʔ", "ئ": "ʔ", "آ": "ʔ",
    "ا": "ʔ", "ى": "j", "ة": "t",
    "ب": "b", "ت": "t", "ث": "θ", "ج": "d͡ʒ", "ح": "ħ", "خ": "x",
    "د": "d", "ذ": "ð", "ر": "r", "ز": "z", "س": "s", "ش": "ʃ",
    "ص": "sˤ", "ض": "dˤ", "ط": "tˤ", "ظ": "ðˤ", "ع": "ʕ", "غ": "ɣ",
    "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "و": "w", "ي": "j",
}

# Hebrew letter → IPA radical, reading the *script* for conservative values:
# tsadi → emphatic, tet → emphatic t, bare shin → *š (the sin dot reassigns it).
_HE_LETTER_IPA: dict[str, str] = {
    "א": "ʔ", "ב": "b", "ג": "g", "ד": "d", "ה": "h", "ו": "w",
    "ז": "z", "ח": "ħ", "ט": "tˤ", "י": "j", "כ": "k", "ך": "k",
    "ל": "l", "מ": "m", "ם": "m", "נ": "n", "ן": "n", "ס": "s",
    "ע": "ʕ", "פ": "p", "ף": "p", "צ": "sˤ", "ץ": "sˤ", "ק": "q",
    "ר": "r", "ש": "ʃ", "ת": "t",
}
_HE_SHIN_DOT = "ׁ"  # שׁ → keep *š (ʃ)
_HE_SIN_DOT = "ׂ"   # שׂ → reassign to *ś (→ s here; reconstruction laterals it)

_VOWELS_AND_MARKS = re.compile(r"[aeiou]")


def arabic_root_radicals(tag: str) -> list[str]:
    """Radicals (IPA) of an Arabic root tag, e.g. 'قدم' → ['q','d','m']."""
    return [_AR_LETTER_IPA[c] for c in tag if c in _AR_LETTER_IPA]


def hebrew_root_radicals(tag: str) -> list[str]:
    """Radicals (IPA) of a Hebrew root tag, reading the shin/sin dot."""
    out: list[str] = []
    for ch in tag:
        if ch == _HE_SIN_DOT:
            if out:
                out[-1] = "s"
        elif ch == _HE_SHIN_DOT:
            continue
        elif ch in _HE_LETTER_IPA:
            out.append(_HE_LETTER_IPA[ch])
    return out


def _consonant_skeleton(ipa: str) -> list[str]:
    """Consonant tokens of an IPA string, length/stress marks dropped and
    adjacent duplicates collapsed — so a geminate (binyan D, batˤːala) yields a
    single radical (tˤ), matching the un-geminated root-tag radicals."""
    out: list[str] = []
    for tok in Phoneme.parse(ipa):
        if isinstance(tok, Consonant):
            radical = tok.tok.replace("ː", "").replace("ˈ", "")
            if not out or out[-1] != radical:
                out.append(radical)
    return out


def stem_radicals(stem_ipa: str) -> list[str]:
    """Radicals of an already-affix-stripped bare-stem IPA (e.g. a merge-trace
    stem), reusing the morphology layer-stripping the reconstruction did."""
    return _consonant_skeleton(stem_ipa)


def surface_radicals(word: "object", romanization: str) -> list[str]:
    """Best-effort radicals from a surface romanization (no root tag): its bare
    consonant skeleton.  *word* is the language's Word class (ArabicWord /
    HebrewWord) for the romanization→IPA conversion."""
    if not romanization:
        return []
    try:
        ipa = word.from_romanization(romanization).word  # type: ignore[attr-defined]
    except ReconstructionError:
        return []
    return _consonant_skeleton(ipa)


def proto_root(
    ar_radicals: list[str], he_radicals: list[str],
) -> tuple[str, str] | None:
    """Joint PS reconstruction of two radical lists → (ipa_key, latin_label).

    ``ipa_key`` is the canonical grouping key (consonant skeleton of the
    reconstruction, in IPA); ``latin_label`` is the readable pansemitic-scholar
    rendering.  None when either side has no radicals or reconstruction fails."""
    if not ar_radicals or not he_radicals:
        return None
    try:
        merged = reconstruct_from_words(
            ArabicWord.from_ipa("a".join(ar_radicals)),
            HebrewWord.from_ipa("a".join(he_radicals)),
        )
    except ReconstructionError:
        return None
    key = "".join(_consonant_skeleton(merged.word))
    if not key:
        return None
    label = _VOWELS_AND_MARKS.sub(
        "", PansemiticWord.from_word(merged).to_protosemitic_convention())
    return key, label
