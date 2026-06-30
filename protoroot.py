#!/usr/bin/env python3
"""Root-radical letter maps for proto-root reconstruction.

The actual root analysis (radical gathering, de-patterning, and the joint
Proto-Semitic reconstruction) lives in the morphology layer (morphology.py),
which owns a SINGLE root computation shared by the merge path and the
shared-source path.  This module is the radical primitives that analyzer reuses:
the conservative Arabic/Hebrew letter → IPA radical maps (so a root *tag* — the
authority, complete even for weak ق-و-م and geminate ر-ب-ب roots — yields its
radicals directly), plus the consonant-skeleton helper.

Hebrew tags are read from the *script*, which preserves distinctions (שׁ vs שׂ,
ט vs ת) that the pronunciation merges; the shin/sin dot is honoured so a bare
shin keeps *š and a sin dot reassigns to *ś.
"""

from __future__ import annotations

from reconstruction import Consonant, Phoneme

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


def consonant_skeleton(ipa: str) -> list[str]:
    """Consonant tokens of an IPA string, length/stress marks dropped and
    *adjacent* duplicates collapsed — so a surface geminate (binyan D, batˤːala)
    yields a single radical (tˤ), matching the un-geminated root-tag radicals,
    while a true geminate ROOT whose two like radicals are separated by a vowel
    (the tag-joined rabab → r-b-b) keeps both.  A vowel breaks adjacency."""
    out: list[str] = []
    prev: str | None = None
    for tok in Phoneme.parse(ipa):
        if isinstance(tok, Consonant):
            radical = tok.tok.replace("ː", "").replace("ˈ", "")
            if radical != prev:
                out.append(radical)
            prev = radical
        else:
            prev = None  # an intervening vowel breaks consonant adjacency
    return out
