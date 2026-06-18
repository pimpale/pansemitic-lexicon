"""Morphological analysis & merge-time promotion for Arabic/Hebrew surfaces.

Wraps the surface forms of a cognate pair with just enough morphological
structure that reconstruction compares like with like:

  - Compounds are split on whitespace and merged word-by-word, so the
    aligner never smears one word's segments into its neighbour.
  - Definite articles (ar al-/aC-, he ha-) are functional morphemes, never
    cognate material; this module owns their stripping (ArabicWord no longer
    strips them during IPA conversion).  Arabic articles are stripped
    wherever detected (script ال + hyphenated romanization is unambiguous);
    Hebrew articles are stripped in multi-word phrases, and on single words
    only when the Arabic side is also definite — the pair-level symmetry is
    the corroborating evidence that ha- is an article rather than a
    word-initial pattern like hifʕil-derived הַצָּלָה.  When BOTH sides are
    definite, definiteness is preserved in the merged ancestor as the
    space-separated compromise particle "hal" (al-/ha- blend).
  - A feminine ending (ar tāʔ marbūṭa, he qamats-he) present on only ONE side
    is stripped; present on both sides the stems are stripped to a clean common
    stem and the compromise -a is re-attached (shared morphology is always
    underived to the deepest lossless layer, then rebuilt, rather than left for
    the aligner to reconcile).
  - A nisba adjectivizer (ar -iyy, he -i; adjective POS required) present
    on one side is stripped (de-adjectivization); present on both sides,
    the stems are merged and the suffix re-attached as the compromise -i.
  - Verb forms are normalized to Proto-Semitic stem notation (G, D, Š, N, L,
    tD, …; see each LangMorphology.verb_stem_map), and deverbal lexemes carry a
    templatic category (active/passive participle, verbal noun, instance noun;
    see Derivation).  When both sides share a deverbal category it is preserved
    — the surface is the deepest lossless layer, since underiving to the
    vowelless root would drop the template's vowel melody — and the shared stem
    + category are recorded on the MergeResult.  An asymmetric verb/deverbal form
    is reduced toward its G-stem base: the cited base lexeme kaikki links
    (form_of), falling back to per-language stem synthesis (D: degeminate C2;
    Š: strip the ʔa-/hi- causative prefix).

Detection is evidence-gated: every strip needs BOTH the script-side signal
(pointing/letters) and a matching romanization shape, and must leave at
least two letters behind, otherwise the word passes through untouched.

Language knowledge lives in one LangMorphology subclass per language,
mirroring reconstruction.py's one-Word-class-per-language pattern.  To
extend coverage (e.g. Aramaic), subclass LangMorphology, override the
script-evidence hooks / strip patterns / synthesis methods, and register
the class in MORPHOLOGY_CONFIG.

`merge` aligns the (ar_roman, he_roman) word pairs, reconstructs each pair
into its proto-Semitic ancestor (and pansemitic reduction) word-by-word, and
returns a MergeResult plus human-readable notes describing exactly what was
normalized; the notes are surfaced in the output so pansemitic forms stay
auditable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, ClassVar, Iterable

from loss import Consonant, Phoneme
from reconstruction import (
    ArabicWord,
    HebrewWord,
    PansemiticWord,
    ReconstructedSemProWord,
    Word,
    reconstruct_from_words,
)


class Layer(Enum):
    """Strippable surface morphology detectable from script + romanization.

    The first group is concatenative nominal morphology; the GEMINATION/
    *_PREFIX group are the verbal-derivation exponents (decomposed so a
    Semiticist stem is a *set* of layers: D = {GEMINATION}, N = {N_PREFIX},
    Š = {CAUSATIVE}, tD = {T_PREFIX, GEMINATION}).  Verb-stem layers are
    stripped to the G base for the merge and re-applied on the pansemitic
    side, exactly like the nominal layers."""
    DEFINITE = "definite"
    FEMININE = "feminine"
    NISBA = "nisba"
    DUAL = "dual"
    PLURAL = "plural"
    # Verbal-derivation exponents (re-applied as pansemitic templates).
    GEMINATION = "gemination"   # D: doubled C2
    N_PREFIX = "n-prefix"       # N: na-
    CAUSATIVE = "causative"     # Š: ša-
    T_PREFIX = "t-prefix"       # t-stems: ta-


# Proto-Semitic stem → the set of verbal-derivation layers it decomposes into.
# G carries none; only the common active stems are templatized (see the
# pansemitic-morphology-conventions memory).  A stem absent here stays raw.
_STEM_LAYERS: dict[str, frozenset[Layer]] = {
    "G": frozenset(),
    "D": frozenset({Layer.GEMINATION}),
    "N": frozenset({Layer.N_PREFIX}),
    "Š": frozenset({Layer.CAUSATIVE}),
    "tD": frozenset({Layer.T_PREFIX, Layer.GEMINATION}),
}


class Derivation(Enum):
    """Templatic deverbal category of a lexeme (non-concatenative, so detected
    from kaikki metadata rather than surface stripping).  A plain finite verb
    carries none of these."""
    ACTIVE_PARTICIPLE = "active-participle"
    PASSIVE_PARTICIPLE = "passive-participle"
    VERBAL_NOUN = "verbal-noun"
    INSTANCE_NOUN = "instance-noun"


# Templatic categories whose vowel melody is data: a shared one is preserved by
# merging surfaces (the deepest lossless layer), never reduced to a root.
_DEVERBAL_PRIORITY = (
    Derivation.ACTIVE_PARTICIPLE,
    Derivation.PASSIVE_PARTICIPLE,
    Derivation.VERBAL_NOUN,
    Derivation.INSTANCE_NOUN,
)


@dataclass
class AnalyzedWord:
    """One orthographic word with its detected strippable layers.

    ``ipa`` is the word's phoneme string (from word_from_sharedsource), the
    representation the merge strips and reconstructs in; ``roman``/``script``
    are kept for layer detection and cited-base lookup."""
    script: str
    roman: str
    ipa: str = ""
    layers: set[Layer] = field(default_factory=set)


@dataclass
class AnalyzedPhrase:
    """A headword split into words, plus lexeme-level kaikki metadata."""
    lang: str
    roman: str                       # full original romanization
    words: list[AnalyzedWord]
    ipa: str = ""                    # full phrase IPA (word_from_sharedsource)
    pos: frozenset[str] = frozenset()
    verb_forms: frozenset[str] = frozenset()   # ar form (I..X) / he binyan (raw)
    verb_stem: frozenset[str] = frozenset()    # Proto-Semitic stem(s): G, D, Š, …
    derivation: frozenset[Derivation] = frozenset()  # templatic deverbal categories
    number: frozenset[str] = frozenset()       # ⊆ {"p", "d"} (plural/dual lemma)
    gender: frozenset[str] = frozenset()       # ⊆ {"m", "f"}
    derived_from: frozenset[str] = frozenset() # normalized derivational bases
    singular_of: frozenset[str] = frozenset()  # normalized cited singular(s) of a
                                               # plural/dual lemma
    masculine_of: frozenset[str] = frozenset() # normalized cited masculine base(s)
                                               # of a derived feminine lemma


@dataclass
class PlannedPair:
    """One aligned word pair, morphology-normalized, ready to merge.

    ``layers`` are the shared morphemes stripped from both sides for a clean
    stem merge.  They are re-attached on the *pansemitic* side — after the
    merged stem is reduced to the pansemitic inventory — via
    PansemiticMorphology.produce_word, so the compromise affixes (hal-, -i,
    -im/-at, -a) live in pansemitic phonology rather than being glued onto the
    proto-Semitic ancestor.

    ar_ipa/he_ipa are the morphology-stripped stems in phoneme space, ready to
    align and reconstruct."""
    ar_ipa: str
    he_ipa: str
    layers: set[Layer] = field(default_factory=set)


@dataclass
class MergeResult:
    """The reconstructed ancestor of a merged cognate pair.

    ``ancestor`` is the bare merged proto-Semitic stem; ``pansemitic`` is that
    stem reduced to the pansemitic inventory with shared morphology re-applied.
    verb_stem/derivation carry the ancestor's verbal analysis when both sides
    share a deverbal category (preserved, not reduced); None otherwise."""
    ancestor: Word
    pansemitic: PansemiticWord
    notes: list[str]
    verb_stem: str | None = None
    derivation: str | None = None


# Looks up the (canonical, romanization) of a base lexeme in the caller's
# word index, given (lang, normalized base candidates, preferred POS set —
# so a nominal base beats a verb homograph and vice-versa).
BaseLookup = Callable[[str, frozenset[str], frozenset[str]], tuple[str, str] | None]

# POS the singular/masculine reductions expect their base to be (a noun-like
# lexeme), used to outrank verb homographs sharing the consonantal skeleton.
_NOMINAL_BASE_POS = frozenset({"noun", "adj", "name", "num"})
_VERB_BASE_POS = frozenset({"verb"})


def _letters(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


# Gemination in IPA is ː after a consonant; vowel length (aː, iː …) uses the
# same mark, so only strip a ː that does NOT follow a base vowel.
_GEMINATE_IPA = re.compile(r"(?<![aeiou])ː")


def _degeminate_ipa(ipa: str) -> str | None:
    """Remove the first consonant gemination (Cː → C); None if none present."""
    out = _GEMINATE_IPA.sub("", ipa, count=1)
    return out if out != ipa else None


# Modifier letters that can't stand alone after a prefix/article strip lops off
# the consonant or vowel they attached to (length ː, pharyngealization ˤ, the
# tie bar, etc.).  Phoneme.parse rejects them as bare tokens.
_LEADING_ORPHAN = re.compile(r"^[ːˤʰʲʷˠ͡]+")


def _clean_leading(ipa: str) -> str:
    """Drop orphan modifier letters left at the start of a stripped stem."""
    return _LEADING_ORPHAN.sub("", ipa)


# IPA word boundaries: whitespace and the undertie ‿ (liaison in native IPA,
# e.g. ʔin ʃaːʔa‿lˤːaːh).
_IPA_WORD_SPLIT = re.compile(r"[\s‿]+")


class LangMorphology:
    """Per-language morphological knowledge.

    One subclass per language, registered in MORPHOLOGY_CONFIG.  The base
    class implements the generic detect/strip machinery; subclasses supply
    the script-evidence hooks, the romanization strip shapes, and (where
    the language has them) template-level de-causativization and the
    article shape in already-converted IPA."""

    lang: ClassVar[str]
    # Romanization-side strip shape per layer, used for *detection*
    # corroboration in analyze_word; a missing entry means the language does
    # not support that layer.
    strip_patterns: ClassVar[dict[Layer, re.Pattern[str]]] = {}
    # IPA-side strip shape per layer, used by strip_ipa to actually remove the
    # exponent during the merge (the merge operates in phoneme space).  The
    # definite article is handled separately by strip_article_ipa.
    strip_patterns_ipa: ClassVar[dict[Layer, re.Pattern[str]]] = {}
    # Verb forms regarded as the underived base stem, used to rank homograph
    # candidates during base substitution (ar "I", he "pa").
    base_verb_forms: ClassVar[frozenset[str]] = frozenset()
    # Maps a language's raw verb-form label (Arabic Roman numeral, Hebrew
    # binyan) to the Proto-Semitic stem notation (G, D, Š, N, …); a missing
    # entry leaves the form unnormalized (uncommon stems pass through raw).
    verb_stem_map: ClassVar[dict[str, str]] = {}
    # Whether a single-word article strip needs the other side of the pair
    # to also be definite (script evidence alone too weak — Hebrew).
    article_needs_corroboration: ClassVar[bool] = False

    # Inverse of strip_patterns: the surface shape each layer re-attaches as
    # when *producing* a word back from its analyzed stem.  Prefixal layers
    # (the definite article) live in produce_prefixes, suffixal ones in
    # produce_suffixes; a missing entry means the layer leaves no surface
    # mark on this side.
    produce_prefixes: ClassVar[dict[Layer, str]] = {}
    produce_suffixes: ClassVar[dict[Layer, str]] = {}
    # Order suffixal layers stack onto the stem (innermost first).
    _PRODUCE_SUFFIX_ORDER: ClassVar[tuple[Layer, ...]] = (
        Layer.FEMININE, Layer.NISBA, Layer.DUAL, Layer.PLURAL)

    # ── script-side evidence hooks ──────────────────────────────────
    @classmethod
    def script_definite(cls, script: str) -> bool:
        return False

    @classmethod
    def script_feminine(cls, script: str) -> bool:
        return False

    @classmethod
    def script_nisba(cls, script: str) -> bool:
        return False

    @classmethod
    def script_dual(cls, script: str) -> bool:
        return False

    @classmethod
    def script_plural(cls, script: str) -> bool:
        return False

    # ── tokenization ────────────────────────────────────────────────
    @classmethod
    def script_tokens(cls, script: str) -> list[str]:
        return script.split()

    @classmethod
    def roman_tokens(cls, roman: str) -> list[str]:
        return roman.split()

    @classmethod
    def ipa_tokens(cls, ipa: str) -> list[str]:
        """IPA word tokens — split on whitespace and the liaison undertie ‿."""
        return [t for t in _IPA_WORD_SPLIT.split(ipa) if t]

    @classmethod
    def tokenize(cls, script: str, roman: str, ipa: str) -> list[tuple[str, str, str]]:
        """Aligned (script, roman, ipa) word tokens.

        All three must tokenize to the same word count to split; otherwise the
        phrase is kept whole (so a failed alignment degrades to whole-string
        merging, never to misaligned words)."""
        s_toks = cls.script_tokens(script)
        r_toks = cls.roman_tokens(roman)
        i_toks = cls.ipa_tokens(ipa)
        if (len(s_toks) == len(r_toks) == len(i_toks) and len(s_toks) > 1):
            return list(zip(s_toks, r_toks, i_toks))
        return [(script, roman, ipa)]

    # ── generic machinery ───────────────────────────────────────────
    @classmethod
    def _roman_matches(cls, layer: Layer, roman: str) -> bool:
        pattern = cls.strip_patterns.get(layer)
        return bool(pattern and pattern.search(roman))

    @classmethod
    def analyze_word(
        cls,
        script: str,
        roman: str,
        ipa: str,
        pos: frozenset[str],
        number: frozenset[str] = frozenset(),
    ) -> AnalyzedWord:
        """Detect strippable layers; each needs script AND romanization
        evidence.  Nisba additionally needs the adjective POS gate: nouns
        ending in -iyy (nabiyy, kursiyy …) carry a root consonant, not the
        adjectivizer.  Dual/plural additionally need the kaikki number
        metadata gate (singulars like תָּמִים / אָחוֹת share the surface
        shapes); within marked lemmas the surface shape disambiguates dual
        vs plural — kaikki's g=m-p on מַיִם notwithstanding, its ־ַיִם
        ending is the dual template."""
        layers: set[Layer] = set()
        if cls.script_definite(script) and cls._roman_matches(Layer.DEFINITE, roman):
            layers.add(Layer.DEFINITE)
        if cls.script_feminine(script) and cls._roman_matches(Layer.FEMININE, roman):
            layers.add(Layer.FEMININE)
        if ("adj" in pos and cls.script_nisba(script)
                and cls._roman_matches(Layer.NISBA, roman)):
            layers.add(Layer.NISBA)
        if number & {"p", "d"}:
            if cls.script_dual(script) and cls._roman_matches(Layer.DUAL, roman):
                layers.add(Layer.DUAL)
            elif cls.script_plural(script) and cls._roman_matches(Layer.PLURAL, roman):
                layers.add(Layer.PLURAL)
        return AnalyzedWord(script=script, roman=roman, ipa=ipa, layers=layers)

    @classmethod
    def strip(cls, roman: str, layer: Layer) -> str | None:
        """Strip *layer*'s romanization shape; None if absent or too destructive."""
        pattern = cls.strip_patterns.get(layer)
        if pattern is None:
            return None
        out = pattern.sub("", roman, count=1)
        if out == roman or _letters(out) < 2:
            return None
        return out

    @classmethod
    def strip_ipa(cls, ipa: str, layer: Layer) -> str | None:
        """Strip *layer*'s IPA exponent; None if absent or too destructive.

        The IPA analogue of strip(); the article is handled by
        strip_article_ipa, not here."""
        pattern = cls.strip_patterns_ipa.get(layer)
        if pattern is None:
            return None
        out = pattern.sub("", ipa, count=1)
        if out == ipa or _letters(out) < 2:
            return None
        return out

    @classmethod
    def produce_word(cls, word: AnalyzedWord) -> str:
        """Re-attach *word*'s layers onto its romanized stem — the structural
        inverse of analyze_word.

        Prefixal layers (the definite article) precede the stem; suffixal
        layers stack after it in _PRODUCE_SUFFIX_ORDER.  Producing with an
        empty stem yields the bare affix string for a single layer, which is
        how merge sources its compromise affixes (see
        PansemiticMorphology)."""
        suffix = "".join(
            cls.produce_suffixes[layer]
            for layer in cls._PRODUCE_SUFFIX_ORDER
            if layer in word.layers and layer in cls.produce_suffixes
        )
        prefix = (cls.produce_prefixes.get(Layer.DEFINITE, "")
                  if Layer.DEFINITE in word.layers else "")
        return prefix + word.roman + suffix

    @classmethod
    def verb_stem(cls, raw: str) -> str | None:
        """Normalize a raw verb-form label to Proto-Semitic stem notation."""
        return cls.verb_stem_map.get(raw)

    # ── language-specific operations (override where applicable) ────
    @classmethod
    def synthesize_base_stem(cls, ipa: str, stem: str) -> tuple[str, str] | None:
        """Reduce a derived-stem verb IPA toward the G (base) stem.

        Returns (new_ipa, note) or None when the language has no usable
        template for *stem* (or the IPA doesn't fit one).  Used as the fallback
        when no cited base lexeme is available."""
        return None

    @classmethod
    def strip_article_ipa(cls, ipa: str, script: str) -> str | None:
        """Strip a leading definite article from an already-converted IPA
        string (used for shared-source ancestors, which never pass through
        merge).  None when unsupported or unevidenced."""
        return None


class ArabicMorphology(LangMorphology):
    lang = "ar"
    strip_patterns = {
        # kaikki Arabic romanizations always hyphenate the article (al-, aš- …).
        Layer.DEFINITE: re.compile(r"^[aā](?:sh|š|ṣ|ḍ|ṭ|ẓ|ḏ|ṯ|[ltdsznr])-"),
        Layer.FEMININE: re.compile(r"(?:āh|ah|at|a)$"),
        Layer.NISBA: re.compile(r"(?:iyy|īy|ī)$"),
        Layer.DUAL: re.compile(r"(?:āni|ayni|ān|ayn)$"),
        Layer.PLURAL: re.compile(r"(?:āt|ūna|īna|ūn|īn)$"),
    }
    # IPA exponents stripped during the merge (detection already confirmed the
    # layer from the script).  The article is handled by strip_article_ipa.
    strip_patterns_ipa = {
        Layer.FEMININE: re.compile(r"(?:at|a)$"),       # tāʔ marbūṭa → /a/
        Layer.NISBA: re.compile(r"(?:ijː|iːj|iː|i)$"),  # nisba /ijj/
        Layer.DUAL: re.compile(r"(?:aːni|ajni|aːn|ajn)$"),
        Layer.PLURAL: re.compile(r"(?:aːt|uːna|iːna|uːn|iːn)$"),
    }
    # Inverse shapes for produce_word: a representative surface realization of
    # each layer (the article unassimilated, the sound-masculine endings).
    produce_prefixes = {Layer.DEFINITE: "al-"}
    produce_suffixes = {
        Layer.FEMININE: "a",
        Layer.NISBA: "iyy",
        Layer.DUAL: "ān",
        Layer.PLURAL: "ūn",
    }
    base_verb_forms = frozenset({"I"})
    # Roman numeral form → Proto-Semitic stem.  Common triliteral forms only;
    # IX (rare) and quadriliterals (Iq, IIq …) are left raw.
    verb_stem_map = {
        "I": "G", "II": "D", "III": "L", "IV": "Š", "V": "tD",
        "VI": "tL", "VII": "N", "VIII": "tG", "X": "Št",
    }

    # Doubled consonant in a romanization (form-II/D gemination); long vowels
    # are single precomposed codepoints (ā, ī …) so excluding plain vowels
    # suffices.
    _DOUBLED = re.compile(r"([^\W\d_aeiou])\1")
    _STEM_PREFIX = re.compile(r"^[ʔʾˀ]?a")  # form IV/Š ʔa-
    _N_PREFIX = re.compile(r"^[ʔʾˀ]?i?n")   # form VII/N (i)n-
    _T_PREFIX = re.compile(r"^t[aā]")        # form V/tD ta-

    # Article shapes in already-converted IPA.  Word.from_ipa strips syllable
    # dots and stress marks, so by Word time the article shows up as either a
    # hyphen-delimited prefix (ar-raħmaːn, romanization-built words), an
    # assimilated geminate (ʔarːaħmaːn — the consonant is kept), or a bare
    # ʔ?al prefix (ʔalqurʔaːn).  The bare/geminate shapes are ambiguous
    # against root material, so they demand script-side evidence.
    _ARTICLE_IPA_DELIM = re.compile(r"^ʔ?a(?:sˤ|tˤ|dˤ|ðˤ|[tθdðrzsʃln])[.\-]")
    _ARTICLE_IPA_GATED: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r"^ʔ?a(sˤ|tˤ|dˤ|ðˤ|[tθdðrzsʃln])ː"), r"\1"),
        (re.compile(r"^ʔ?al(?!ː)"), ""),
    ]

    @classmethod
    def script_definite(cls, script: str) -> bool:
        return ArabicWord.normalize(script).startswith("ال")

    @classmethod
    def script_feminine(cls, script: str) -> bool:
        return ArabicWord.normalize(script).endswith("ة")

    @classmethod
    def script_nisba(cls, script: str) -> bool:
        return ArabicWord.normalize(script).endswith("ي")

    @classmethod
    def script_dual(cls, script: str) -> bool:
        return ArabicWord.normalize(script).endswith("ان")

    @classmethod
    def script_plural(cls, script: str) -> bool:
        # Sound plurals only; broken plurals have no suffix to detect and
        # are reachable solely via their plural-of form_of link.
        return ArabicWord.normalize(script).endswith(("ات", "ون", "ين"))

    @classmethod
    def synthesize_base_stem(cls, ipa: str, stem: str) -> tuple[str, str] | None:
        """Reduce a derived-stem verb IPA toward the G base."""
        if stem == "D":  # faʕʕala → degeminate C2
            out = _degeminate_ipa(ipa)
            if out:
                return out, "D-stem (form II) degeminated"
        if stem == "Š":  # ʔafʕala → strip causative ʔa-
            m = cls._STEM_PREFIX.match(ipa)
            if m and _letters(ipa[m.end():]) >= 3:
                return _clean_leading(ipa[m.end():]), "Š-stem (form IV) ʔa- prefix stripped"
        if stem == "N":  # infaʕala → strip n- prefix
            m = cls._N_PREFIX.match(ipa)
            if m and _letters(ipa[m.end():]) >= 3:
                return _clean_leading(ipa[m.end():]), "N-stem (form VII) n- prefix stripped"
        if stem == "tD":  # tafaʕʕala → strip ta- then degeminate
            m = cls._T_PREFIX.match(ipa)
            if m and _letters(ipa[m.end():]) >= 3:
                base = _clean_leading(ipa[m.end():])
                return (_degeminate_ipa(base) or base,
                        "tD-stem (form V) ta- prefix stripped, degeminated")
        return None

    @classmethod
    def strip_article_ipa(cls, ipa: str, script: str) -> str | None:
        """*script* is the lexeme's cited form — either Arabic script or a
        romanized citation.  The hyphen-delimited IPA shape (ʔal-d͡ʒabr) is
        self-evident — root material never contains a hyphen — and may come
        from a citing template's tr even when *script* lacks the article.
        The geminate/bare shapes additionally require the cited form to
        start with ال (or carry the hyphenated article in a Latin citation).
        Returns None when no article is evidenced or stripping would empty
        the string."""
        out = cls._ARTICLE_IPA_DELIM.sub("", ipa, count=1)
        if out != ipa:
            return _clean_leading(out) or None
        if not (cls.script_definite(script)
                or cls.strip_patterns[Layer.DEFINITE].match(script.lower())):
            return None
        for pattern, repl in cls._ARTICLE_IPA_GATED:
            out = pattern.sub(repl, ipa, count=1)
            if out != ipa:
                return _clean_leading(out) or None
        return None


class HebrewMorphology(LangMorphology):
    lang = "he"
    strip_patterns = {
        # Hebrew romanizations fuse the article (hamolád), so the pattern
        # eats exactly the h + vowel.
        Layer.DEFINITE: re.compile(r"^h[ae]-?"),
        Layer.FEMININE: re.compile(r"[āáa]$"),
        Layer.NISBA: re.compile(r"[íi]$"),
        Layer.DUAL: re.compile(r"[áa]yim$"),
        Layer.PLURAL: re.compile(r"(?:[íi]m|[óo]t)$"),
    }
    # IPA exponents stripped during the merge (Modern Hebrew IPA, so no
    # gemination); the article is handled by strip_article_ipa.
    strip_patterns_ipa = {
        Layer.FEMININE: re.compile(r"a$"),
        Layer.NISBA: re.compile(r"i$"),
        Layer.DUAL: re.compile(r"ajim$"),
        Layer.PLURAL: re.compile(r"(?:im|ot)$"),
    }
    # Inverse shapes for produce_word: ha- article, the masculine -im plural
    # and -áyim dual; feminine/nisba take the qamats-he / hiriq-yod endings.
    produce_prefixes = {Layer.DEFINITE: "ha"}
    produce_suffixes = {
        Layer.FEMININE: "á",
        Layer.NISBA: "i",
        Layer.DUAL: "áyim",
        Layer.PLURAL: "im",
    }
    base_verb_forms = frozenset({"pa"})
    # Binyan → Proto-Semitic stem.  Common binyanim only; rare ones (poal,
    # hitpuʕal, nitpaʕel) are left raw.
    verb_stem_map = {
        "pa": "G", "qal": "G", "nif": "N", "ni": "N", "pi": "D", "pu": "Dp",
        "hif": "Š", "hi": "Š", "huf": "Šp", "ho": "Šp", "hit": "tD",
    }
    # A he/ha-initial word is weak evidence on its own (hifʕil-derived nouns
    # like הַצָּלָה share the shape) — single-word strips need the other
    # side of the pair to also be definite.
    article_needs_corroboration = True

    _DAGESH = "ּ"
    _PATACH = "ַ"
    _QAMATS = "ָ"
    _SEGOL = "ֶ"
    _GUTTURALS = frozenset("אהחער")
    _FEMININE_END = _QAMATS + "ה"
    _NISBA_END = "ִי"  # hiriq + yod
    _MAQAF = "־"
    # A hyphen splits a romanization only when every fragment keeps more
    # than this many letters — protects particles and citation forms
    # (al-, tel-avív) from being torn apart.
    _MIN_HYPHEN_SPLIT_LETTERS = 4

    @classmethod
    def script_tokens(cls, script: str) -> list[str]:
        """Hebrew compounds join words with maqaf (בֵּית־הַמִּקְדָּשׁ) as
        often as with spaces; treat both as word boundaries."""
        return [tok for chunk in script.split()
                for tok in chunk.split(cls._MAQAF) if tok]

    @classmethod
    def roman_tokens(cls, roman: str) -> list[str]:
        out: list[str] = []
        for chunk in roman.split():
            frags = chunk.split("-")
            if len(frags) > 1 and all(
                    _letters(f) >= cls._MIN_HYPHEN_SPLIT_LETTERS for f in frags):
                out.extend(frags)
            else:
                out.append(chunk)
        return out

    @classmethod
    def script_definite(cls, script: str) -> bool:
        """Pointed-script test for the definite article: הַ + dagesh forte in
        the next letter, or הָ/הֶ before a guttural (which cannot take dagesh)."""
        if len(script) < 4 or script[0] != "ה":
            return False
        vowel = script[1]
        if vowel == cls._PATACH:
            i = 3
            while i < len(script) and unicodedata.category(script[i]) == "Mn":
                if script[i] == cls._DAGESH:
                    return True
                i += 1
            return False
        if vowel in (cls._QAMATS, cls._SEGOL):
            return script[2] in cls._GUTTURALS
        return False

    @classmethod
    def _suffix_form(cls, script: str) -> str:
        """Prepare pointed script for suffix checks: NFC fixes the
        free-order placement of combining marks (hiriq+dagesh vs
        dagesh+hiriq on the same letter), and dagesh is dropped entirely
        so gemination dots (חַיִּים, רַבָּה) can't break endswith tests."""
        return unicodedata.normalize("NFC", script).replace(cls._DAGESH, "")

    @classmethod
    def script_feminine(cls, script: str) -> bool:
        return cls._suffix_form(script).endswith(cls._FEMININE_END)

    @classmethod
    def script_nisba(cls, script: str) -> bool:
        return cls._suffix_form(script).endswith(cls._NISBA_END)

    # Dual ־ַיִם (patach-yod-hiriq-mem) vs plural ־ִים (hiriq-yod-mem):
    # the pointing keeps them distinct even though both romanize to …im.
    _DUAL_END = "ַיִם"
    _PLURAL_ENDS = ("ִים", "וֹת")

    @classmethod
    def script_dual(cls, script: str) -> bool:
        return cls._suffix_form(script).endswith(cls._DUAL_END)

    @classmethod
    def script_plural(cls, script: str) -> bool:
        return cls._suffix_form(script).endswith(cls._PLURAL_ENDS)

    # D (piʕel) gemination of C2; Š (hifʕil) hi-/he- prefix; N (nifʕal) ni-/na-;
    # tD (hitpaʕel) hit-.  Prefixes match IPA (Modern Hebrew IPA rarely marks
    # the geminate, so D degemination is usually a no-op → cited base preferred).
    _STEM_PREFIX = re.compile(r"^h[ie]")
    _N_PREFIX = re.compile(r"^n[ie]")
    _T_PREFIX = re.compile(r"^hit")
    # ha- article + dagesh-forte gemination of the next consonant, in IPA.
    _ARTICLE_IPA = re.compile(r"^ha")

    @classmethod
    def synthesize_base_stem(cls, ipa: str, stem: str) -> tuple[str, str] | None:
        """Reduce a derived-stem verb IPA toward the G base."""
        if stem == "D":
            out = _degeminate_ipa(ipa)
            if out:
                return out, "D-stem (piʕel) degeminated"
        if stem == "Š":
            m = cls._STEM_PREFIX.match(ipa)
            if m and _letters(ipa[m.end():]) >= 3:
                return _clean_leading(ipa[m.end():]), "Š-stem (hifʕil) hi- prefix stripped"
        if stem == "N":
            m = cls._N_PREFIX.match(ipa)
            if m and _letters(ipa[m.end():]) >= 3:
                return _clean_leading(ipa[m.end():]), "N-stem (nifʕal) ni- prefix stripped"
        if stem == "tD":
            m = cls._T_PREFIX.match(ipa)
            if m and _letters(ipa[m.end():]) >= 3:
                base = _clean_leading(ipa[m.end():])
                return (_degeminate_ipa(base) or base,
                        "tD-stem (hitpaʕel) hit- prefix stripped, degeminated")
        return None

    @classmethod
    def strip_article_ipa(cls, ipa: str, script: str = "") -> str | None:
        """Strip the Hebrew ha- article in IPA: leading /ha/ plus the dagesh-
        forte gemination it triggers on the next consonant."""
        m = cls._ARTICLE_IPA.match(ipa)
        if not m:
            return None
        rest = _clean_leading(ipa[m.end():])
        out = _degeminate_ipa(rest) or rest
        return out if _letters(out) >= 2 else None


class PansemiticMorphology(LangMorphology):
    """The merged ancestor's morphology: how shared layers re-attach to a
    pansemitic stem.

    Pansemitic only ever *produces* — it is the merge target, not a parsed
    source — so it carries no script-evidence hooks or strip patterns, just
    the compromise affixes that blend the Arabic and Hebrew shapes (see the
    pansemitic-morphology-conventions memory):

      - definite: the space-separated particle "hal " (al-/ha- blend);
      - nisba: -i;
      - dual collapses into the plural, and the plural is -at for feminine
        stems, -im otherwise;
      - a bare feminine ending is the compromise -a (ar -a(t) / he -á blend),
        re-attached after both sides are stripped to a clean common stem;
      - verbal stems re-attach Proto-Semitic exponents (D geminates C2; N na-,
        Š ša-, t-stems ta-).  Gemination is re-introduced *after* the pansemitic
        reduction (which strips lexical gemination), so it is a morphology-only
        exponent in the pansemitic layer.
    """
    lang = "pansemitic"
    produce_prefixes = {Layer.DEFINITE: "hal "}
    produce_suffixes = {
        Layer.FEMININE: "a", Layer.NISBA: "i", Layer.DUAL: "im", Layer.PLURAL: "im",
    }
    # Feminine number marking overrides the default -im plural.
    feminine_plural = "at"
    # Verbal-stem prefixes (innermost, between any definite article and the
    # stem); at most one applies.  Proto-Semitic markers in IPA — the merged
    # word is stored as IPA, so the causative is ʃa (to_protosemitic_convention
    # renders it back to ša).
    stem_prefixes = {Layer.CAUSATIVE: "ʃa", Layer.N_PREFIX: "na", Layer.T_PREFIX: "ta"}
    _STEM_PREFIX_ORDER = (Layer.CAUSATIVE, Layer.N_PREFIX, Layer.T_PREFIX)

    _PAN_VOWELS = frozenset("aiu")

    @classmethod
    def _geminate_c2(cls, stem: str) -> str:
        """Double the second consonant (the D exponent) on a pansemitic stem.

        Re-tokenizes via Phoneme.parse so tie-bar digraphs (d͡ʒ) and emphatics
        (sˤ) are treated as single consonants, and marks length with ː."""
        toks = [p.tok for p in Phoneme.parse(stem)]
        cons = [i for i, p in enumerate(Phoneme.parse(stem))
                if isinstance(p, Consonant)]
        if len(cons) < 2:
            return stem
        i2 = cons[1]
        if "ː" not in toks[i2]:
            toks[i2] = toks[i2] + "ː"
        return "".join(toks)

    @classmethod
    def produce_word(cls, word: AnalyzedWord) -> str:
        """Re-attach shared layers to a merged pansemitic stem.

        Order: definite article, then verbal-stem prefix, then the (possibly
        geminated) stem, then the single nominal suffix.  Number (dual/plural)
        outranks nisba and bare feminine for that suffix slot, and a feminine
        stem takes -at rather than -im."""
        layers = word.layers
        stem = word.roman
        if Layer.GEMINATION in layers:
            stem = cls._geminate_c2(stem)
        stem_prefix = next((cls.stem_prefixes[L] for L in cls._STEM_PREFIX_ORDER
                            if L in layers), "")
        definite = (cls.produce_prefixes[Layer.DEFINITE]
                    if Layer.DEFINITE in layers else "")
        if {Layer.DUAL, Layer.PLURAL} & layers:
            suffix = (cls.feminine_plural if Layer.FEMININE in layers
                      else cls.produce_suffixes[Layer.PLURAL])
        elif Layer.NISBA in layers:
            suffix = cls.produce_suffixes[Layer.NISBA]
        elif Layer.FEMININE in layers:
            suffix = cls.produce_suffixes[Layer.FEMININE]
        else:
            suffix = ""
        return definite + stem_prefix + stem + suffix


MORPHOLOGY_CONFIG: dict[str, type[LangMorphology]] = {
    "ar": ArabicMorphology,
    "he": HebrewMorphology,
    "pansemitic": PansemiticMorphology,
}


def morphology_for(lang: str) -> type[LangMorphology] | None:
    return MORPHOLOGY_CONFIG.get(lang)


def analyze_phrase(
    lang: str,
    script: str,
    roman: str,
    ipa: str = "",
    pos: Iterable[str] = (),
    verb_forms: Iterable[str] = (),
    derivation: Iterable[str] = (),
    number: Iterable[str] = (),
    gender: Iterable[str] = (),
    derived_from: Iterable[str] = (),
    singular_of: Iterable[str] = (),
    masculine_of: Iterable[str] = (),
) -> AnalyzedPhrase:
    """Split a headword into analyzed words via the language's tokenizer.

    *ipa* is the phrase's phoneme string (from word_from_sharedsource); it is
    tokenized in step with script/roman so each word carries its own IPA for
    the merge to strip and reconstruct."""
    morph = MORPHOLOGY_CONFIG[lang]
    pos = frozenset(pos)
    number = frozenset(number)
    verb_forms = frozenset(verb_forms)
    return AnalyzedPhrase(
        lang=lang,
        roman=roman,
        ipa=ipa,
        words=[morph.analyze_word(s, r, i, pos, number)
               for s, r, i in morph.tokenize(script, roman, ipa)],
        pos=pos,
        verb_forms=verb_forms,
        verb_stem=frozenset(s for f in verb_forms
                            if (s := morph.verb_stem(f)) is not None),
        derivation=frozenset(Derivation(d) for d in derivation),
        number=number,
        gender=frozenset(gender),
        derived_from=frozenset(derived_from),
        singular_of=frozenset(singular_of),
        masculine_of=frozenset(masculine_of),
    )


_CAUSATIVE_NOMINAL_POS = {"noun", "adj", "name", "num"}


def _verb_vs_nominal(verb_side: AnalyzedPhrase, nominal_side: AnalyzedPhrase) -> bool:
    return ("verb" in verb_side.pos
            and "verb" not in nominal_side.pos
            and bool(nominal_side.pos & _CAUSATIVE_NOMINAL_POS))


def _strictly_feminine(phrase: AnalyzedPhrase) -> bool:
    """Feminine without competing masculine marking — picks the shared
    plural compromise suffix (-at vs -im)."""
    return "f" in phrase.gender and "m" not in phrase.gender


_ROMAN_TO_WORD = {
    "ar": ArabicWord.from_romanization,
    "he": HebrewWord.from_romanization,
}


def _roman_to_ipa(lang: str, roman: str) -> str:
    """Convert a cited base romanization to IPA (the merge's working space)."""
    return _ROMAN_TO_WORD[lang](roman).word


def _lookup_base_ipa(
    lang: str,
    norms: frozenset[str],
    prefer_pos: frozenset[str],
    base_lookup: BaseLookup,
    notes: list[str],
    label: str,
) -> str | None:
    """Resolve *norms* to a cited lexeme's IPA via base_lookup, preferring a
    base of *prefer_pos* among homographs (the lookup returns romanization,
    converted to IPA here)."""
    if not norms:
        return None
    hit = base_lookup(lang, norms, prefer_pos)
    if hit is None:
        return None
    canonical, roman = hit
    notes.append(f"{label}: substituted cited base {canonical} ({roman})")
    return _roman_to_ipa(lang, roman)


def _substitute_base(
    phrase: AnalyzedPhrase,
    base_lookup: BaseLookup,
    notes: list[str],
    label: str,
) -> str | None:
    """Swap in the IPA of the derivational (noun-from-verb) base kaikki
    cites — a verb."""
    return _lookup_base_ipa(
        phrase.lang, phrase.derived_from, _VERB_BASE_POS, base_lookup, notes, label)


def _reduce_to_base(
    phrase: AnalyzedPhrase,
    morph: type[LangMorphology],
    ipa: str,
    layer: Layer,
    base_norms: frozenset[str],
    base_kind: str,
    base_lookup: BaseLookup,
    notes: list[str],
    label: str,
) -> tuple[str | None, bool]:
    """Reduce an inflected side to its stem, in IPA.

    Prefers the base lexeme Wiktionary actually cites (resolved via
    base_lookup) — exact, and it reaches irregular forms the regex can't
    (broken plurals, feminines with stem-vowel changes like he malká/melekh);
    falls back to the language's IPA suffix-strip.  Returns (stem, used_cited_base)
    so the caller can phrase its note accurately."""
    base = _lookup_base_ipa(
        phrase.lang, base_norms, _NOMINAL_BASE_POS, base_lookup, notes,
        f"{label}: cited {base_kind}")
    if base:
        return base, True
    return morph.strip_ipa(ipa, layer), False


def _deverbal_category(phrase: AnalyzedPhrase) -> Derivation | None:
    """The phrase's single templatic deverbal category, by priority."""
    for c in _DEVERBAL_PRIORITY:
        if c in phrase.derivation:
            return c
    return None


def _reduce_verb_stem(
    phrase: AnalyzedPhrase,
    ipa: str,
    base_lookup: BaseLookup,
    notes: list[str],
    label: str,
) -> str | None:
    """Reduce a verb / deverbal IPA toward its G-stem base: the cited base
    lexeme first (a participle's or verbal noun's underlying verb, or a derived
    stem's form-I), then the language's stem template synthesis."""
    base = _substitute_base(phrase, base_lookup, notes, label)
    if base:
        return base
    morph = MORPHOLOGY_CONFIG[phrase.lang]
    for stem in phrase.verb_stem:
        if stem == "G":
            continue
        synth = morph.synthesize_base_stem(ipa, stem)
        if synth is not None:
            new_ipa, note = synth
            notes.append(f"{label}: {note}")
            return new_ipa
    return None


def _pansemitic_affix(layers: set[Layer]) -> str:
    """The bare compromise affix a merged stem carries for *layers*.

    Produces a pansemitic word from an empty stem, so the definite/nisba/
    number shapes come from PansemiticMorphology rather than living as
    literals in merge."""
    return PansemiticMorphology.produce_word(AnalyzedWord("", "", layers=set(layers)))


def _strip_definite(
    morph: type[LangMorphology],
    word: AnalyzedWord,
    ipa: str,
    other_definite: bool,
    multiword: bool,
) -> str | None:
    """Strip *word*'s article (in IPA) if detected and sufficiently evidenced."""
    if Layer.DEFINITE not in word.layers:
        return None
    if (morph.article_needs_corroboration
            and not multiword and not other_definite):
        return None
    return morph.strip_article_ipa(ipa, word.script)


def _reconstruct_word_pairs(
    word_pairs: list[PlannedPair],
) -> tuple[Word, PansemiticWord]:
    """Reconstruct each aligned pair into its proto-Semitic ancestor stem and
    pansemitic reduction; multi-word ancestors are space-joined.

    Inputs are already IPA stems (morphology stripped in phoneme space).  The
    ancestor is the bare merged proto-Semitic stem; the shared morphology is
    re-attached on the pansemitic side, after the stem is reduced to the
    pansemitic inventory, so the compromise affixes (hal-, -i, -im/-at, -a)
    live in pansemitic phonology rather than being glued onto the ancestor."""
    anc_parts: list[str] = []
    pan_parts: list[str] = []
    for pair in word_pairs:
        merged = reconstruct_from_words(
            ArabicWord.from_ipa(pair.ar_ipa), HebrewWord.from_ipa(pair.he_ipa))
        anc_parts.append(merged.word)
        pan_stem = PansemiticWord.from_word(merged).word
        pan_parts.append(
            PansemiticMorphology.produce_word(
                AnalyzedWord("", pan_stem, layers=pair.layers)))
    return (ReconstructedSemProWord(word=" ".join(anc_parts)),
            PansemiticWord(word=" ".join(pan_parts)))


def merge(
    ar: AnalyzedPhrase,
    he: AnalyzedPhrase,
    base_lookup: BaseLookup,
) -> MergeResult:
    """Align, morphology-normalize, and reconstruct a cognate pair.

    Aligns the two phrases word-by-word, stripping asymmetric layers only: a
    layer detected on both sides of an aligned word pair is shared morphology —
    kept, or re-attached as a compromise affix.  Each aligned pair is then
    reconstructed into its proto-Semitic ancestor (and pansemitic reduction).
    Falls back to the unsplit pair when word counts differ.
    """
    ar_m = MORPHOLOGY_CONFIG[ar.lang]
    he_m = MORPHOLOGY_CONFIG[he.lang]
    notes: list[str] = []
    if len(ar.words) != len(he.words):
        notes.append(
            f"word-count mismatch (ar {len(ar.words)} vs he {len(he.words)}); merged unsplit")
        ancestor, pansemitic = _reconstruct_word_pairs([PlannedPair(ar.ipa, he.ipa)])
        return MergeResult(ancestor=ancestor, pansemitic=pansemitic, notes=notes)

    multiword = len(ar.words) > 1
    word_pairs: list[PlannedPair] = []
    plan_verb_stem: str | None = None
    plan_derivation: str | None = None
    for i, (aw, hw) in enumerate(zip(ar.words, he.words)):
        a_i, h_i = aw.ipa, hw.ipa  # working stems, in phoneme space
        shared: set[Layer] = set()  # shared morphemes, re-attached on the pansemitic side
        where = f" (word {i + 1})" if multiword else ""

        a_stripped = _strip_definite(
            ar_m, aw, a_i, Layer.DEFINITE in hw.layers, multiword)
        h_stripped = _strip_definite(
            he_m, hw, h_i, Layer.DEFINITE in aw.layers, multiword)
        if a_stripped:
            a_i = a_stripped
        if h_stripped:
            h_i = h_stripped
        if a_stripped and h_stripped:
            shared.add(Layer.DEFINITE)
            notes.append(
                f"shared definite article → {_pansemitic_affix({Layer.DEFINITE}).strip()}{where}")
        elif a_stripped:
            notes.append(f"ar: definite article stripped{where}")
        elif h_stripped:
            notes.append(f"he: definite article stripped{where}")

        # Asymmetric feminine is reduced to the masculine stem: the masculine
        # base Wiktionary cites (he-noun m=, "feminine of X") where available
        # — exact, and it handles stem-vowel changes like he malká/melekh that
        # suffix-stripping can't — else the IPA strip.  masculine_of is
        # lexeme-level, so it only applies to single-word lemmas.
        ar_fem = Layer.FEMININE in aw.layers
        he_fem = Layer.FEMININE in hw.layers
        ar_masc = frozenset() if multiword else ar.masculine_of
        he_masc = frozenset() if multiword else he.masculine_of
        acted = False
        if ar_fem and not he_fem:
            stripped, used = _reduce_to_base(
                ar, ar_m, a_i, Layer.FEMININE, ar_masc, "masculine base",
                base_lookup, notes, "ar")
            if stripped:
                a_i = stripped
                acted = True
                if not used:
                    notes.append(f"ar: feminine ending stripped{where}")
        elif he_fem and not ar_fem:
            stripped, used = _reduce_to_base(
                he, he_m, h_i, Layer.FEMININE, he_masc, "masculine base",
                base_lookup, notes, "he")
            if stripped:
                h_i = stripped
                acted = True
                if not used:
                    notes.append(f"he: feminine ending stripped{where}")
        elif ar_fem and he_fem:
            # Shared feminine: strip both to the bare common stem (lossless)
            # and re-attach the pansemitic compromise -a, rather than leaving
            # ar -a(t) / he -á for the aligner to reconcile.
            sa = ar_m.strip_ipa(a_i, Layer.FEMININE)
            sh = he_m.strip_ipa(h_i, Layer.FEMININE)
            if sa and sh:
                a_i, h_i = sa, sh
                shared.add(Layer.FEMININE)
                acted = True
                notes.append(
                    f"shared feminine ending → -{_pansemitic_affix({Layer.FEMININE})}{where}")

        ar_nis = Layer.NISBA in aw.layers
        he_nis = Layer.NISBA in hw.layers
        if ar_nis and he_nis:
            sa = ar_m.strip_ipa(a_i, Layer.NISBA)
            sh = he_m.strip_ipa(h_i, Layer.NISBA)
            if sa and sh:
                a_i, h_i = sa, sh
                shared.add(Layer.NISBA)
                acted = True
                notes.append(
                    f"shared nisba suffix → -{_pansemitic_affix({Layer.NISBA})}{where}")
        elif ar_nis:
            stripped = ar_m.strip_ipa(a_i, Layer.NISBA)
            if stripped:
                a_i = stripped
                acted = True
                notes.append(f"ar: nisba suffix stripped (de-adjectivized){where}")
        elif he_nis:
            stripped = he_m.strip_ipa(h_i, Layer.NISBA)
            if stripped:
                h_i = stripped
                acted = True
                notes.append(f"he: nisba suffix stripped (de-adjectivized){where}")

        # Dual/plural normalization (lexeme-level metadata, single-word
        # only).  The singular stem comes from the singular Wiktionary cites
        # (exact, and handles broken plurals) where available, else the
        # romanization suffix-strip.  Asymmetric number is reduced outright;
        # when both sides are non-singular the stems merge and the compromise
        # plural suffix is re-attached — duals merge into plurals in
        # pansemitic, -im for masculine, -at for feminine.  Runs even after a
        # feminine strip: ar ḥayāh (fem) vs he khayím (plural) needs both.
        if not multiword:
            a_num = next(iter({Layer.DUAL, Layer.PLURAL} & aw.layers), None)
            h_num = next(iter({Layer.DUAL, Layer.PLURAL} & hw.layers), None)
            if a_num and h_num:
                sa, _ = _reduce_to_base(
                    ar, ar_m, a_i, a_num, ar.singular_of, "singular base",
                    base_lookup, notes, "ar")
                sh, _ = _reduce_to_base(
                    he, he_m, h_i, h_num, he.singular_of, "singular base",
                    base_lookup, notes, "he")
                if sa and sh:
                    a_i, h_i = sa, sh
                    fem = _strictly_feminine(ar) or _strictly_feminine(he)
                    shared.add(Layer.PLURAL)
                    if fem:
                        shared.add(Layer.FEMININE)
                    acted = True
                    kind = "/".join(sorted({a_num.value, h_num.value}))
                    notes.append(f"shared {kind} → -{_pansemitic_affix(shared)}")
            elif a_num:
                stripped, used = _reduce_to_base(
                    ar, ar_m, a_i, a_num, ar.singular_of, "singular base",
                    base_lookup, notes, "ar")
                if stripped:
                    a_i = stripped
                    acted = True
                    if not used:
                        notes.append(f"ar: {a_num.value} suffix stripped")
            elif h_num:
                stripped, used = _reduce_to_base(
                    he, he_m, h_i, h_num, he.singular_of, "singular base",
                    base_lookup, notes, "he")
                if stripped:
                    h_i = stripped
                    acted = True
                    if not used:
                        notes.append(f"he: {h_num.value} suffix stripped")

        # Verbal morphology (lexeme-level metadata, single-word only).
        if not multiword:
            a_cat = _deverbal_category(ar)
            h_cat = _deverbal_category(he)
            a_stem = next(iter(ar.verb_stem), None)
            h_stem = next(iter(he.verb_stem), None)

            if a_cat is not None and a_cat == h_cat:
                # Shared deverbal category: preserve.  The deepest lossless
                # common layer is the surface itself — underiving to the
                # vowelless root would drop the template's vowel melody — so the
                # stems merge as-is and we only record the shared analysis.
                plan_verb_stem = a_stem if a_stem == h_stem else None
                plan_derivation = a_cat.value
                label = " ".join(x for x in (plan_verb_stem, a_cat.value) if x)
                notes.append(f"shared {label} preserved")
            elif (a_cat is None and h_cat is None
                  and "verb" in ar.pos and "verb" in he.pos):
                # Two plain finite verbs sharing a stem.  Derived stems (D, N,
                # Š, tD) go through the layers process: strip both sides to the
                # G base, merge the clean stems, and re-attach the stem's
                # pansemitic exponent (gemination / na-/ša-/ta-).  G needs no
                # work.  If either side won't cleanly reduce, fall back to
                # merging the surfaces whole.
                if a_stem is not None and a_stem == h_stem:
                    plan_verb_stem = a_stem
                    stem_layers = _STEM_LAYERS.get(a_stem, frozenset())
                    if stem_layers:
                        # The stem is known from the binyan/form, so always
                        # re-apply its exponent; reduce each side to the G base
                        # best-effort (a no-op where the exponent isn't in the
                        # surface — e.g. Modern Hebrew piʕel carries no
                        # gemination — leaves an already-bare stem).
                        sa = ar_m.synthesize_base_stem(a_i, a_stem)
                        sh = he_m.synthesize_base_stem(h_i, a_stem)
                        a_i = sa[0] if sa else a_i
                        h_i = sh[0] if sh else h_i
                        shared |= stem_layers
                        notes.append(
                            f"shared {a_stem}-stem verb → reduced to G, {a_stem} re-applied")
                    else:
                        notes.append(f"shared {a_stem}-stem verb")
            elif not acted:
                # Asymmetric: reduce the more-derived side toward its G-stem
                # base so the two sides align.  A lone deverbal noun/adj
                # (participle, verbal noun, instance noun) is reduced to its
                # cited verb; failing that, a finite verb opposite a plain
                # nominal is de-derived (or the nominal's verb base swapped in).
                if a_cat is not None and h_cat is None:
                    new_a = _reduce_verb_stem(ar, a_i, base_lookup, notes, "ar")
                    if new_a:
                        a_i = new_a
                elif h_cat is not None and a_cat is None:
                    new_h = _reduce_verb_stem(he, h_i, base_lookup, notes, "he")
                    if new_h:
                        h_i = new_h
                elif _verb_vs_nominal(ar, he):
                    new_a = _reduce_verb_stem(ar, a_i, base_lookup, notes, "ar")
                    if new_a:
                        a_i = new_a
                    else:
                        new_h = _substitute_base(he, base_lookup, notes, "he")
                        if new_h:
                            h_i = new_h
                elif _verb_vs_nominal(he, ar):
                    new_h = _reduce_verb_stem(he, h_i, base_lookup, notes, "he")
                    if new_h:
                        h_i = new_h
                    else:
                        new_a = _substitute_base(ar, base_lookup, notes, "ar")
                        if new_a:
                            a_i = new_a

        word_pairs.append(PlannedPair(a_i, h_i, layers=shared))

    ancestor, pansemitic = _reconstruct_word_pairs(word_pairs)
    return MergeResult(ancestor=ancestor, pansemitic=pansemitic, notes=notes,
                       verb_stem=plan_verb_stem, derivation=plan_derivation)
