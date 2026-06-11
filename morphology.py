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
  - A feminine ending present on only ONE side (ar tāʔ marbūṭa, he qamats-he)
    is stripped; present on both sides it is shared morphology and kept —
    the aligner naturally merges ar -a with he -á into the shared -a.
  - A nisba adjectivizer (ar -iyy, he -i; adjective POS required) present
    on one side is stripped (de-adjectivization); present on both sides,
    the stems are merged and the suffix re-attached as the compromise -i.
  - A verb paired with a nominal is de-causativized / de-verbalized:
    preferring substitution of the base lexeme kaikki cites (form_of with
    the noun-from-verb tag), falling back to per-language template synthesis
    (Arabic form II: degeminate C2; form IV: strip ʔa- prefix).

Detection is evidence-gated: every strip needs BOTH the script-side signal
(pointing/letters) and a matching romanization shape, and must leave at
least two letters behind, otherwise the word passes through untouched.

Language knowledge lives in one LangMorphology subclass per language,
mirroring reconstruction.py's one-Word-class-per-language pattern.  To
extend coverage (e.g. Aramaic), subclass LangMorphology, override the
script-evidence hooks / strip patterns / synthesis methods, and register
the class in MORPHOLOGY_CONFIG.

`plan_merge` returns aligned (ar_roman, he_roman) word pairs plus
human-readable notes describing exactly what was normalized; the notes are
surfaced in the output so pansemitic forms stay auditable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, ClassVar, Iterable

from reconstruction import ArabicWord


class Layer(Enum):
    """Strippable surface morphology detectable from script + romanization."""
    DEFINITE = "definite"
    FEMININE = "feminine"
    NISBA = "nisba"
    DUAL = "dual"
    PLURAL = "plural"


@dataclass
class AnalyzedWord:
    """One orthographic word with its detected strippable layers."""
    script: str
    roman: str
    layers: set[Layer] = field(default_factory=set)


@dataclass
class AnalyzedPhrase:
    """A headword split into words, plus lexeme-level kaikki metadata."""
    lang: str
    roman: str                       # full original romanization
    words: list[AnalyzedWord]
    pos: frozenset[str] = frozenset()
    verb_forms: frozenset[str] = frozenset()   # ar form (I..X) / he binyan
    number: frozenset[str] = frozenset()       # ⊆ {"p", "d"} (plural/dual lemma)
    gender: frozenset[str] = frozenset()       # ⊆ {"m", "f"}
    derived_from: frozenset[str] = frozenset() # normalized derivational bases


@dataclass
class PlannedPair:
    """One aligned word pair, morphology-normalized, ready to merge.

    prefix/suffix carry shared morphology that was stripped from both sides
    for a clean stem merge and should be re-attached to the merged ancestor
    (e.g. shared definiteness as "hal ", shared nisba as "i")."""
    ar_roman: str
    he_roman: str
    prefix: str = ""
    suffix: str = ""


@dataclass
class MergePlan:
    """Aligned per-word pairs ready for reconstruction."""
    word_pairs: list[PlannedPair]
    notes: list[str]


# Looks up the (canonical, romanization) of a derivational base lexeme in
# the caller's word index, given (lang, normalized base candidates).
BaseLookup = Callable[[str, frozenset[str]], tuple[str, str] | None]


def _letters(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


class LangMorphology:
    """Per-language morphological knowledge.

    One subclass per language, registered in MORPHOLOGY_CONFIG.  The base
    class implements the generic detect/strip machinery; subclasses supply
    the script-evidence hooks, the romanization strip shapes, and (where
    the language has them) template-level de-causativization and the
    article shape in already-converted IPA."""

    lang: ClassVar[str]
    # Romanization-side strip shape per layer; a missing entry means the
    # language does not support that layer.
    strip_patterns: ClassVar[dict[Layer, re.Pattern[str]]] = {}
    # Verb forms regarded as the underived base stem, used to rank homograph
    # candidates during base substitution (ar "I", he "pa").
    base_verb_forms: ClassVar[frozenset[str]] = frozenset()
    # Whether a single-word article strip needs the other side of the pair
    # to also be definite (script evidence alone too weak — Hebrew).
    article_needs_corroboration: ClassVar[bool] = False

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
    def tokenize(cls, script: str, roman: str) -> list[tuple[str, str]]:
        """Aligned (script, roman) word tokens.

        Script and romanization must tokenize to the same word count to
        split; otherwise the phrase is kept whole (so a failed alignment
        degrades to whole-string merging, never to misaligned words)."""
        s_toks = cls.script_tokens(script)
        r_toks = cls.roman_tokens(roman)
        if len(s_toks) == len(r_toks) and len(s_toks) > 1:
            return list(zip(s_toks, r_toks))
        return [(script, roman)]

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
        return AnalyzedWord(script=script, roman=roman, layers=layers)

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

    # ── language-specific operations (override where applicable) ────
    @classmethod
    def synthesize_decausative(cls, roman: str, verb_forms: frozenset[str]) -> tuple[str, str] | None:
        """Template-level de-causativization of a verb romanization.

        Returns (new_roman, note) or None when the language has no usable
        template (or the romanization doesn't fit one)."""
        return None

    @classmethod
    def strip_article_ipa(cls, ipa: str, script: str) -> str | None:
        """Strip a leading definite article from an already-converted IPA
        string (used for shared-source ancestors, which never pass through
        plan_merge).  None when unsupported or unevidenced."""
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
    base_verb_forms = frozenset({"I"})

    # Doubled consonant in a romanization (form-II gemination); long vowels
    # are single precomposed codepoints (ā, ī …) so excluding plain vowels
    # suffices.
    _DOUBLED = re.compile(r"([^\W\d_aeiou])\1")
    _FORM_IV_PREFIX = re.compile(r"^[ʔʾˀ]?a")

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
    def synthesize_decausative(cls, roman: str, verb_forms: frozenset[str]) -> tuple[str, str] | None:
        if "II" in verb_forms:
            m = cls._DOUBLED.search(roman)
            if m:
                out = roman[: m.start() + 1] + roman[m.end():]
                return out, "form-II verb degeminated"
        if "IV" in verb_forms:
            m = cls._FORM_IV_PREFIX.match(roman)
            if m and _letters(roman[m.end():]) >= 3:
                return roman[m.end():], "form-IV ʔa- prefix stripped"
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
            return out or None
        if not (cls.script_definite(script)
                or cls.strip_patterns[Layer.DEFINITE].match(script.lower())):
            return None
        for pattern, repl in cls._ARTICLE_IPA_GATED:
            out = pattern.sub(repl, ipa, count=1)
            if out != ipa:
                return out or None
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
    base_verb_forms = frozenset({"pa"})
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


MORPHOLOGY_CONFIG: dict[str, type[LangMorphology]] = {
    "ar": ArabicMorphology,
    "he": HebrewMorphology,
}


def morphology_for(lang: str) -> type[LangMorphology] | None:
    return MORPHOLOGY_CONFIG.get(lang)


def analyze_phrase(
    lang: str,
    script: str,
    roman: str,
    pos: Iterable[str] = (),
    verb_forms: Iterable[str] = (),
    number: Iterable[str] = (),
    gender: Iterable[str] = (),
    derived_from: Iterable[str] = (),
) -> AnalyzedPhrase:
    """Split a headword into analyzed words via the language's tokenizer."""
    morph = MORPHOLOGY_CONFIG[lang]
    pos = frozenset(pos)
    number = frozenset(number)
    pairs = morph.tokenize(script, roman)
    return AnalyzedPhrase(
        lang=lang,
        roman=roman,
        words=[morph.analyze_word(s, r, pos, number) for s, r in pairs],
        pos=pos,
        verb_forms=frozenset(verb_forms),
        number=number,
        gender=frozenset(gender),
        derived_from=frozenset(derived_from),
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


def _substitute_base(
    phrase: AnalyzedPhrase,
    base_lookup: BaseLookup,
    notes: list[str],
    label: str,
) -> str | None:
    """Swap in the romanization of the derivational base kaikki cites."""
    if not phrase.derived_from:
        return None
    hit = base_lookup(phrase.lang, phrase.derived_from)
    if hit is None:
        return None
    canonical, roman = hit
    notes.append(f"{label}: substituted cited base {canonical} ({roman})")
    return roman


def _decausativize(
    phrase: AnalyzedPhrase,
    roman: str,
    base_lookup: BaseLookup,
    notes: list[str],
    label: str,
) -> str | None:
    """De-causativize a verb-side romanization: cited base first, then the
    language's template synthesis."""
    base = _substitute_base(phrase, base_lookup, notes, label)
    if base:
        return base
    synth = MORPHOLOGY_CONFIG[phrase.lang].synthesize_decausative(
        roman, phrase.verb_forms)
    if synth is not None:
        new_roman, note = synth
        notes.append(f"{label}: {note}")
        return new_roman
    return None


def _strip_definite(
    morph: type[LangMorphology],
    word: AnalyzedWord,
    roman: str,
    other_definite: bool,
    multiword: bool,
) -> str | None:
    """Strip *word*'s article if detected and sufficiently evidenced."""
    if Layer.DEFINITE not in word.layers:
        return None
    if (morph.article_needs_corroboration
            and not multiword and not other_definite):
        return None
    return morph.strip(roman, Layer.DEFINITE)


def plan_merge(
    ar: AnalyzedPhrase,
    he: AnalyzedPhrase,
    base_lookup: BaseLookup,
) -> MergePlan:
    """Produce aligned, morphology-normalized word pairs for reconstruction.

    Strips asymmetric layers only: a layer detected on both sides of an
    aligned word pair is shared morphology — kept, or re-attached as a
    compromise affix.  Falls back to the unsplit pair when word counts
    differ.
    """
    ar_m = MORPHOLOGY_CONFIG[ar.lang]
    he_m = MORPHOLOGY_CONFIG[he.lang]
    notes: list[str] = []
    if len(ar.words) != len(he.words):
        notes.append(
            f"word-count mismatch (ar {len(ar.words)} vs he {len(he.words)}); merged unsplit")
        return MergePlan(word_pairs=[PlannedPair(ar.roman, he.roman)], notes=notes)

    multiword = len(ar.words) > 1
    word_pairs: list[PlannedPair] = []
    for i, (aw, hw) in enumerate(zip(ar.words, he.words)):
        a_r, h_r = aw.roman, hw.roman
        prefix = suffix = ""
        where = f" (word {i + 1})" if multiword else ""

        a_stripped = _strip_definite(
            ar_m, aw, a_r, Layer.DEFINITE in hw.layers, multiword)
        h_stripped = _strip_definite(
            he_m, hw, h_r, Layer.DEFINITE in aw.layers, multiword)
        if a_stripped:
            a_r = a_stripped
        if h_stripped:
            h_r = h_stripped
        if a_stripped and h_stripped:
            prefix = "hal "
            notes.append(f"shared definite article → hal{where}")
        elif a_stripped:
            notes.append(f"ar: definite article stripped{where}")
        elif h_stripped:
            notes.append(f"he: definite article stripped{where}")

        ar_fem = Layer.FEMININE in aw.layers
        he_fem = Layer.FEMININE in hw.layers
        acted = False
        if ar_fem and not he_fem:
            stripped = ar_m.strip(a_r, Layer.FEMININE)
            if stripped:
                a_r = stripped
                acted = True
                notes.append(f"ar: feminine ending stripped{where}")
        elif he_fem and not ar_fem:
            stripped = he_m.strip(h_r, Layer.FEMININE)
            if stripped:
                h_r = stripped
                acted = True
                notes.append(f"he: feminine ending stripped{where}")
        # Symmetric feminine needs no handling: the aligner merges ar -a
        # with he -á into the shared -a on its own.

        ar_nis = Layer.NISBA in aw.layers
        he_nis = Layer.NISBA in hw.layers
        if ar_nis and he_nis:
            sa = ar_m.strip(a_r, Layer.NISBA)
            sh = he_m.strip(h_r, Layer.NISBA)
            if sa and sh:
                a_r, h_r = sa, sh
                suffix = "i"
                acted = True
                notes.append(f"shared nisba suffix → -i{where}")
        elif ar_nis:
            stripped = ar_m.strip(a_r, Layer.NISBA)
            if stripped:
                a_r = stripped
                acted = True
                notes.append(f"ar: nisba suffix stripped (de-adjectivized){where}")
        elif he_nis:
            stripped = he_m.strip(h_r, Layer.NISBA)
            if stripped:
                h_r = stripped
                acted = True
                notes.append(f"he: nisba suffix stripped (de-adjectivized){where}")

        # Dual/plural normalization (lexeme-level metadata, single-word
        # only).  Asymmetric number is stripped outright; when both sides
        # are non-singular the stems merge and the compromise plural suffix
        # is re-attached — duals merge into plurals in pansemitic, -im for
        # masculine, -at for feminine.  Runs even after a feminine strip:
        # ar ḥayāh (fem) vs he khayím (plural) needs both reductions.
        if not multiword:
            a_num = next(iter({Layer.DUAL, Layer.PLURAL} & aw.layers), None)
            h_num = next(iter({Layer.DUAL, Layer.PLURAL} & hw.layers), None)
            if a_num and h_num:
                sa = ar_m.strip(a_r, a_num)
                sh = he_m.strip(h_r, h_num)
                if sa and sh:
                    a_r, h_r = sa, sh
                    suffix = ("at" if _strictly_feminine(ar) or _strictly_feminine(he)
                              else "im")
                    acted = True
                    kind = "/".join(sorted({a_num.value, h_num.value}))
                    notes.append(f"shared {kind} → -{suffix}")
            elif a_num:
                stripped = ar_m.strip(a_r, a_num)
                if stripped:
                    a_r = stripped
                    acted = True
                    notes.append(f"ar: {a_num.value} suffix stripped")
            elif h_num:
                stripped = he_m.strip(h_r, h_num)
                if stripped:
                    h_r = stripped
                    acted = True
                    notes.append(f"he: {h_num.value} suffix stripped")

        # POS promotion uses lexeme-level metadata, so single-word only; a
        # feminine/nisba/number strip already reduced one side to its stem.
        if not multiword and not acted:
            if _verb_vs_nominal(ar, he):
                new_a = _decausativize(ar, a_r, base_lookup, notes, "ar")
                if new_a:
                    a_r = new_a
                else:
                    new_h = _substitute_base(he, base_lookup, notes, "he")
                    if new_h:
                        h_r = new_h
            elif _verb_vs_nominal(he, ar):
                new_h = _decausativize(he, h_r, base_lookup, notes, "he")
                if new_h:
                    h_r = new_h
                else:
                    new_a = _substitute_base(ar, base_lookup, notes, "ar")
                    if new_a:
                        a_r = new_a

        word_pairs.append(PlannedPair(a_r, h_r, prefix=prefix, suffix=suffix))

    return MergePlan(word_pairs=word_pairs, notes=notes)
