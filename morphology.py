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
    see DeverbalCategory).  When both sides share a deverbal category it is preserved
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
from protoroot import (
    arabic_root_radicals,
    consonant_skeleton,
    hebrew_root_radicals,
)
from reconstruction import (
    ArabicWord,
    HebrewWord,
    PansemiticWord,
    ReconstructedSemProWord,
    ReconstructionError,
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
    # Deverbal templates (non-concatenative): the lexeme is reduced to its verb
    # root in deep_parse, and a shared category re-applies the pansemitic
    # root-and-pattern melody (PansemiticMorphology.DEVERBAL_TEMPLATES).
    ACTIVE_PARTICIPLE = "active-participle"
    PASSIVE_PARTICIPLE = "passive-participle"
    VERBAL_NOUN = "verbal-noun"
    INSTANCE_NOUN = "instance-noun"


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


class DeverbalCategory(Enum):
    """Templatic deverbal category of a lexeme (non-concatenative, so detected
    from kaikki metadata rather than surface stripping).  A plain finite verb
    carries none of these."""
    ACTIVE_PARTICIPLE = "active-participle"
    PASSIVE_PARTICIPLE = "passive-participle"
    VERBAL_NOUN = "verbal-noun"
    INSTANCE_NOUN = "instance-noun"


# Priority among co-occurring templatic categories (a lexeme usually carries
# exactly one).
_DEVERBAL_PRIORITY = (
    DeverbalCategory.ACTIVE_PARTICIPLE,
    DeverbalCategory.PASSIVE_PARTICIPLE,
    DeverbalCategory.VERBAL_NOUN,
    DeverbalCategory.INSTANCE_NOUN,
)

# A deverbal lexeme is reduced to its verb root in deep_parse and the category
# carried as a re-appliable Layer (so a *shared* category re-applies the
# pansemitic template, and an asymmetric one is dropped like any other layer).
_DERIVATION_LAYER: dict[DeverbalCategory, Layer] = {
    DeverbalCategory.ACTIVE_PARTICIPLE: Layer.ACTIVE_PARTICIPLE,
    DeverbalCategory.PASSIVE_PARTICIPLE: Layer.PASSIVE_PARTICIPLE,
    DeverbalCategory.VERBAL_NOUN: Layer.VERBAL_NOUN,
    DeverbalCategory.INSTANCE_NOUN: Layer.INSTANCE_NOUN,
}


# ── Typed derivational model (Base + ordered POS-typed Transforms) ────────
# A word is a BASE plus an ordered list of TRANSFORMS; the pansemitic form is
# produced by realizing the base and applying each transform in canonical
# morphotactic order (base → stem → deverbal → gender → nisba → number →
# state).  POS PRECONDITIONS are enforced at reconciliation: a transform whose
# in_pos isn't met by the running POS is dropped — so a feminine left stranded
# on a bare verb base (its licensing nominalization asymmetric, hence dropped)
# is dropped too.  This is what kills the doubled-`aa` feminine bug (مَحَبَّة →
# xabba, not xabbaa).  Realization reuses the existing melody/affix tables
# (_STEM_LAYERS gemination, DEVERBAL_TEMPLATES, NOUN_PATTERNS, _wrap_affixes).


class Pos(Enum):
    VERB = "verb"
    NOUN = "noun"
    ADJ = "adj"


# The nominal POSes — the accepted-input set for the concatenative transforms
# (feminine/plural apply to a noun OR an adjective; see Transform.in_pos).
_NOMINAL = frozenset({Pos.NOUN, Pos.ADJ})


@dataclass(frozen=True)
class Transform:
    """One POS-typed derivational step.

    ``kind`` ∈ {"stem", "deverbal", "concatenative"}.  ``in_pos`` is the set of
    input POSes the step accepts (the precondition); ``out_pos`` is the result
    POS, or None to preserve the input POS (feminine/plural/definite).  The
    realization payload is kind-specific: a *deverbal* carries a ``melody`` that
    re-templates the base's radicals; a *stem* carries ``geminates`` (D exponent
    on C2) and/or a wrap ``prefix`` (Š ša-, N na-, tD ta-); a *concatenative*
    carries the ``layer`` consumed by PansemiticMorphology._wrap_affixes."""
    name: str
    kind: str
    in_pos: frozenset[Pos]
    out_pos: Pos | None = None
    melody: str | None = None
    geminates: bool = False
    prefix: str = ""
    layer: Layer | None = None

    def out(self, pos: Pos) -> Pos:
        return self.out_pos if self.out_pos is not None else pos


# ── Base variants ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VerbalRoot:
    """A root realized as its vocalized G verb (the reconciled pansemitic stem);
    stem/deverbal transforms re-slot/re-template it.  base_pos = VERB.

    ``surface`` is the pre-reduction nominal stem of a DEVERBAL lexeme (a maṣdar
    or participle before it was de-derived to its verb) — used as the fallback
    when its deverbal category turns out to be asymmetric and is dropped, so it
    reconstructs as a noun instead of the bare verb (see merge)."""
    stem: str
    radicals: tuple[str, ...]
    surface: str = ""
    base_pos: ClassVar[Pos] = Pos.VERB

    def realize(self) -> str:
        return self.stem


@dataclass(frozen=True)
class PrimaryNominal:
    """A noun/adjective base.  With a catalogued ``melody`` (segholate, qatal,
    qatīl, maqtal place/instrument, qattāl agent) it is synthesized from the
    root + that pattern's compromise melody; without one it is the surface-
    aligned stem.  A STOP POINT — never de-derived to a fabricated verb."""
    pos: Pos
    stem: str = ""
    radicals: tuple[str, ...] = ()
    melody: str | None = None
    psid: str | None = None

    @property
    def base_pos(self) -> Pos:
        return self.pos

    def realize(self) -> str:
        if self.melody is not None and len(self.radicals) == 3:
            return self.melody.format(*self.radicals)
        return self.stem


@dataclass(frozen=True)
class Loan:
    """A borrowed noun/adjective, no Semitic root (golf, kinnar).  Realized as
    its stem; only concatenative transforms may apply."""
    stem: str
    pos: Pos

    @property
    def base_pos(self) -> Pos:
        return self.pos

    def realize(self) -> str:
        return self.stem


@dataclass
class Derivation:
    """A BASE plus its ordered, precondition-valid TRANSFORMS (canonical order)."""
    base: VerbalRoot | PrimaryNominal | Loan
    transforms: tuple[Transform, ...] = ()


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
    roots: frozenset[str] = frozenset()        # consonantal root tag(s)
    verb_forms: frozenset[str] = frozenset()   # ar form (I..X) / he binyan (raw)
    verb_stem: frozenset[str] = frozenset()    # Proto-Semitic stem(s): G, D, Š, …
    derivation: frozenset[DeverbalCategory] = frozenset()  # templatic deverbal categories
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
    merged stem is reduced to the pansemitic inventory — by realize(), so the
    compromise affixes (hal-, -i, -im/-at, -a) live in pansemitic phonology
    rather than being glued onto the proto-Semitic ancestor.

    ar_ipa/he_ipa are the morphology-stripped stems in phoneme space, ready to
    align and reconstruct.  ``nominal`` is set when both sides reflect a shared
    catalogued noun pattern: the pansemitic form is then synthesized from the
    root + that pattern's compromise melody instead of from the aligned stem.

    ``layers`` are the RECONCILED shared morphemes — the two per-side chains
    aligned and precondition-filtered (see _reconcile) — so the merged
    Derivation only carries transforms the shared base actually licenses.
    ``base_pos`` is that shared base's POS (VERB, or NOUN for a primary
    noun/adjective pair), which the realization uses to build the Base variant
    and which the precondition filter walked from."""
    ar_ipa: str
    he_ipa: str
    layers: set[Layer] = field(default_factory=set)
    nominal: NominalSynthesis | None = None
    base_pos: Pos = Pos.VERB
    # Set for a single-word finite-verb pair: the reconstructed root, used to
    # regenerate a root-VISIBLE G base so weak/geminate roots reconstruct as
    # qawama/θamama (not qām/θām) and the ancestor's consonants equal proto_root.
    proto_root: "ProtoRoot | None" = None
    verb_base: bool = False
    # The deverbal category that was dropped when a maṣdar/participle reverted to
    # its nominal surface — used to regenerate a root-VISIBLE form for a WEAK root
    # whose glide was lost in surface reconstruction (زَانِيَة znj → zaːnij, not zana).
    reverted_deverbal: "Layer | None" = None


@dataclass
class WordTrace:
    """One aligned word's deep-parse result, for the merge trace: its bare stem
    and the names of the layers stripped from it."""
    stem: str
    layers: list[str]


@dataclass
class MergeTrace:
    """The normalization trace for one aligned word pair (surfaced in the JSON):
    each side's stem + stripped layers, the merged stem with the shared layers
    re-applied, and the produced pansemitic word."""
    ar: WordTrace
    he: WordTrace
    merged_stem: str            # merged pansemitic stem, before affixes
    applied_layers: list[str]   # the shared layers re-applied
    final: str                  # produced pansemitic word for this component


@dataclass
class ProtoRoot:
    """The de-patterned, jointly-reconstructed Proto-Semitic root of a pair —
    the canonical index label, computed once as a co-output of the merge.

    ``ipa`` is the consonantal grouping key (e.g. ``ħdθ``); ``label`` is the
    scholarly Proto-Semitic rendering (``ḥdṯ``).  The root is purely
    consonantal — قَدَم/قَدِيم/قَدَّمَ all collapse to ``qdm``.  ``ar_pattern`` /
    ``he_pattern`` record each side's nominal melody (wazn / mishqal) as an IPA
    template with the radicals slotted out (``{0}{1}{2}``), e.g. maḵzan →
    ``ma{0}{1}a{2}`` — the full VOCALIC melody, so Phase 2 can regenerate
    qadam vs qidam vs qadim.  ``radicals`` are the reconstructed radicals in the
    RECON-SEM-PRO inventory (= consonant_skeleton of the merged root, so
    ``"".join(radicals) == ipa``); ``pan_radicals`` are the same in the
    PANSEMITIC inventory (ħ→x, interdentals merged).  Both are complete even for
    weak/geminate roots since the root tag is authoritative (q-w-m → ``q w m``,
    r-b-b → ``r b b``): Phase-2 nominal synthesis slots pan_radicals into a
    shared pattern's melody, and the generative G verb base slots radicals /
    pan_radicals into *qatala (qawama) so the ancestor is root-visible."""
    ipa: str
    label: str
    ar_pattern: str | None = None
    he_pattern: str | None = None
    radicals: list[str] = field(default_factory=list)      # recon-sem-pro inventory
    pan_radicals: list[str] = field(default_factory=list)  # pansemitic inventory


@dataclass
class MergeResult:
    """The reconstructed ancestor of a merged cognate pair.

    ``ancestor`` is the bare merged proto-Semitic stem; ``pansemitic`` is that
    stem reduced to the pansemitic inventory with shared morphology re-applied.
    ``trace`` is the per-word normalization audit.  verb_stem/derivation carry
    the ancestor's verbal analysis when both sides share a stem/deverbal category;
    None otherwise.  ``proto_root`` is the canonical de-patterned root (a single-
    word lexeme property; None for multi-word compounds)."""
    ancestor: Word
    pansemitic: PansemiticWord
    trace: list[MergeTrace]
    verb_stem: str | None = None
    derivation: str | None = None
    proto_root: ProtoRoot | None = None
    # Compact `+`-joined recipe for the pansemitic form (see derivation_label):
    # base (verb stem / noun pattern / n) then surviving transforms, space-joined
    # across the words of a compound.
    derivation_chain: str | None = None


# ── Proto-root analysis ──────────────────────────────────────────────────
# The morphology layer owns a SINGLE root computation (no second parallel
# reconstruction): given each side's root tag and best stem IPA, it reduces both
# sides to their bare radicals — the root TAG is the authority, complete even for
# weak (ق-و-م) and geminate (ر-ب-ب) roots; a tagless side is de-patterned by
# ALIGNING the other (tagged) side's radicals to its surface and keeping the
# matched skeleton, dropping the templatic residue — then jointly reconstructs
# them, reconciling the regular correspondences (Arabic's conservative phonology
# breaks Hebrew's mergers).  The merge emits the result; the shared-source path
# (which bypasses the merge) calls the same function with surface IPA.

# Modifier letters stripped to a radical's base for alignment/equality.
_RADICAL_MODIFIERS = re.compile(r"[ːˈˤʰʲʷˠ]")
_ROOT_VOWELS = re.compile(r"[aeiou]")

# Correspondence classes letting an authoritative radical (conservative IPA, from
# a tag or the opposite side) align to a drifted surface consonant — Modern
# Hebrew spirantization/mergers (b→v, p→f, k/ħ→x, r→ʁ), the sibilant/interdental
# correspondences, the guttural merges.  Alignment is order-preserving, so edge
# formatives (preformative m-, suffix -ān/-īm) never match an interior radical
# and fall out as the melody residue.
_RADICAL_EQUIV: tuple[frozenset[str], ...] = (
    frozenset({"b", "v", "w"}),
    frozenset({"p", "f"}),
    frozenset({"k", "x", "χ", "q"}),
    frozenset({"ħ", "x", "χ", "h"}),
    frozenset({"ʕ", "ʔ", "ɣ", "ʁ"}),
    frozenset({"r", "ʁ", "ʀ"}),
    frozenset({"sˤ", "ts", "t͡s", "s"}),
    frozenset({"dˤ", "sˤ", "d"}),
    frozenset({"tˤ", "t"}),
    frozenset({"ðˤ", "sˤ", "z"}),
    frozenset({"θ", "t", "ʃ", "s"}),
    frozenset({"ð", "d", "z"}),
    frozenset({"z", "s"}),
    frozenset({"ɬ", "ʃ", "s"}),
    frozenset({"ʃ", "s"}),
)

_ROOT_RADICALS = {"ar": arabic_root_radicals, "he": hebrew_root_radicals}


@dataclass
class _SideRoot:
    """One side's reconstruction radicals plus its recorded nominal melody."""
    radicals: list[str]
    pattern: str | None


def _radical_base(tok: str) -> str:
    return _RADICAL_MODIFIERS.sub("", tok)


def _radicals_compatible(a: str, b: str) -> bool:
    a, b = _radical_base(a), _radical_base(b)
    if a == b:
        return True
    return any({a, b} <= cls for cls in _RADICAL_EQUIV)


def _align_radicals(radicals: list[str], ipa: str) -> list[int] | None:
    """Indices (into Phoneme.parse(ipa)) of the consonant tokens that *radicals*
    align to, as an order-preserving fuzzy subsequence; None if any radical has
    no compatible surface consonant."""
    toks = list(Phoneme.parse(ipa))
    cons = [(i, t.tok) for i, t in enumerate(toks) if isinstance(t, Consonant)]
    idxs: list[int] = []
    ci = 0
    for r in radicals:
        j = ci
        while j < len(cons) and not _radicals_compatible(r, cons[j][1]):
            j += 1
        if j >= len(cons):
            return None
        idxs.append(cons[j][0])
        ci = j + 1
    return idxs


def _melody(ipa: str, idxs: list[int]) -> str:
    """Build a nominal-melody template: the surface IPA with the radical tokens
    at *idxs* replaced by slots ({0}{1}{2}), everything else (vowels and any
    templatic affix consonants) kept verbatim — the full vocalic melody."""
    toks = list(Phoneme.parse(ipa))
    slot = {pos: k for k, pos in enumerate(idxs)}
    return "".join(f"{{{slot[i]}}}" if i in slot else t.tok
                   for i, t in enumerate(toks))


def _resolve_side(
    tag_radicals: list[str] | None, ipa: str, guide: list[str] | None,
) -> _SideRoot:
    """Reduce one side to its reconstruction radicals + recorded melody.

    The tag is authoritative: its radicals drive the reconstruction, and the
    melody is recovered by aligning them to the surface.  A tagless side falls
    back to its surface skeleton, but when the OTHER side is tagged it is
    de-patterned by aligning that tag's radicals to this surface and keeping the
    matched consonants (dropping preformatives/suffixes as melody residue)."""
    if tag_radicals is not None:
        idxs = _align_radicals(tag_radicals, ipa)
        return _SideRoot(tag_radicals, _melody(ipa, idxs) if idxs else None)
    skeleton = consonant_skeleton(ipa)
    if guide is not None and len(skeleton) > len(guide):
        idxs = _align_radicals(guide, ipa)
        if idxs is not None:
            toks = list(Phoneme.parse(ipa))
            radicals = [_RADICAL_MODIFIERS.sub("", toks[i].tok).replace("ː", "")
                        for i in idxs]
            return _SideRoot(radicals, _melody(ipa, idxs))
    # A surface geminate collapses the doubled final radical (gārar → g-r), but a
    # geminate GUIDE (the tagged side's r-b-b) shows the root has it; restore the
    # duplicate so both sides feed the same radical count and key consistently.
    if (guide is not None and len(guide) >= 2 and guide[-1] == guide[-2]
            and len(skeleton) == len(guide) - 1 and skeleton):
        return _SideRoot(skeleton + [skeleton[-1]], None)
    idxs = _align_radicals(skeleton, ipa)
    return _SideRoot(skeleton, _melody(ipa, idxs) if idxs else None)


def reconstruct_proto_root(
    ar_tag: str | None, ar_ipa: str,
    he_tag: str | None, he_ipa: str,
) -> ProtoRoot | None:
    """Jointly reconstruct a pair's de-patterned Proto-Semitic root from each
    side's root tag (authority) and best stem IPA, recording each side's nominal
    melody.  None when either side has no radicals or reconstruction fails."""
    ar_tag_rad = _ROOT_RADICALS["ar"](ar_tag) if ar_tag else None
    he_tag_rad = _ROOT_RADICALS["he"](he_tag) if he_tag else None
    ar = _resolve_side(ar_tag_rad, ar_ipa, guide=he_tag_rad)
    he = _resolve_side(he_tag_rad, he_ipa, guide=ar_tag_rad)
    if not ar.radicals or not he.radicals:
        return None
    try:
        merged = reconstruct_from_words(
            ArabicWord.from_ipa("a".join(ar.radicals)),
            HebrewWord.from_ipa("a".join(he.radicals)),
        )
    except ReconstructionError:
        return None
    recon_radicals = consonant_skeleton(merged.word)
    key = "".join(recon_radicals)
    if not key:
        return None
    # The reconstructed SemPro word keeps the base scholar table (ħ→ḥ, θ→ṯ),
    # unlike PansemiticWord which merges them — so this is the real PS root.
    label = _ROOT_VOWELS.sub("", merged.to_protosemitic_convention())
    # The same radicals in the pansemitic inventory (ħ→x, interdentals merged),
    # so Phase-2 nominal synthesis slots pansemitic-shaped radicals into a
    # compromise melody, consistent with the rest of the pansemitic form.
    pan_radicals = consonant_skeleton(PansemiticWord.from_word(merged).word)
    return ProtoRoot(ipa=key, label=label,
                     ar_pattern=ar.pattern, he_pattern=he.pattern,
                     radicals=recon_radicals, pan_radicals=pan_radicals)


def select_root_tag(tags: Iterable[str], ipa: str, lang: str) -> str | None:
    """Deterministically choose a lexeme's root tag for de-patterning.

    A lemma can carry SEVERAL root tags (e.g. قَرْيَة → قرر/قرو/قري); picking one by
    set-iteration order is nondeterministic AND arbitrary.  Prefer the tag whose
    radicals actually ALIGN to the surface (قري aligns to qarja; the geminate قرر
    does not) — stable and more correct than either an arbitrary pick or blind
    alphabetical order, which here would wrongly take the geminate.  Falls back to
    the sorted-first tag when none align, so the result is always determined."""
    radicals_fn = _ROOT_RADICALS.get(lang)
    ordered = sorted(tags)
    if not ordered or radicals_fn is None:
        return ordered[0] if ordered else None
    for tag in ordered:
        rad = radicals_fn(tag)
        if rad and _align_radicals(rad, ipa) is not None:
            return tag
    return ordered[0]


def select_verb_stem(stems: Iterable[str]) -> str | None:
    """Deterministically choose a lexeme's Proto-Semitic verb stem when kaikki
    tags several binyan forms (e.g. آكَلَ = form I 'eat' + form IV 'feed' → {G, Š},
    otherwise picked by set order).  Prefer the G base — the conservative reading
    that doesn't assert a derivational exponent from an ambiguous lemma — else the
    sorted-first stem."""
    stems = set(stems)
    if not stems:
        return None
    return "G" if "G" in stems else sorted(stems)[0]


# Looks up the (canonical, romanization) of a base lexeme in the caller's
# word index, given (lang, normalized base candidates, preferred POS set —
# so a nominal base beats a verb homograph and vice-versa).
BaseLookup = Callable[[str, frozenset[str], frozenset[str]], tuple[str, str] | None]

# Looks up a component word's lexeme-level metadata (the analyze_phrase kwargs:
# pos, verb_forms, derivation, number, gender, derived_from, singular_of,
# masculine_of) by (lang, script token), so multi-word phrases can be dispatched
# component-by-component with each component's own metadata.  None when the
# component is not a known lemma (→ concatenative-only fallback).
MetaLookup = Callable[[str, str], dict | None]

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
    base_verb_forms = frozenset({"pa"})
    # Binyan → Proto-Semitic stem.  Common binyanim only; rare ones (poal,
    # hitpuʕal, nitpaʕel) are left raw.
    verb_stem_map = {
        "pa": "G", "qal": "G", "nif": "N", "ni": "N", "pi": "D", "pu": "Dp",
        "hif": "Š", "hi": "Š", "huf": "Šp", "ho": "Šp", "hit": "tD",
    }

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
        the next letter, or הָ before a guttural (which cannot take dagesh).

        The segol form הֶ is deliberately NOT accepted: before a guttural it is
        overwhelmingly the hifʕil prefix (הֶאָרָה, הֶרְגֵּל) or part of a loan
        (הֶרְץ "Hertz"), not the article — the article's reliable signatures are
        gemination (dagesh forte) and qamats compensatory lengthening."""
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
        if vowel == cls._QAMATS:
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


# ── Proto-Semitic noun-pattern catalog (Phase 2) ─────────────────────────
# Primary NOUNS are brought up to the verb standard: when BOTH sides' recorded
# nominal melodies reflect the SAME Proto-Semitic noun pattern, the pansemitic
# form is re-synthesized from the root + that pattern's single compromise melody
# (mirroring how the binyan is re-applied for verbs), rather than left to the
# surface aligner.  An asymmetric pair (no shared catalogued pattern) keeps the
# surface-aligned form.
#
# Matching is by melody string: a pair reflects a pattern iff its ar_pattern is
# among the pattern's ``ar`` melodies AND its he_pattern among the ``he`` ones.
# The Arabic melody is the conservative witness — it preserves the segholate
# vowel that Hebrew neutralizes to qetel (qatl/qitl/qutl all surface as the same
# he ``{0}e{1}e{2}``) — so it disambiguates which segholate, while the shared he
# melody only confirms membership.  Melodies are the radical-slotted templates
# reconstruct_proto_root records; the compromise is the IPA melody the (complete,
# tag-authoritative) pan_radicals slot into, so weak/geminate roots realize by
# literal slotting (q-w-m + qatl → qawm, ħ-b-b + qatal → xabab).


@dataclass(frozen=True)
class NounPattern:
    """One Proto-Semitic nominal pattern: its scholarly id, the single pansemitic
    compromise melody (IPA, ``{0}{1}{2}`` radical slots), and the per-language
    surface melodies that reflect it (matched against ProtoRoot.ar_pattern /
    he_pattern)."""
    psid: str
    pansemitic: str
    ar: frozenset[str]
    he: frozenset[str]


# Hebrew segholates collapse *qatl/*qitl/*qutl to qetel (``{0}e{1}e{2}``), so the
# same he melody appears under all three; the Arabic vowel picks the entry.
_HE_SEGHOLATE = frozenset({"{0}e{1}e{2}", "{0}a{1}a{2}", "{0}i{1}{2}",
                           "{0}o{1}e{2}", "{0}e{1}a{2}"})

NOUN_PATTERNS: tuple[NounPattern, ...] = (
    # Segholates — Arabic faʿl/fiʿl/fuʿl keep the proto vowel; pansemitic keeps
    # the conservative CVCC cluster (no Hebrew epenthesis).
    NounPattern("qatl", "{0}a{1}{2}", frozenset({"{0}a{1}{2}"}), _HE_SEGHOLATE),
    NounPattern("qitl", "{0}i{1}{2}", frozenset({"{0}i{1}{2}"}), _HE_SEGHOLATE),
    NounPattern("qutl", "{0}u{1}{2}", frozenset({"{0}u{1}{2}"}), _HE_SEGHOLATE),
    # Base triliteral noun *qatal (ar faʿal ↔ he qatal).
    NounPattern("qatal", "{0}a{1}a{2}",
                frozenset({"{0}a{1}a{2}"}), frozenset({"{0}a{1}a{2}"})),
    # Adjective *qatīl (ar faʿīl ↔ he qātīl); pansemitic keeps the long ī.
    NounPattern("qatiːl", "{0}a{1}iː{2}",
                frozenset({"{0}a{1}iː{2}"}), frozenset({"{0}a{1}i{2}"})),
    # Place/instrument *maqtal — only when BOTH sides carry the m-preformative
    # (a half-with-m pair is asymmetric → falls back).
    NounPattern("maqtal", "ma{0}{1}a{2}",
                frozenset({"ma{0}{1}a{2}", "mi{0}{1}aː{2}", "mi{0}{1}a{2}"}),
                frozenset({"ma{0}{1}e{2}", "mi{0}{1}a{2}", "ma{0}{1}a{2}"})),
)

# *qattāl intensive/agent (ar faʿʿāl) is detected from the ARABIC SURFACE, not the
# recorded melody: the radical-alignment that builds ar_pattern absorbs the
# doubled C2, and Modern Hebrew loses gemination too, so neither melody witnesses
# it.  This is the one Arabic-WITNESSED (not bilaterally-shared) pattern — it
# fires on Arabic geminate evidence with Hebrew only required to be a plausible
# triconsonantal cognate.  Compromise: faʿʿāl with the geminate C2 + long ā, a
# morphology-only exponent like the D-stem gemination.
_QATTAL_PANSEMITIC = "{0}a{1}ːaː{2}"


def match_noun_pattern(ar_pattern: str | None,
                       he_pattern: str | None) -> NounPattern | None:
    """The catalogued PS noun pattern a pair reflects, or None.  Shared = the
    ar melody is among a pattern's ar melodies AND the he melody among its he
    melodies (first match; the catalog's ar melodies are mutually exclusive)."""
    if not ar_pattern or not he_pattern:
        return None
    for pat in NOUN_PATTERNS:
        if ar_pattern in pat.ar and he_pattern in pat.he:
            return pat
    return None


def arabic_intensive(ipa: str) -> bool:
    """Whether an Arabic surface stem has the faʿʿāl shape: a triconsonantal
    core whose second radical is geminated and is followed by a long ā (the
    reliable witness of *qattāl, since the recorded melody absorbs the
    gemination)."""
    toks = list(Phoneme.parse(ipa))
    cons = [i for i, t in enumerate(toks) if isinstance(t, Consonant)]
    if len(cons) != 3:
        return False
    if "ː" not in toks[cons[1]].tok:
        return False
    return any(t.tok == "aː" for t in toks[cons[1] + 1:cons[2]])


@dataclass(frozen=True)
class NominalSynthesis:
    """A resolved nominal pattern ready to produce: the compromise melody and the
    (pansemitic-inventory) radicals to slot into it, plus the PS id for the
    trace/report."""
    psid: str
    melody: str
    radicals: tuple[str, ...]


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

    # Pansemitic deverbal melodies, applied root-and-pattern to the merged
    # triliteral root (a shared deverbal lexeme is reduced to its verb root in
    # deep_parse, then re-templatized here).  PROPOSED conventions — this dict is
    # the single place the deverbal surface shape is defined, so adjust freely.
    # IPA format strings slotting the three root consonants ({0}{1}{2}); ː = long.
    DEVERBAL_TEMPLATES: ClassVar[dict[Layer, str]] = {
        Layer.ACTIVE_PARTICIPLE: "{0}aː{1}i{2}",   # *kaːtib (qātil / fāʕil ~ qōtēl)
        Layer.PASSIVE_PARTICIPLE: "{0}a{1}uː{2}",  # *katuːb (qatūl, common to ar/he)
        Layer.VERBAL_NOUN: "{0}a{1}aː{2}",         # *kataːb (neutral maṣdar qatāl)
        Layer.INSTANCE_NOUN: "{0}a{1}{2}a",        # *katla (faʕla nomen vicis, fem.)
    }
    _DEVERBAL_ORDER = (Layer.ACTIVE_PARTICIPLE, Layer.PASSIVE_PARTICIPLE,
                       Layer.VERBAL_NOUN, Layer.INSTANCE_NOUN)

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
    def _wrap_affixes(cls, stem: str, layers: set[Layer]) -> str:
        """Wrap a produced *stem* with the shared concatenative affixes: definite
        article, verbal-stem prefix, and the single nominal suffix.  Number
        (dual/plural) outranks nisba and bare feminine for that suffix slot, and a
        feminine stem takes -at rather than -im."""
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


# ── Transform catalog & realization ──────────────────────────────────────
# The catalog names every derivational step as a POS-typed Transform, with its
# realization drawn from the existing melody/affix tables (so realize() below
# subsumes the old produce_word + produce_nominal + apply_verb_stem_ipa paths
# without changing any melody).  Stem exponents and deverbal melodies are keyed
# to their Layer, so a reconciled shared-Layer set maps straight to transforms.

# Atomic verbal-stem exponents (one Transform per exponent Layer).  A Semiticist
# stem is a SET of exponents (D = {gemination}, tD = {gemination, ta-}), so
# representing them atomically makes reconciling two chains by transform NAME
# equal to reconciling their Layer sets — essential because a pair can share a
# SUB-exponent (ar tD تَقَدَّمَ / he D קִדֵּם → shared gemination → D → qaddama).
_STEM_EXPONENTS: dict[Layer, Transform] = {
    Layer.GEMINATION: Transform("gemination", "stem", frozenset({Pos.VERB}),
                                Pos.VERB, geminates=True, layer=Layer.GEMINATION),
    Layer.CAUSATIVE: Transform(
        "causative", "stem", frozenset({Pos.VERB}), Pos.VERB,
        prefix=PansemiticMorphology.stem_prefixes[Layer.CAUSATIVE],
        layer=Layer.CAUSATIVE),
    Layer.N_PREFIX: Transform(
        "n_prefix", "stem", frozenset({Pos.VERB}), Pos.VERB,
        prefix=PansemiticMorphology.stem_prefixes[Layer.N_PREFIX],
        layer=Layer.N_PREFIX),
    Layer.T_PREFIX: Transform(
        "t_prefix", "stem", frozenset({Pos.VERB}), Pos.VERB,
        prefix=PansemiticMorphology.stem_prefixes[Layer.T_PREFIX],
        layer=Layer.T_PREFIX),
}
# Gemination before the prefixes, so tD geminates C2 then prefixes ta-.
_STEM_EXPONENT_ORDER = (Layer.GEMINATION, Layer.CAUSATIVE, Layer.N_PREFIX,
                        Layer.T_PREFIX)
_STEM_LAYER_SET = frozenset(_STEM_EXPONENTS)
# The deverbal-category layers (participle/maṣdar/instance) — their presence on
# EITHER side means the feminine/etc. rides a deverbal noun, so the shared base
# is verbal (and a stranded affix drops when the deverbal isn't shared).
_DEVERBAL_LAYER_SET = frozenset(_DERIVATION_LAYER.values())

_DEV = PansemiticMorphology.DEVERBAL_TEMPLATES
_DEVERBAL_ORDER = PansemiticMorphology._DEVERBAL_ORDER
# Deverbal Layer → Transform (verb → noun/adj).  Participles are adjectival;
# the maṣdar/instance nouns are nominal.
_DEVERBAL_TRANSFORMS: dict[Layer, Transform] = {
    Layer.ACTIVE_PARTICIPLE: Transform(
        "active_participle", "deverbal", frozenset({Pos.VERB}), Pos.ADJ,
        melody=_DEV[Layer.ACTIVE_PARTICIPLE], layer=Layer.ACTIVE_PARTICIPLE),
    Layer.PASSIVE_PARTICIPLE: Transform(
        "passive_participle", "deverbal", frozenset({Pos.VERB}), Pos.ADJ,
        melody=_DEV[Layer.PASSIVE_PARTICIPLE], layer=Layer.PASSIVE_PARTICIPLE),
    Layer.VERBAL_NOUN: Transform(
        "verbal_noun", "deverbal", frozenset({Pos.VERB}), Pos.NOUN,
        melody=_DEV[Layer.VERBAL_NOUN], layer=Layer.VERBAL_NOUN),
    Layer.INSTANCE_NOUN: Transform(
        "instance_noun", "deverbal", frozenset({Pos.VERB}), Pos.NOUN,
        melody=_DEV[Layer.INSTANCE_NOUN], layer=Layer.INSTANCE_NOUN),
}

# Concatenative Layer → Transform (nominal → nominal/adj).  DUAL is collapsed
# into PLURAL by deep_parse, so only PLURAL appears here.
_CONCAT_TRANSFORMS: dict[Layer, Transform] = {
    Layer.FEMININE: Transform("feminine", "concatenative", _NOMINAL, None,
                              layer=Layer.FEMININE),
    Layer.NISBA: Transform("nisba", "concatenative", frozenset({Pos.NOUN}),
                           Pos.ADJ, layer=Layer.NISBA),
    Layer.PLURAL: Transform("plural", "concatenative", _NOMINAL, None,
                            layer=Layer.PLURAL),
    Layer.DEFINITE: Transform("definite", "concatenative", _NOMINAL, None,
                              layer=Layer.DEFINITE),
}
# Canonical order of the concatenative slot (gender → nisba → number → state).
_CONCAT_ORDER = (Layer.FEMININE, Layer.NISBA, Layer.PLURAL, Layer.DEFINITE)

# Compact labels for the human-readable derivation chain (see derivation_label):
# the verb stem set → its name (G/D/Š/N/tD), and each deverbal/concatenative
# Layer → a short tag.
_STEM_LABEL: dict[frozenset[Layer], str] = {v: k for k, v in _STEM_LAYERS.items()}
_DEVERBAL_LABEL = {Layer.ACTIVE_PARTICIPLE: "pcpl.act",
                   Layer.PASSIVE_PARTICIPLE: "pcpl.pass",
                   Layer.VERBAL_NOUN: "vnoun", Layer.INSTANCE_NOUN: "inst"}
_CONCAT_LABEL = {Layer.FEMININE: "fem", Layer.NISBA: "nisba",
                 Layer.PLURAL: "pl", Layer.DEFINITE: "def"}


def derivation_label(pair: "PlannedPair") -> str:
    """A compact recipe for how the pansemitic form was built: the BASE (verb
    stem G/D/Š/N/tD, a catalogued noun pattern like qatl/maqtal/qattaːl, or `n`
    for a bare surface nominal) then each surviving transform, `+`-joined —
    e.g. `G`, `D`, `maqtal`, `G+vnoun`, `qatl+fem+pl`.

    Reflects the RECONCILED shared derivation, so an asymmetric layer that was
    dropped doesn't appear.  A noun whose maṣdar was de-derived to its verb and
    then dropped shows as `G` — the signal that it fell back to the bare verb."""
    layers = pair.layers
    if pair.nominal is not None:
        base = pair.nominal.psid
    elif pair.base_pos is Pos.VERB or pair.verb_base:
        base = _STEM_LABEL.get(frozenset(layers & _STEM_LAYER_SET), "G")
    else:
        base = "n"
    parts = [base]
    dv = next((L for L in _DEVERBAL_ORDER if L in layers), None)
    if dv is not None:
        parts.append(_DEVERBAL_LABEL[dv])
    parts.extend(_CONCAT_LABEL[L] for L in _CONCAT_ORDER if L in layers)
    return "+".join(parts)


def _chain_from_layers(layers: set[Layer]) -> tuple[Transform, ...]:
    """The canonical-order Transform chain for a stripped layer set (atomic stem
    exponents, then the ≤1 deverbal, then the concatenatives)."""
    out: list[Transform] = [_STEM_EXPONENTS[L] for L in _STEM_EXPONENT_ORDER
                            if L in layers]
    dv = next((L for L in _DEVERBAL_ORDER if L in layers), None)
    if dv is not None:
        out.append(_DEVERBAL_TRANSFORMS[dv])
    out.extend(_CONCAT_TRANSFORMS[L] for L in _CONCAT_ORDER if L in layers)
    return tuple(out)


def _chain_layers(chain: tuple[Transform, ...]) -> set[Layer]:
    """The Layer set a chain carries (every Transform maps to exactly one Layer)."""
    return {t.layer for t in chain if t.layer is not None}


def _dropped_deverbal(deriv: Derivation, shared: set[Layer]) -> Layer | None:
    """The deverbal category this side carried that the reconciliation dropped
    (asymmetric), or None — the signal to revert it to its nominal surface (and,
    for a weak root, to regenerate root-visibly with that category's template)."""
    for t in deriv.transforms:
        if t.kind == "deverbal" and t.layer is not None and t.layer not in shared:
            return t.layer
    return None


def _reconcile(ar_chain: tuple[Transform, ...],
               he_chain: tuple[Transform, ...]) -> tuple[Pos, set[Layer]]:
    """Align two per-side Transform chains into the shared, precondition-valid
    layer set (Stage B — replaces the layer-intersection + filter).

    Keeps transforms present in BOTH chains (by name), in canonical order,
    walking a running POS from the shared base POS and dropping any whose in_pos
    isn't met — so a concatenative can't attach while the running POS is still
    VERB (a stranded feminine drops, the doubled-`aa` fix falls out of the walk).
    The base is VERBAL when a deverbal is SHARED (both chains) or a stem exponent
    is shared; else a primary NOMINAL.  An ASYMMETRIC deverbal does NOT force a
    verb base — the deverbal side reverts to its nominal surface in merge, so a
    maṣdar/participle paired with a plain noun stays nominal (ذِكْر/זֵכֶר → zikr,
    not the bare verb zakara)."""
    he_names = {t.name for t in he_chain}
    shared = [t for t in ar_chain if t.name in he_names]   # ar order is canonical
    shared_deverbal = any(t.kind == "deverbal" for t in shared)
    shared_stem = any(t.kind == "stem" for t in shared)
    base_pos = Pos.VERB if (shared_deverbal or shared_stem) else Pos.NOUN
    pos = base_pos
    kept: set[Layer] = set()
    for t in shared:
        if pos in t.in_pos:
            if t.layer is not None:
                kept.add(t.layer)
            pos = t.out(pos)
    return base_pos, kept


def _base_from_pair(pair: "PlannedPair", pan_stem: str) -> VerbalRoot | PrimaryNominal:
    """Build the reconciled shared Base from a reconstructed pansemitic stem.

    A catalogued noun pattern → a PrimaryNominal synthesized from the root +
    melody; a primary noun/adjective pair with no pattern → a surface-stem
    PrimaryNominal; otherwise a VerbalRoot whose radicals (the stem consonants)
    feed any deverbal re-templating — matching the old produce_nominal /
    produce_word radical sourcing exactly."""
    if pair.nominal is not None:
        return PrimaryNominal(pos=Pos.NOUN, radicals=pair.nominal.radicals,
                              melody=pair.nominal.melody, psid=pair.nominal.psid)
    radicals = tuple(p.tok for p in Phoneme.parse(pan_stem)
                     if isinstance(p, Consonant))
    if pair.base_pos in _NOMINAL:
        return PrimaryNominal(pos=pair.base_pos, stem=pan_stem, radicals=radicals)
    return VerbalRoot(stem=pan_stem, radicals=radicals)


def realize(deriv: Derivation) -> str:
    """Produce the pansemitic surface of a Derivation: realize the base, apply
    the ≤1 stem/deverbal re-templating, then wrap the concatenative affixes.

    Replaces produce_word (verb/deverbal stems), produce_nominal (catalogued
    noun patterns) and the bare-stem path with one pipeline; the melodies and
    the affix wrap are the same tables as before, so an unchanged Derivation
    realizes to the same string."""
    form = deriv.base.realize()
    radicals = getattr(deriv.base, "radicals", ())
    conc: set[Layer] = set()
    for t in deriv.transforms:
        if t.kind == "deverbal":
            if t.melody is not None and len(radicals) == 3:
                form = t.melody.format(*radicals)
        elif t.kind == "stem":
            if t.geminates:
                form = PansemiticMorphology._geminate_c2(form)
            if t.prefix:
                form = t.prefix + form
        elif t.layer is not None:  # concatenative
            conc.add(t.layer)
    return PansemiticMorphology._wrap_affixes(form, conc)


MORPHOLOGY_CONFIG: dict[str, type[LangMorphology]] = {
    "ar": ArabicMorphology,
    "he": HebrewMorphology,
    "pansemitic": PansemiticMorphology,
}


def morphology_for(lang: str) -> type[LangMorphology] | None:
    return MORPHOLOGY_CONFIG.get(lang)


def apply_verb_stem_ipa(stem_ipa: str, verb_stem: str) -> str:
    """Re-apply a Proto-Semitic verb-stem exponent (D gemination, Š/N/tD
    prefixes) to a bare pansemitic IPA stem — the shared-source path's
    equivalent of the layer re-application that merge does for surface pairs.
    Returns the stem unchanged for G (no exponent) or an unknown stem."""
    layers = _STEM_LAYERS.get(verb_stem)
    if not layers:
        return stem_ipa
    transforms = tuple(_STEM_EXPONENTS[L] for L in _STEM_EXPONENT_ORDER
                       if L in layers)
    radicals = tuple(p.tok for p in Phoneme.parse(stem_ipa)
                     if isinstance(p, Consonant))
    return realize(Derivation(VerbalRoot(stem=stem_ipa, radicals=radicals),
                              transforms))


def analyze_phrase(
    lang: str,
    script: str,
    roman: str,
    ipa: str = "",
    pos: Iterable[str] = (),
    roots: Iterable[str] = (),
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
        roots=frozenset(roots),
        verb_forms=verb_forms,
        verb_stem=frozenset(s for f in verb_forms
                            if (s := morph.verb_stem(f)) is not None),
        derivation=frozenset(DeverbalCategory(d) for d in derivation),
        number=number,
        gender=frozenset(gender),
        derived_from=frozenset(derived_from),
        singular_of=frozenset(singular_of),
        masculine_of=frozenset(masculine_of),
    )


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
) -> str | None:
    """Resolve *norms* to a cited lexeme's IPA via base_lookup, preferring a
    base of *prefer_pos* among homographs (the lookup returns romanization,
    converted to IPA here)."""
    if not norms:
        return None
    hit = base_lookup(lang, norms, prefer_pos)
    if hit is None:
        return None
    _canonical, roman = hit
    return _roman_to_ipa(lang, roman)


def _substitute_base(phrase: AnalyzedPhrase, base_lookup: BaseLookup) -> str | None:
    """Swap in the IPA of the derivational (noun-from-verb) base kaikki
    cites — a verb."""
    return _lookup_base_ipa(
        phrase.lang, phrase.derived_from, _VERB_BASE_POS, base_lookup)


def _reduce_to_base(
    phrase: AnalyzedPhrase,
    morph: type[LangMorphology],
    ipa: str,
    layer: Layer,
    base_norms: frozenset[str],
    base_lookup: BaseLookup,
) -> str | None:
    """Reduce an inflected side to its stem, in IPA.

    Prefers the base lexeme Wiktionary actually cites (resolved via base_lookup)
    — exact, and it reaches irregular forms the regex can't (broken plurals,
    feminines with stem-vowel changes like he malká/melekh); falls back to the
    language's IPA suffix-strip."""
    return _lookup_base_ipa(phrase.lang, base_norms, _NOMINAL_BASE_POS, base_lookup) \
        or morph.strip_ipa(ipa, layer)


def _deverbal_category(phrase: AnalyzedPhrase) -> DeverbalCategory | None:
    """The phrase's single templatic deverbal category, by priority."""
    for c in _DEVERBAL_PRIORITY:
        if c in phrase.derivation:
            return c
    return None


def _reduce_verb_stem(
    phrase: AnalyzedPhrase, ipa: str, base_lookup: BaseLookup,
) -> str | None:
    """Reduce a verb / deverbal IPA toward its G-stem base: the cited base
    lexeme first (a participle's or verbal noun's underlying verb, or a derived
    stem's form-I), then the language's stem template synthesis."""
    base = _substitute_base(phrase, base_lookup)
    if base:
        return base
    morph = MORPHOLOGY_CONFIG[phrase.lang]
    for stem in phrase.verb_stem:
        if stem == "G":
            continue
        synth = morph.synthesize_base_stem(ipa, stem)
        if synth is not None:
            return synth[0]
    return None


def _reconstruct_word_pairs(
    word_pairs: list[PlannedPair],
) -> tuple[Word, PansemiticWord, list[tuple[str, str]]]:
    """Reconstruct each aligned pair into its proto-Semitic ancestor stem and
    pansemitic reduction; multi-word ancestors are space-joined.

    Inputs are already IPA stems (morphology stripped in phoneme space).  The
    ancestor is the bare merged proto-Semitic stem; the shared morphology is
    re-attached on the pansemitic side, after the stem is reduced to the
    pansemitic inventory, so the compromise affixes (hal-, -i, -im/-at, -a) live
    in pansemitic phonology rather than being glued onto the ancestor.  Also
    returns, per pair, the (merged pansemitic stem, produced final word) for the
    merge trace."""
    anc_parts: list[str] = []
    pan_parts: list[str] = []
    steps: list[tuple[str, str]] = []
    for pair in word_pairs:
        merged = reconstruct_from_words(
            ArabicWord.from_ipa(pair.ar_ipa), HebrewWord.from_ipa(pair.he_ipa))
        anc_word = merged.word
        pan_stem = PansemiticWord.from_word(merged).word
        # For a finite verb, generate the G base from the authoritative root so
        # the ancestor is root-VISIBLE: identity when the surface already exposes
        # every radical (sound roots), else the *qatala melody over the root
        # (weak/geminate → qawama/θamama, not qām/θām).  ancestor ≡ realized base,
        # and consonant_skeleton(ancestor) == proto_root.ipa by construction.
        if pair.verb_base and pair.proto_root is not None:
            anc_word, pan_stem = _generative_verb_base(
                anc_word, pan_stem, pair.proto_root)
        elif pair.reverted_deverbal is not None and pair.proto_root is not None:
            anc_word, pan_stem = _generative_deverbal_base(
                anc_word, pan_stem, pair.proto_root, pair.reverted_deverbal)
        anc_parts.append(anc_word)
        # Build the reconciled Derivation (Base + shared, precondition-valid
        # transforms) and realize it — one pipeline for verbs, catalogued noun
        # patterns and bare stems alike.
        base = _base_from_pair(pair, pan_stem)
        final = realize(Derivation(base, _chain_from_layers(pair.layers)))
        pan_parts.append(final)
        steps.append((pan_stem, final))
    return (ReconstructedSemProWord(word=" ".join(anc_parts)),
            PansemiticWord(word=" ".join(pan_parts)), steps)


# The neutral G perfect *qatala: C1 a C2 a C3 a — the trailing -a matches the
# surface-reconstructed sound verbs (bitˤala, ʔaːkala) this generation converges.
def _gen_g(radicals: list[str]) -> str:
    return "a".join(radicals) + "a"


def _generative_verb_base(
    anc_word: str, pan_stem: str, pr: ProtoRoot,
) -> tuple[str, str]:
    """Return (ancestor, pan_stem) for a finite verb with the root made visible.

    Identity ONLY when the surface reconstruction is already the clean
    triliteral G base — exactly three consonants, all aligning to the root — so
    its reconciled thematic vowels are kept (sound G verbs: kataba, bitˤala).
    Otherwise regenerate from the authoritative radicals as the *qatala melody
    (C1aC2aC3a), which fixes BOTH deficiencies that break ancestor≡proto_root:
      - a MISSING radical (weak/geminate contraction: qwm qaːma → qawama);
      - an EXTRA consonant (a derived-stem preformative that _STEM_LAYERS doesn't
        reduce — form VIII/X ista-/-t-, passive hu-/pu- : istaðkara → ðakara).
    The ancestor uses the recon-sem-pro radicals; the pansemitic stem uses the
    pansemitic-inventory radicals.  A same-length correspondence disagreement
    (surface s vs tag sˤ, or a metathesis) still aligns at 3 consonants and is
    left to the surface — regenerating from the tag there would just relabel a
    genuine reconstruction discrepancy."""
    if len(pr.pan_radicals) != 3 or len(pr.radicals) != 3:
        return anc_word, pan_stem
    surface_cons = consonant_skeleton(pan_stem)
    if len(surface_cons) == 3 and _align_radicals(pr.pan_radicals, pan_stem) is not None:
        return anc_word, pan_stem      # clean triliteral G base: keep surface vowels
    return _gen_g(pr.radicals), _gen_g(pr.pan_radicals)


# Glide radicals — a weak root whose w/y is vocalized away in surface
# reconstruction (leaving a defective 2-consonant stem).
_GLIDES = frozenset({"w", "j"})
# Only the PARTICIPLES are regenerated: their melody (qaːtil / qatuːl) is a fixed
# shape, so slotting the root is faithful.  A maṣdar's actual noun pattern varies
# far more than the one neutral qatāl template, so imposing it (إِرْث → warās)
# would be wrong; those keep their surface reconstruction.
_REGEN_DEVERBAL = frozenset({Layer.ACTIVE_PARTICIPLE, Layer.PASSIVE_PARTICIPLE})


def _generative_deverbal_base(
    anc_word: str, pan_stem: str, pr: ProtoRoot, deverbal: Layer,
) -> tuple[str, str]:
    """Return (ancestor, pan_stem) for a reverted PARTICIPLE whose WEAK root lost
    its glide in surface reconstruction — regenerate it root-visibly from the
    participle's melody (زَانِيَة znj + active-participle → zaːnij, so + feminine is
    zaːnija, not the glide-less zana that would double).

    Left to the surface otherwise: a maṣdar (whose noun pattern the one qatāl
    template can't stand in for), a geminate (no glide — ħabba stays ħabba), or a
    sound root whose radicals still align."""
    if (deverbal not in _REGEN_DEVERBAL
            or len(pr.pan_radicals) != 3 or len(pr.radicals) != 3
            or not (_GLIDES & set(pr.pan_radicals))
            or _align_radicals(pr.pan_radicals, pan_stem) is not None):
        return anc_word, pan_stem
    mel = PansemiticMorphology.DEVERBAL_TEMPLATES[deverbal]
    return mel.format(*pr.radicals), mel.format(*pr.pan_radicals)


def _pure_verb(phrase: AnalyzedPhrase) -> bool:
    """A verb with no nominal reading — the only POS that can't take the definite
    article (its he-/hi- is the binyan prefix, not the article)."""
    return "verb" in phrase.pos and not (phrase.pos & {"noun", "adj", "name", "num"})


def deep_parse(
    phrase: AnalyzedPhrase, base_lookup: BaseLookup,
) -> Derivation:
    """Parse a single-word phrase as deep as it can directly be: strip every
    detected layer down to the bare reconstruction base (in phoneme space),
    returning a per-side Derivation (Base carrying that stem + the ordered typed
    Transform chain for the stripped layers), so the merge only has to align the
    two chains.

    Strips, in order: the definite article (eager — script_definite is the gate,
    plus a pure-verb guard so a hifʕil he- isn't mistaken for it); the suffixal
    nominal layers (feminine, nisba, dual/plural, preferring the Wiktionary-cited
    masculine/singular base, else the IPA suffix-strip); a finite verb's derived
    stem toward its G base; and a deverbal lexeme to its verb root (its category
    carried as a re-appliable Layer)."""
    morph = MORPHOLOGY_CONFIG[phrase.lang]
    word = phrase.words[0]
    ipa = word.ipa
    layers: set[Layer] = set()

    if Layer.DEFINITE in word.layers and not _pure_verb(phrase):
        stem = morph.strip_article_ipa(ipa, word.script)
        if stem:
            ipa = stem
            layers.add(Layer.DEFINITE)

    if Layer.FEMININE in word.layers:
        stem = _reduce_to_base(
            phrase, morph, ipa, Layer.FEMININE, phrase.masculine_of, base_lookup)
        if stem:
            ipa = stem
            layers.add(Layer.FEMININE)

    if Layer.NISBA in word.layers:
        stem = morph.strip_ipa(ipa, Layer.NISBA)
        if stem:
            ipa = stem
            layers.add(Layer.NISBA)

    num = next((L for L in (Layer.DUAL, Layer.PLURAL) if L in word.layers), None)
    if num is not None:
        stem = _reduce_to_base(
            phrase, morph, ipa, num, phrase.singular_of, base_lookup)
        if stem:
            ipa = stem
            layers.add(Layer.PLURAL)  # dual collapses into the plural slot

    # Verbal morphology.  A deverbal lexeme is reduced to its verb root and its
    # category recorded as a Layer (re-applied as a pansemitic template only if
    # shared); a plain finite verb's derived stem is reduced to the G base and
    # its exponent layer(s) recorded.  ``surface`` snapshots the nominal stem
    # BEFORE the deverbal reduction, so an asymmetric (dropped) deverbal falls
    # back to its noun rather than the bare verb (see merge).
    surface = ipa
    cat = _deverbal_category(phrase)
    vs = select_verb_stem(phrase.verb_stem)
    if cat is not None:
        reduced = _reduce_verb_stem(phrase, ipa, base_lookup)
        if reduced is not None:
            ipa = reduced
            layers.add(_DERIVATION_LAYER[cat])
    elif "verb" in phrase.pos and vs is not None and vs != "G":
        stem_layers = _STEM_LAYERS.get(vs, frozenset())
        if stem_layers:
            synth = morph.synthesize_base_stem(ipa, vs)
            if synth is not None:
                ipa = synth[0]
            layers |= stem_layers

    # This side's Base carries the stripped stem; its POS is verbal iff a
    # deverbal category or a verb-stem exponent was stripped here (the shared
    # base POS is decided at reconciliation).  The Transform chain is the
    # stripped layers in canonical order.
    verbal = bool(layers & (_STEM_LAYER_SET | _DEVERBAL_LAYER_SET))
    base = (VerbalRoot(stem=ipa, radicals=(), surface=surface) if verbal
            else PrimaryNominal(pos=Pos.NOUN, stem=ipa))
    return Derivation(base, _chain_from_layers(layers))


def _component_phrase(
    lang: str, word: AnalyzedWord, meta_lookup: MetaLookup,
) -> AnalyzedPhrase:
    """Re-analyze one component word as a standalone single-word phrase, folding
    in its own lexeme-level metadata (looked up by script token).  Unknown
    components get empty metadata → concatenative-only treatment."""
    meta = meta_lookup(lang, word.script) or {}
    return analyze_phrase(lang, word.script, word.roman, word.ipa, **meta)


def _component_pairs(
    ar: AnalyzedPhrase, he: AnalyzedPhrase, meta_lookup: MetaLookup,
) -> list[tuple[AnalyzedPhrase, AnalyzedPhrase]]:
    """Aligned component phrase pairs.  A single-word phrase is its own only
    component (carrying its full metadata); a multi-word phrase is dispatched
    word-by-word, each component re-analyzed with its own looked-up metadata."""
    if len(ar.words) == 1:
        return [(ar, he)]
    return [
        (_component_phrase(ar.lang, aw, meta_lookup),
         _component_phrase(he.lang, hw, meta_lookup))
        for aw, hw in zip(ar.words, he.words)
    ]


def _layer_names(layers: set[Layer]) -> list[str]:
    """Layer set → sorted value strings, for the (deterministic) merge trace."""
    return sorted(layer.value for layer in layers)


# Shared layers that mark a VERBAL derivation (a finite stem exponent or a
# deverbal template) — their presence means the lexeme isn't a primary noun, so
# nominal root-and-pattern synthesis stays out.
_VERBAL_LAYERS = frozenset({
    Layer.GEMINATION, Layer.N_PREFIX, Layer.CAUSATIVE, Layer.T_PREFIX,
    Layer.ACTIVE_PARTICIPLE, Layer.PASSIVE_PARTICIPLE,
    Layer.VERBAL_NOUN, Layer.INSTANCE_NOUN,
})


def _nominal_pair(arc: AnalyzedPhrase, hec: AnalyzedPhrase,
                  shared: set[Layer]) -> bool:
    """Whether a pair is eligible for root-and-pattern noun synthesis.  Excludes
    verbs and any SHARED verbal exponent.  A deverbal category blocks synthesis
    only when it is SHARED (both sides the same deverbal → templated as that
    category); an ASYMMETRIC deverbal was dropped and the side reverted to its
    nominal surface, so the pair is a primary-noun pair whose catalogued pattern
    should fire (ذِكْر maṣdar + זֵכֶר segholate → qitl → zikr, not surface zikar)."""
    if "verb" in arc.pos or "verb" in hec.pos:
        return False
    if not (arc.pos & _NOMINAL_BASE_POS) or not (hec.pos & _NOMINAL_BASE_POS):
        return False
    a_cat, h_cat = _deverbal_category(arc), _deverbal_category(hec)
    if a_cat is not None and a_cat == h_cat:
        return False
    return not (shared & _VERBAL_LAYERS)


def _resolve_nominal(
    arc: AnalyzedPhrase, hec: AnalyzedPhrase, ar_stem: str,
    shared: set[Layer], proto_root: "ProtoRoot | None",
) -> NominalSynthesis | None:
    """Resolve a shared catalogued noun pattern for an eligible nominal pair, or
    None (→ keep the surface-aligned form).  Requires a triconsonantal root (3
    pansemitic radicals); tries the bilateral melody catalog first, then the
    Arabic-witnessed faʿʿāl intensive."""
    if (proto_root is None or len(proto_root.pan_radicals) != 3
            or not _nominal_pair(arc, hec, shared)):
        return None
    radicals = tuple(proto_root.pan_radicals)
    pat = match_noun_pattern(proto_root.ar_pattern, proto_root.he_pattern)
    if pat is not None:
        return NominalSynthesis(pat.psid, pat.pansemitic, radicals)
    if arabic_intensive(ar_stem):
        return NominalSynthesis("qattaːl", _QATTAL_PANSEMITIC, radicals)
    return None


def merge(
    ar: AnalyzedPhrase,
    he: AnalyzedPhrase,
    base_lookup: BaseLookup,
    meta_lookup: MetaLookup,
) -> MergeResult:
    """Align, morphology-normalize, and reconstruct a cognate pair.

    Each aligned word is parsed as deep as it can directly be (deep_parse strips
    it to a bare stem, recording its layers), then the pair is reconciled by
    re-applying only the *shared* layers — a layer on just one side was asymmetric
    morphology and is dropped.  Multi-word phrases are dispatched
    component-by-component, each carrying its own looked-up metadata, and the
    per-word ancestors joined.  Falls back to the unsplit pair when word counts
    differ.  ``trace`` records, per word, each side's stem + stripped layers, the
    merged stem + re-applied layers, and the produced word."""
    if len(ar.words) != len(he.words):
        fallback = PlannedPair(ar.ipa, he.ipa)
        ancestor, pansemitic, steps = _reconstruct_word_pairs([fallback])
        merged_stem, final = steps[0]
        trace = [MergeTrace(WordTrace(ar.ipa, []), WordTrace(he.ipa, []),
                            merged_stem, [], final)]
        # Word counts differ → a compound; the root is a single-word property.
        return MergeResult(ancestor=ancestor, pansemitic=pansemitic, trace=trace,
                           derivation_chain=derivation_label(fallback))

    pairs = _component_pairs(ar, he, meta_lookup)
    multiword = len(pairs) > 1
    word_pairs: list[PlannedPair] = []
    parsed: list[tuple[str, set[Layer], str, set[Layer]]] = []
    verb_stems: list[str | None] = []
    derivations: list[str | None] = []
    for arc, hec in pairs:
        ard = deep_parse(arc, base_lookup)
        hed = deep_parse(hec, base_lookup)
        ar_stem, he_stem = ard.base.stem, hed.base.stem
        # Reconcile the two per-side chains: keep transforms present in both, in
        # canonical order, POS-walking from the shared base — a stranded nominal
        # affix whose licensing nominalization wasn't shared drops (doubled-`aa`
        # fix).  The base is VERBAL when a deverbal rode either chain or a stem
        # exponent is shared; else a primary NOMINAL (a verb-POS homograph or a
        # one-sided root gemination does NOT make it verbal).
        base_pos, shared = _reconcile(ard.transforms, hed.transforms)
        # A side whose deverbal was asymmetric (dropped) reverts to its nominal
        # surface, so a maṣdar/participle paired with a plain noun reconstructs
        # as a noun rather than the bare verb it was de-derived to.
        ar_drop = _dropped_deverbal(ard, shared)
        he_drop = _dropped_deverbal(hed, shared)
        if ar_drop is not None:
            ar_stem = getattr(ard.base, "surface", "") or ar_stem
        if he_drop is not None:
            he_stem = getattr(hed.base, "surface", "") or he_stem
        # The feminine-plural -at marker is a property of the reconstructed form,
        # not a shared strip: a shared (surviving) plural on a feminine stem takes
        # -at.  Applied post-filter, so it only fires when the plural is licensed.
        if Layer.PLURAL in shared and (_strictly_feminine(arc) or _strictly_feminine(hec)):
            shared.add(Layer.FEMININE)
        word_pairs.append(PlannedPair(ar_stem, he_stem, layers=shared,
                                      base_pos=base_pos,
                                      reverted_deverbal=ar_drop or he_drop))
        parsed.append((ar_stem, _chain_layers(ard.transforms),
                       he_stem, _chain_layers(hed.transforms)))

        a_vs, h_vs = select_verb_stem(arc.verb_stem), select_verb_stem(hec.verb_stem)
        verb_stems.append(a_vs if (a_vs is not None and a_vs == h_vs) else None)
        a_cat, h_cat = _deverbal_category(arc), _deverbal_category(hec)
        derivations.append(a_cat.value if (a_cat is not None and a_cat == h_cat) else None)

    # The shared verb_stem / derivation / root are single-word lexeme properties.
    verb_stem = None if multiword else verb_stems[0]
    derivation = None if multiword else derivations[0]
    proto_root = None
    if not multiword:
        proto_root = reconstruct_proto_root(
            select_root_tag(ar.roots, word_pairs[0].ar_ipa, "ar"), word_pairs[0].ar_ipa,
            select_root_tag(he.roots, word_pairs[0].he_ipa, "he"), word_pairs[0].he_ipa,
        )
        # A primary noun whose two sides share a catalogued pattern is
        # re-synthesized from the root + that pattern's compromise melody (set
        # here so _reconstruct_word_pairs produces it instead of the aligned stem).
        arc0, hec0 = pairs[0]
        word_pairs[0].nominal = _resolve_nominal(
            arc0, hec0, word_pairs[0].ar_ipa, word_pairs[0].layers, proto_root)
        # A finite verb regenerates its G base from the root, so weak/geminate
        # roots reconstruct root-visibly and the ancestor converges with
        # proto_root.  Both sides must be verbs and it must not be a catalogued
        # nominal.  A deverbal category only blocks regeneration when it is
        # SHARED (a genuine participle/maṣdar lexeme, produced by its template) —
        # an ASYMMETRIC deverbal is dropped in reconciliation, leaving a plain
        # verb base that should still be root-visible (اِسْتَسْأَلَ / نִשְׁאַל[ptcp] → saʔala).
        a_cat0, h_cat0 = _deverbal_category(arc0), _deverbal_category(hec0)
        word_pairs[0].proto_root = proto_root
        word_pairs[0].verb_base = (
            "verb" in arc0.pos and "verb" in hec0.pos
            and not (a_cat0 is not None and a_cat0 == h_cat0)
            and word_pairs[0].nominal is None)

    ancestor, pansemitic, steps = _reconstruct_word_pairs(word_pairs)
    trace = [
        MergeTrace(
            ar=WordTrace(ar_stem, _layer_names(ar_layers)),
            he=WordTrace(he_stem, _layer_names(he_layers)),
            merged_stem=merged_stem,
            applied_layers=_layer_names(pp.layers),
            final=final)
        for (ar_stem, ar_layers, he_stem, he_layers), pp, (merged_stem, final)
        in zip(parsed, word_pairs, steps)
    ]
    chain = " ".join(derivation_label(pp) for pp in word_pairs)
    return MergeResult(ancestor=ancestor, pansemitic=pansemitic, trace=trace,
                       verb_stem=verb_stem, derivation=derivation,
                       proto_root=proto_root, derivation_chain=chain)
