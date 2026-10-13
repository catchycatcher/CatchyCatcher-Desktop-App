"""Phonetic helpers for rhyme matching.

Backed by ``pronouncing`` (the CMU pronunciation dictionary) when it is
installed, falling back to a spelling-only rime when it is not, so the line
grouping still runs on a bare machine. The line-break model uses
:func:`rhyme_similarity` to let a rhyme pull two nearby line endings together.
"""

from __future__ import annotations

import difflib
import re

try:  # pragma: no cover - depends on the install environment
    import pronouncing as _pronouncing
except Exception:  # noqa: BLE001 - any import problem means "no dictionary"
    _pronouncing = None

#: ARPAbet vowel phones; a word's rime begins at its last one.
_VOWELS = frozenset(
    {
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
        "IH", "IY", "OW", "OY", "UH", "UW", "AX",
    }
)

#: Two pronunciations scoring at or above this ratio are "the same sound".
SOUND_ALIKE_RATIO = 0.8
#: A rime this similar also counts as sounding alike.
SOUND_ALIKE_RHYME = 0.8

_STRESS = re.compile(r"\d")
_SPELLING_VOWELS = "aeiou"


def available() -> bool:
    """Whether the pronunciation dictionary is loaded."""
    return _pronouncing is not None


def _letters(word: object) -> str:
    return "".join(ch for ch in str(word).lower() if ch.isalpha())


def phones(word: object) -> tuple[str, ...] | None:
    """Stress-free ARPAbet phones for ``word``, or ``None`` if unknown."""
    if _pronouncing is None:
        return None
    text = _letters(word)
    if not text:
        return None
    options = _pronouncing.phones_for_word(text)
    if not options:
        return None
    return tuple(_STRESS.sub("", token) for token in options[0].split())


def _spelling_rime(word: object) -> tuple[str, tuple[str, ...]]:
    """Fallback rime from spelling when there is no dictionary entry."""
    text = _letters(word)
    if len(text) >= 3 and text.endswith("e") and text[-2] not in _SPELLING_VOWELS:
        text = text[:-1]
    vowels = [i for i, ch in enumerate(text) if ch in _SPELLING_VOWELS]
    if not vowels:
        return "", tuple(text[-2:])
    last = vowels[-1]
    start = last
    while start - 1 >= 0 and text[start - 1] in _SPELLING_VOWELS:
        start -= 1
    return text[start : last + 1], tuple(text[last + 1 :])


def rime(word: object) -> tuple[str, tuple[str, ...]]:
    """The sounded tail of a word: ``(last vowel, following consonants)``.

    ``pain`` -> ``('EY', ('N',))`` and ``insane`` -> ``('EY', ('N',))``, so the
    two are a perfect rhyme; ``masquerade`` -> ``('EY', ('D',))`` assonates.
    """
    tokens = phones(word)
    if tokens is None:
        return _spelling_rime(word)
    last = None
    for index, token in enumerate(tokens):
        if token in _VOWELS:
            last = index
    if last is None:
        return "", tuple(tokens[-2:])
    return tokens[last], tuple(tokens[last + 1 :])


def rhyme_similarity(first: object, second: object) -> float:
    """0..1: 1.0 a perfect rhyme, lower for assonance/consonance, 0.0 for none."""
    vowel_a, coda_a = rime(first)
    vowel_b, coda_b = rime(second)
    if not vowel_a and not vowel_b:
        return 0.0
    if vowel_a == vowel_b and coda_a == coda_b:
        return 1.0
    if coda_a == coda_b:
        if not coda_a:
            return 0.7
        return 0.9 if vowel_a == vowel_b else 0.6
    if vowel_a == vowel_b:
        return 0.5
    return 0.0


def signature(word: object) -> tuple[str, ...]:
    """A comparison key that maps identical-sounding words together."""
    tokens = phones(word)
    if tokens is not None:
        return tokens
    return tuple(_letters(word))


def sound_alike(first: object, second: object) -> bool:
    """Whether two words are close enough to be the same word mis-heard."""
    if _letters(first) == _letters(second):
        return True
    sig_a, sig_b = signature(first), signature(second)
    if not sig_a or not sig_b:
        return False
    if sig_a == sig_b:
        return True
    if difflib.SequenceMatcher(None, sig_a, sig_b).ratio() >= SOUND_ALIKE_RATIO:
        return True
    return rhyme_similarity(first, second) >= SOUND_ALIKE_RHYME
