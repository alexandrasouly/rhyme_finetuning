from typing import Any, Callable, List
import nltk
from itertools import dropwhile
'''Credit to https://github.com/diego-vicente/dandelion for the logic'''

nltk.download('cmudict')
pronounciation = dict(nltk.corpus.cmudict.entries())


def vowel_rhyme(word_a: str, word_b: str) -> bool:
    """Return whether two words form a rhyme on vowels.
    :param word_a: first word.
    :param word_b: second word.
    :returns: True if they rhyme on vowels.
    :rtype: bool
    """
    phonemes_a = get_phonemes(word_a, lambda p: any(ch.isdigit() for ch in p))
    phonemes_b = get_phonemes(word_b, lambda p: any(ch.isdigit() for ch in p))
    return phonemes_a == phonemes_b


def perfect_rhyme(word_a: str, word_b: str) -> bool:
    """Return whether two words form a perfect rhyme.
    :param word_a: first word.
    :param word_b: second word.
    :returns: True if they rhyme perfectly.
    :rtype: bool
    """
    phonemes_a = get_phonemes(word_a, lambda p: True)
    phonemes_b = get_phonemes(word_b, lambda p: True)
    return phonemes_a == phonemes_b


def get_phonemes(word: str, selection_criteria: Callable) -> List[Any]:
    """Get the phonetic representation of the syllables after the stress.
    :param word: String containing the word.
    :param selection_criteria: Function to filter the selected phonemes.
    :returns: Syllables corresponding to the word.
    :rtype: list
    """
    try:
        key = word.strip().lower()
        ending = dropwhile(lambda x: '1' not in x, pronounciation[key])
        return [p for p in ending if selection_criteria(p)]
    except KeyError:
        return []
