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


def stanza_rhymes(lines: List[str]) -> bool:
    """
    Check if given lines rhyme.
    Expects preprocessed lines with no punctuation at the end of the lines.
    Two lines: checks if AA
    Four lines: checks if ABBA, ABAB, AABB or AAAA.
    """
    if not ((len(lines) == 4) or (len(lines) == 2)):
        raise RuntimeError(
            f'number of lines is not 2 or 4a')
    last_words = [line.split(' ')[-1] for line in lines]
    assert '' not in last_words and None not in last_words
    if len(last_words) == 2:
        return vowel_rhyme(last_words[0], last_words[1])
    elif len(last_words) == 4:
        abba = (vowel_rhyme(last_words[0], last_words[3]) and vowel_rhyme(
            last_words[1], last_words[2]))
        abab = (vowel_rhyme(last_words[0], last_words[2]) and vowel_rhyme(
            last_words[1], last_words[3]))
        aabb = (vowel_rhyme(last_words[0], last_words[1]) and vowel_rhyme(
            last_words[2], last_words[3]))
        return ((abab or abba) or aabb)

    else:
        raise RuntimeError(
            f'something went wrong during preprocessing, rhyme checker got weird lines:\n{lines}')
