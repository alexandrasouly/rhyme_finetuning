from rhyme_finetuning.rhyming import vowel_rhyme, perfect_rhyme, stanza_rhymes
import pytest


def test_vowel_rhyms():
    assert vowel_rhyme('race', 'tail')
    assert not vowel_rhyme('car', 'house')


def test_perfect_rhyms():
    assert perfect_rhyme('mellow', 'yellow')
    assert not perfect_rhyme('race', 'tail')


def test_stanza_rhymes():
    # aabb
    assert stanza_rhymes(
        ['the car', 'went far', 'across the sky', 'up up high'])
    # abab
    assert stanza_rhymes(
        ['the car', 'across the sky', 'went far',  'up up high'])
    # abba
    assert stanza_rhymes(
        ['the car', 'across the sky',  'up up high', 'went far'])
    assert not stanza_rhymes(
        ['hi here', 'not a rhyme', 'i love unicorns', 'good bye'])
    # aa
    assert stanza_rhymes(['the car', 'went far'])

    with pytest.raises(RuntimeError):
        stanza_rhymes(['the car', 'went far', 'hello 3 lines erro'])
