# Checking if pronounce library is faster than my hacked rhyming library
# Spoilers: my version is almost two order of magnitues faster
# %%
from rhyme_finetuning.rhyming import vowel_rhyme, perfect_rhyme
import pronouncing

import time

with open('benchmark.txt') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

    test_words = list(zip(lines, reversed(lines)))

# %%
# Benchmarking rhyme
start = time.perf_counter()
rhyme_results = []
for a, b in test_words:
    rhyme_results.append(rhyme(a, b))
end = time.perf_counter()
total = end-start


print('My hacked stuff:', total)
print('Rhymes:', sum(rhyme_results))

# Benchmarking pronounce
start = time.perf_counter()
pronounce_results = []
for a, b in test_words:
    pronounce_results.append(a in pronouncing.rhymes(b))
end = time.perf_counter()
total = end-start

print('Pronouncing lib:', total)
print('Rhymes:', sum(pronounce_results))
# %%

# Rhymes predicted by hacked one
vowel_rhyme_results = []
perfect_rhyme_results = []
for a, b in test_words:
    vowel_rhyme_results.append(vowel_rhyme(a, b))
    perfect_rhyme_results.append(perfect_rhyme(a, b))

results = zip(test_words, vowel_rhyme_results)
print('These words rhyme on vowels:')
for words, rhymes in results:
    if rhymes is True:
        print(words)

results = zip(test_words, perfect_rhyme_results)
print('These words rhyme perfectly:')
for words, rhymes in results:
    if rhymes is True:
        print(words)

# Conclusion: prefect rhymes are way to restrictive
# %%
