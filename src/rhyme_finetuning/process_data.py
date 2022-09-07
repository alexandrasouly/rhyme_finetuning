# %%
import copy
from dataclasses import asdict, dataclass
import math
import pickle
from typing import List, Optional
import torch as t
from pathlib import Path
from tqdm import tqdm
from lingua import Language, LanguageDetectorBuilder
from rhyme_finetuning.rhyming import stanza_rhymes
from torch.utils.data import random_split


@dataclass
class Poem:
    title: str
    text: str
    form: str


@dataclass
class Stanza:
    poem_title: str
    text: str
    form: str
    tokens: Optional[List[int]] = None
    attention_mask: Optional[List[bool]] = None

    def pretty_print(self):
        print(f'{self.poem_title} ({self.form}):')
        print(self.text)


def process_line(line: str, strict: bool = False) -> Optional[str]:
    """
    Remove all but allowed characters. Returns None if line is invalid.
    If strict, we strip everything from the end except ascii alphanumeric characters. The line can't end in a number.
    If not strict, we allow common punctuation as well. The line can end in a number.
    Both filters out copyright lines
    """
    # Filter out copyright statements
    if "copyright" in line.lower():
        return None
    line = line.strip()
    line = "".join([c for c in line if (
        c.isalnum() and c.isascii()) or c in " ,;.!?()"])
    if strict:
        line = line.rstrip(" ,.;!?()")
        if not line:
            return None
        if line[-1].isdigit():
            return None
    return line


def stanzas_from_poem(poem: Poem) -> List[Stanza]:
    '''
    Make processed stanzas from a poem. Only keep the ones that rhyme.

    If a line would be None when strictly filtered, we just get rid of it.
    '''
    processed_stanzas = []
    punctuated_lines = [process_line(line) for line in poem.text.splitlines(
    ) if process_line(line, strict=True) is not None]
    strict_lines = [process_line(line, strict=True) for line in poem.text.splitlines(
    ) if process_line(line, strict=True) is not None]
    assert all(line is not None for line in punctuated_lines)
    assert all(line is not None for line in strict_lines)

    # Put processed lines into stanzas of length 4 each:
    for i in range(0, len(punctuated_lines)):
        # Skip incomplete stanzas
        if i + 4 > len(punctuated_lines):
            continue
        if stanza_rhymes(strict_lines[i:i+4]):
            text = "\n".join(punctuated_lines[i:i+4])
            processed_stanzas.append(
                Stanza(poem_title=poem.title, text=text, form=poem.form))
    return processed_stanzas


def load_poems_from_file(data_folder) -> List[Poem]:
    '''Loads poems from file without any preprocessing'''
    data_path = Path(data_folder) / "forms"
    poems = []
    for subfolder in data_path.iterdir():
        if not subfolder.is_dir():
            raise RuntimeError(
                f"Found unexpected file {subfolder} in data folder")

        form = subfolder.name
        for poem_file in subfolder.iterdir():
            if not poem_file.is_file():
                raise RuntimeError(
                    f"Found unexpected folder {poem_file} in data folder")
            if not poem_file.name.endswith(".txt"):
                raise RuntimeError(
                    f"Found unexpected file {poem_file} in data folder")
            with open(poem_file) as f:
                text = f.read()
            title = poem_file.stem
            poems.append(Poem(title, text, form))
    return poems


def process_all_poems(poems: List[Poem]) -> List[Stanza]:
    detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
    processed_stanzas = []
    for poem in tqdm(poems):
        # if poem is not English, we skip
        if detector.detect_language_of(poem.text) != Language.ENGLISH:
            continue
        stanzas = stanzas_from_poem(poem)
        if stanzas:
            processed_stanzas.extend(stanzas)
    return processed_stanzas


class PoemsDataset(t.utils.data.Dataset):
    def __init__(
        self,
        data_folder: str = "data/poems",
        stanzas=None
    ):
        if stanzas is None:
            self.stanzas = []
        else:
            self.stanzas = copy.deepcopy(stanzas)

        poems = load_poems_from_file(data_folder)
        stanzas = process_all_poems(poems)
        self.stanzas = stanzas

    def __len__(self) -> int:
        return len(self.stanzas)

    def __getitem__(self, idx) -> Stanza:
        return self.stanzas[idx]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump([asdict(stanza) for stanza in self.stanzas], f)

    def save_plaintext(self, path: str):
        with open(path, "w") as f:
            for stanza in self.stanzas:
                f.write(stanza.text)
                f.write("\n\n")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            stanzas = pickle.load(f)
        dataset = cls.__new__(cls)
        dataset.stanzas = [Stanza(**stanza) for stanza in stanzas]
        return dataset


# %%
if __name__ == '__main__':
    raise ValueError(
        'Comment in this warning to run the script to re-process data. WARNING: it will overwrite your existing data and takes ~10 mins.')

    dataset = PoemsDataset()
    dataset.save('data/stanzas.pkl')
    dataset.save_plaintext('data/stanzas.txt')
    train_set, test_set = random_split(dataset, [math.floor(
        0.8*len(dataset)), math.floor(0.2*len(dataset))])
    with open('data/stanzas_test.txt', "w") as f:
        for stanza in test_set:
            f.write(stanza.text)
            f.write("\n\n")
    with open('data/stanzas_train.txt', "w") as f:
        for stanza in train_set:
            f.write(stanza.text)
            f.write("\n\n")
