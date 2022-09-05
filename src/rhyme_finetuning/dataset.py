#%%
from dataclasses import asdict, dataclass
import pickle
from typing import Callable, List, Optional
import torch as t
from pathlib import Path
from tqdm import tqdm
from lingua import Language, LanguageDetectorBuilder
# languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]

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
    processed_lines: List[str]

    def pretty_print(self):
        print(f'{self.poem_title} ({self.form}):')
        print(self.text)

def process_line(line: str, strict: bool = False) -> Optional[str]:
    """Remove all but allowed characters. Returns None if line is invalid."""
    # Filter out copyright statements
    if "copyright" in line.lower():
        return None
    line = line.strip()
    line = "".join([c for c in line if (c.isalnum() and c.isascii()) or c in " ,;.!?()"])
    if strict:
        line = line.rstrip(" ,.;!?()")
        if not line:
            return None
        if line[-1].isdigit():
            return None
    return line

class PoemsDataset(t.utils.data.Dataset):
    def __init__(
        self,
        # tokenizer: 
        data_folder: str = "data/poems",
        stanza_filter: Callable[[List[str]], bool] = lambda x: True,
    ):
        data_path = Path(data_folder) / "forms"
        poems = []
        for subfolder in data_path.iterdir():
            if not subfolder.is_dir():
                raise RuntimeError(f"Found unexpected file {subfolder} in data folder")
            
            form = subfolder.name
            for poem_file in subfolder.iterdir():
                if not poem_file.is_file():
                    raise RuntimeError(f"Found unexpected folder {poem_file} in data folder")
                if not poem_file.name.endswith(".txt"):
                    raise RuntimeError(f"Found unexpected file {poem_file} in data folder")
                with open(poem_file) as f:
                    text = f.read()
                title = poem_file.stem
                poems.append(Poem(title, text, form))
        
        detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
        # Concat list of all poems
        self.stanzas = []
        skipped = 0
        languages = set()
        for poem in tqdm(poems):
            if detector.detect_language_of(poem.text) != Language.ENGLISH:
                skipped += 1
                languages.add(detector.detect_language_of(poem.text))
                continue
            # Remove all but alphabet characters
            strict_lines = [process_line(line, strict=True) for line in poem.text.splitlines()]
            empty_indices = [i for i, line in enumerate(strict_lines) if not line]
            # Remove empty lines
            strict_lines = [line for line in strict_lines if line]
            # Create lines containing punctuation:
            lines = [process_line(line) for i, line in enumerate(poem.text.splitlines()) if i not in empty_indices]

            assert all(line is not None for line in strict_lines)
            assert all(line is not None for line in lines)

            # Put lines into stanzas of length 4 each:
            for i in range(0, len(lines)):
                # Skip incomplete stanzas
                if i + 4 > len(lines):
                    continue
                stanza = "\n".join(lines[i:i+4])
                processed_stanza = strict_lines[i:i+4]
                assert len(processed_stanza) == 4
                assert all(line[-1].isalpha() for line in processed_stanza), processed_stanza
                if stanza_filter(processed_stanza):
                    self.stanzas.append(Stanza(poem.title, stanza, poem.form, processed_stanza))
        
        print(f"Skipped {skipped} poems because they were not in English")
        print(f"Skipped languages: {languages}")

    def __len__(self) -> int:
        return len(self.stanzas)
    
    def __getitem__(self, idx) -> Stanza:
        return self.stanzas[idx]
    
    def save(self, path: str):
        with open("../../data/stanzas.pkl", "wb") as f:
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
