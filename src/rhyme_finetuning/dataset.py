#%%
from dataclasses import dataclass
import torch as t
from pathlib import Path

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

class PoemsDataset(t.utils.data.Dataset):
    def __init__(self, data_folder: str = "data/poems"):
        self.data_folder = Path(data_folder) / "forms"
        self.poems = []
        for subfolder in self.data_folder.iterdir():
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
                self.poems.append(Poem(title, text, form))
        
        # Concat list of all poems
        self.stanzas = []
        for poem in self.poems:
            # Remove empty lines
            lines = [line for line in poem.splitlines() if line.strip()]
            # Put lines into stanzas of length 4 each:
            stanzas = ["\n".join(lines[i:i+4]) for i in range(0, len(lines), 4)]
            for stanza in stanzas:
                self.stanzas.append(Stanza(poem.title, stanza, poem.form))

    def __len__(self) -> int:
        return len(self.stanzas)
    
    def __getitem__(self, idx) -> Stanza:
        return self.stanzas[idx]
# %%
data = PoemsDataset("../../data/poems")
# %%
