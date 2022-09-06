from typing import List, Tuple
from einops import rearrange
import torch as t
import transformers
import argparse
from tqdm import tqdm

from rhyme_finetuning import PoemsDataset
from rhyme_finetuning.dataset_old import Stanza

device = "cuda" if t.cuda.is_available() else "cpu"

def poem_collate_fn(batch: List[Stanza]) -> Tuple[t.Tensor, t.Tensor]:
    # Extract tokens and convert list to tensor:
    tokens = t.tensor([stanza.tokens for stanza in batch], dtype=t.long)
    attention_masks = t.tensor([stanza.attention_mask for stanza in batch], dtype=t.float32)
    assert len(tokens) == len(batch) == len(attention_masks)
    return tokens, attention_masks

def train(args):
    # Load the dataset
    dataset = PoemsDataset.load(args.data_file)

    # Create the model
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

    # Create the dataloader
    dataloader = t.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=poem_collate_fn)

    # Create the optimizer
    optimizer = t.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    model.to(device)
    # Train the model
    for epoch in range(args.epochs):
        for input_ids, attention_masks in tqdm(dataloader):
            optimizer.zero_grad()
            assert input_ids.ndim == 2
            assert input_ids.shape == attention_masks.shape
            B, S = input_ids.shape
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            outputs = model(input_ids, attention_mask=attention_masks, labels=input_ids)
            logits = outputs.logits
            assert logits.ndim == 3
            assert logits.shape[:2] == (B, S)
            logits = logits[attention_masks == 1, :]
            input_ids = input_ids[attention_masks == 1]
            # logits = rearrange(logits, "b s v -> (b s) v")
            # input_ids = rearrange(input_ids, "b s -> (b s)")
            # TODO: We need to shift the labels!
            loss = t.nn.functional.cross_entropy(logits, input_ids)
            assert abs(loss.item() - outputs.loss.item()) < 1e-6, f"{loss.item()} != {outputs.loss.item()}"
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")

    # Save the model
    # model.save_pretrained("trained_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="data/stanzas_70.pkl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    train(args)