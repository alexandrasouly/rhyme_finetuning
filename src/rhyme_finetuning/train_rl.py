#%%
import itertools
import argparse
import transformers
import trl
import datasets
import torch as t
import trl
import wandb
from tqdm import tqdm
import trl.gpt2
import trl.ppo

from rhyme_finetuning.rhyming import stanza_rhymes

def process_line(line: str) -> str:
    """
    Remove all but allowed characters. Returns None if line is invalid.
    If strict, we strip everything except ascii alphanumeric characters. The line can't end in a number.
    If not strict, we allow common punctuation as well. The line can end in a number.
    Both filters out copyright lines
    """
    allowed_chars = " ,;.!?()-_:"
    line = line.strip()
    line = "".join([c for c in line if (
        c.isalnum() and c.isascii()) or c in allowed_chars])
    line = line.rstrip(allowed_chars)
    if len(line) == 0:
        return ""
    if line[-1].isdigit():
        return ""
    return line

def reward(query: str, response: str, c_lines: float = 1.0, c_repetitions = 4.0, debug: bool = False) -> float:
    """Reward function for the RL agent."""
    stanza = query + response
    lines = stanza.splitlines()
    # Truncating before computing num_lines, so no penalty for more than 4 lines
    lines = lines[:4]
    num_lines = len(lines)
    lines = [process_line(line) for line in lines]
    line_penalty = abs(num_lines - 4)
    line_pairs = list(itertools.combinations(lines, 2))
    num_rhymes = sum(float(stanza_rhymes(pair)) for pair in line_pairs)
    num_repetitions = sum(float(line1.split(" ")[-1] == line2.split(" ")[-1]) for line1, line2 in line_pairs)
    # line_penalty has to be squared because num_rhymes can grow quadratically with num_lines
    if debug:
        print(f"num_lines: {num_lines}, num_rhymes: {num_rhymes}, num_repetitions: {num_repetitions}, line_penalty: {line_penalty}")
    return num_rhymes - c_lines * line_penalty ** 2 - c_repetitions * num_repetitions
#%%
def get_queries(data_file: str):
    """Get queries for the RL agent."""
    dataset = datasets.load_dataset(
        "text",
        data_files={"train": data_file},
        sample_by="paragraph",
    )
    # Take only the first line of each stanza as query:
    dataset = dataset.map(
        lambda examples: {"query": examples["text"].splitlines()[0] + "\n"},
        remove_columns=["text"],
    )
    return dataset

#%%


def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def train(args):

    config = {
        "steps": 20000,
        "batch_size": args.batch_size,
        "forward_batch_size": 16,
        "ppo_epochs": 4,   
        # "txt_in_min_len": 2,
        # "txt_in_max_len": 8,
        # "txt_out_min_len": 4,
        # "txt_out_max_len": 16,
        "lr": 1.41e-5,
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1, 
    }
    device = "cuda" if t.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = trl.gpt2.GPT2HeadWithValueModel.from_pretrained(args.checkpoint).to(device)
    model_ref = trl.gpt2.GPT2HeadWithValueModel.from_pretrained(args.checkpoint).to(device)
    ppo_trainer = trl.ppo.PPOTrainer(model, model_ref, tokenizer, **config)
    queries = get_queries(args.query_file)
    tokenized_dataset = queries.map(
        lambda examples: {"tokens": tokenizer(examples["query"], padding="max_length", max_length=15, truncation=True)},
        num_proc=4,
    )
    gen_kwargs = {
        "min_length":-1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 40,
    }
    dataloader = t.utils.data.DataLoader(tokenized_dataset["train"], batch_size=args.batch_size, collate_fn=collater)
    for i, batch in enumerate(tqdm(dataloader)):
        response_tensors = []
        query_tensors = t.tensor([query["input_ids"] for query in batch["tokens"]]).long().to(device)
        response_tensors = model.generate(query_tensors, **gen_kwargs)
        response_tensors = response_tensors[:, query_tensors.shape[1]:]
        # response = t.cat((response, t.full((gen_kwargs["max_new_tokens"] - len(response), ), tokenizer.eos_token_id, dtype=t.long, device=device)))

        response_strs = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        rewards = t.tensor([reward(query, response) for query, response in zip(batch["query"], response_strs)]).to(device)
        
        #### Run PPO step 
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        print(stats)
#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-track", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="./output/supervised_eos/checkpoint-209")
    parser.add_argument("--query-file", type=str, default="./data/stanzas.txt")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train(args)

    
# %%
reward("Three rings for the elven kings high under the sky\n", "Nine for mortal men doomed to die\n die \n die\n die", debug=True)