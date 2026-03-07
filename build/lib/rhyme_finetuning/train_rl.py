# %%
import itertools
from typing import Tuple
import os
import numpy as np
import argparse
import time
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
from rhyme_finetuning.predict import get_samples, return_four_lines


def process_line(line: str) -> str:
    """
    COPIED FROM PROCESS_DATA.PY USING STRICT VERSION 
    Remove all but allowed characters. Returns None if line is invalid.
    We strip everything except ascii alphanumeric characters. The line can't end in a number.
    Filters out copyright lines
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


def reward(query: str, response: str, c_lines: float = 1.0, c_repetitions=4.0, debug: bool = False) -> Tuple[float, int, int, int]:
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
    num_repetitions = sum(float(line1.split(
        " ")[-1] == line2.split(" ")[-1]) for line1, line2 in line_pairs)
    # line_penalty has to be squared because num_rhymes can grow quadratically with num_lines
    if debug:
        print(
            f"num_lines: {num_lines}, num_rhymes: {num_rhymes}, num_repetitions: {num_repetitions}, line_penalty: {line_penalty}")
    reward = num_rhymes - c_lines * line_penalty ** 2 - c_repetitions * num_repetitions

    return reward, num_rhymes, line_penalty, num_repetitions
# %%


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

# %%


def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def train(args):

    config = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "forward_batch_size": args.forward_batch_size,
        "ppo_epochs": args.ppo_epochs,
        # "txt_in_min_len": 2,
        # "txt_in_max_len": 8,
        # "txt_out_min_len": 4,
        # "txt_out_max_len": 16,
        "lr": args.lr,
        "init_kl_coef": 0.2,
        "target": 6,
        "horizon": 10000,
        "gamma": 1,
        "lam": 0.95,
        "cliprange": .2,
        "cliprange_value": .2,
        "vf_coef": .1,
    }

    if not args.no_track:
        wandb.init(
            entity="rhyme_finetuning",
            project="rl",
            config=config,
            name=args.name,
        )

    if args.name is not None:
        args.output_dir = os.path.join(args.output_dir, args.name)
    device = "cuda" if t.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "gpt2", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    checkpoint = args.checkpoint if args.checkpoint else args.reference_checkpoint
    model = trl.gpt2.GPT2HeadWithValueModel.from_pretrained(
        checkpoint).to(device)
    model_ref = trl.gpt2.GPT2HeadWithValueModel.from_pretrained(
        args.reference_checkpoint).to(device)
    ppo_trainer = trl.ppo.PPOTrainer(model, model_ref, tokenizer, **config)
    queries = get_queries(args.query_file)
    tokenized_dataset = queries.map(
        lambda examples: {"tokens": tokenizer(
            examples["query"], padding="max_length", max_length=15, truncation=True)},
        num_proc=4,
    )
    gen_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 40,
    }
    dataloader = t.utils.data.DataLoader(
        tokenized_dataset["train"], batch_size=args.batch_size, collate_fn=collater, shuffle=True)
    global_step = 0
    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(dataloader)):
            global_step += 1
            batch_length = len(batch["tokens"])
            if batch_length < args.batch_size:
                # Skip final batch if it's smaller than batch_size
                print(
                    f"Skipping batch {i} because length ({batch_length}) < batch size ({args.batch_size})")
                continue
            timing = {}
            t_epoch = time.time()
            query_tensors = t.tensor(
                [query["input_ids"] for query in batch["tokens"]]).long().to(device)
            t0 = time.time()
            response_tensors = model.generate(query_tensors, **gen_kwargs)
            response_tensors = response_tensors[:, query_tensors.shape[1]:]

            response_strs = [tokenizer.decode(
                r.squeeze()) for r in response_tensors]
            timing['time/get_responses'] = time.time() - t0

            t0 = time.time()
            rewards, num_rhymes, line_penalties, num_repetitions = zip(
                *[reward(query, response) for query, response in zip(batch["query"], response_strs)])
            rewards = t.tensor(rewards).to(device)
            timing['time/get_rewards'] = time.time() - t0

            # Run PPO step
            t0 = time.time()
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            timing['time/optimization'] = time.time() - t0

            timing['time/epoch'] = time.time() - t_epoch

            if not args.no_track:
                t0 = time.time()
                table = wandb.Table(columns=["Query", "Response", "Reward"])
                for row in zip(batch['query'], response_strs, rewards.cpu().tolist()):
                    table.add_data(*row)
                logs = {}
                logs["examples"] = table
                logs["epoch"] = epoch
                stats_keys = [
                    "objective/kl", "ppo/policy/entropy"
                ]
                logs.update(
                    {k: v for k, v in stats.items() if k in stats_keys})
                logs.update(timing)
                logs['env/reward_mean'] = t.mean(rewards).cpu().numpy()
                logs['env/reward_std'] = t.std(rewards).cpu().numpy()
                # logs['env/reward_dist'] = rewards.cpu().numpy()
                logs['env/num_rhymes_mean'] = np.mean(num_rhymes)
                logs['env/num_rhymes_std'] = np.std(num_rhymes)
                # logs['env/num_rhymes_dist'] = num_rhymes
                logs['env/line_penalty_mean'] = np.mean(line_penalties)
                logs['env/line_penalty_std'] = np.std(line_penalties)
                # logs['env/line_penalties_dist'] = line_penalties
                logs['env/num_repetitions_mean'] = np.mean(num_repetitions)
                logs['env/num_repetitions_std'] = np.std(num_repetitions)
                # logs['env/num_repetitions_dist'] = num_repetitions

                # Generate a few samples from the model
                texts = get_samples(model, tokenizer, gen_kwargs, device)
                texts = [text.split('\n') for text in texts]
                processed_texts = [
                    [process_line(line) for line in text] for text in texts]

                rhymes = [str(stanza_rhymes(text)) if len(
                    text) == 4 and None not in text else 'not 4' for text in processed_texts]

                for text, rhyme in zip(texts, rhymes):
                    print("\n".join(text))
                    print(rhyme)

                logs["global_step"] = global_step
                wandb.log(logs)

                print(f"Logging took {time.time() - t0} seconds")

            if global_step % args.save_every == 0:
                model.save_pretrained(os.path.join(
                    args.output_dir, f"checkpoint_{global_step}"))
                tokenizer.save_pretrained(os.path.join(
                    args.output_dir, f"checkpoint_{global_step}"))

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
# %%


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-track", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--reference-checkpoint", type=str,
                        default="./output/supervised_eos_10_epoch")
    parser.add_argument("--query-file", type=str, default="./data/stanzas.txt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--forward-batch-size", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./output/rl")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.41e-5)
    args = parser.parse_args()

    if args.debug:
        args.batch_size = 2
        args.forward_batch_size = 2
        args.steps = 10
        args.ppo_epochs = 1
    train(args)
