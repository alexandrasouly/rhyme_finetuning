import torch as t
import transformers
import datasets
from tokenizers.processors import TemplateProcessing
import argparse
from tqdm import tqdm

from rhyme_finetuning.rhyming import stanza_rhymes
from rhyme_finetuning.train_rl import reward, collater, get_queries

from rhyme_finetuning.predict import get_samples

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("--batch-size", type=int, default=128)
args = parser.parse_args()

gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    # "top_p": 1.0,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "no_repeat_ngram_size": 2,
    "max_new_tokens": 40,
}

device = t.device("cuda" if t.cuda.is_available() else "cpu")

model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
model.to(device)
model.eval()

################################
# Measure supervised performance
################################
dataset = datasets.load_dataset(
    "text",
    data_files={"train": "data/stanzas_train.txt", "test": "data/stanzas_test.txt"},
    sample_by="paragraph",
)
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# Add an EOS token at the end of each sample
template = TemplateProcessing(
    single=f"$A {tokenizer.eos_token}",
    special_tokens=[(f"{tokenizer.eos_token}", tokenizer.eos_token_id)],
)
tokenizer.post_processor = template
tokenizer._tokenizer.post_processor = template
tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples["text"], padding="max_length", max_length=70, truncation=True),
    batched=True,
    num_proc=4,
    remove_columns=["text"],
)

def supervised_performance(split: str):
    """Measure supervised performance on a given split ('train' or 'test')."""

    dataloader = t.utils.data.DataLoader(tokenized_dataset[split], batch_size=args.batch_size, collate_fn=collater)

    losses = []
    with t.inference_mode():
        for batch in tqdm(dataloader):
            input_ids = t.tensor(batch["input_ids"], dtype=t.long, device=device)
            attention_mask = t.tensor(batch["attention_mask"], dtype=t.long, device=device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            losses.append(outputs.loss.item())
    
    return sum(losses) / len(losses)

print("Supervised loss on train:", supervised_performance("train"))
print("Supervised loss on test:", supervised_performance("test"))

def measure_rl_performance(split: str):
    queries = get_queries(f"data/stanzas_{split}.txt")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = queries.map(
        lambda examples: {"tokens": tokenizer(examples["query"], padding="max_length", max_length=15, truncation=True)},
        num_proc=4,
    )
    dataloader = t.utils.data.DataLoader(tokenized_dataset["train"], batch_size=args.batch_size, collate_fn=collater)

    all_rewards = []
    all_num_rhymes = []
    all_line_penalties = []
    all_num_repetitions = []
    for batch in tqdm(dataloader):
        query_tensors = t.tensor([query["input_ids"] for query in batch["tokens"]]).long().to(device)

        response_tensors = model.generate(query_tensors, **gen_kwargs)
        response_tensors = response_tensors[:, query_tensors.shape[1]:]

        response_strs = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        rewards, num_rhymes, line_penalties, num_repetitions = zip(*[reward(query, response) for query, response in zip(batch["query"], response_strs)])

        all_rewards.extend(rewards)
        all_num_rhymes.extend(num_rhymes)
        all_line_penalties.extend(line_penalties)
        all_num_repetitions.extend(num_repetitions)

    return {
        "reward": sum(all_rewards) / len(all_rewards),
        "num_rhymes": sum(all_num_rhymes) / len(all_num_rhymes),
        "line_penalty": sum(all_line_penalties) / len(all_line_penalties),
        "num_repetitions": sum(all_num_repetitions) / len(all_num_repetitions),
    }

print("RL performance on train:", measure_rl_performance("train"))
print("RL performance on test:", measure_rl_performance("test"))