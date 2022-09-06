# %%
import argparse
import datasets
import transformers


# %%
def train(args):
    ''' Full training loop using Huggingface's Trainer'''
    dataset = datasets.load_dataset(
        "text",
        data_files={"train": args.train_file, "test": args.test_file},
        sample_by="paragraph",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False)
    trainer_args = transformers.TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        logging_steps=10,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        learning_rate=args.lr,
        save_steps=5_000,
        push_to_hub=False,
        report_to="wandb" if not args.no_track else "none",
    )
    # %%

    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

    # %%
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    trainer.train()


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str,
                        default="data/stanzas_train.txt")
    parser.add_argument("--test-file", type=str,
                        default="data/stanzas_test.txt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--no-track", action="store_true")
    args = parser.parse_args()

    train(args)
