# %%
import argparse
import datasets
from rhyme_finetuning.predict import return_four_lines
from rhyme_finetuning.process_data import process_line
from rhyme_finetuning.rhyming import stanza_rhymes
import transformers
from tokenizers.processors import TemplateProcessing

# %%


def train(args):
    ''' Full training loop using Huggingface's Trainer'''
    dataset = datasets.load_dataset(
        "text",
        data_files={"train": args.train_file, "test": args.test_file},
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
    # Dataset is made up of Stanzas, which store poems in text
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )

    # quick check first and last example have eos at the end and attention mask len is fine
    assert tokenized_dataset['train'][-1]['input_ids'][-1] == tokenizer.eos_token_id
    assert tokenized_dataset['train'][0]['input_ids'][-1] == tokenizer.eos_token_id
    assert len(tokenized_dataset['train'][0]['input_ids']) == len(
        tokenized_dataset['train'][0]['attention_mask'])
    assert tokenized_dataset['train'][0]['attention_mask'][-1] == 1
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False)
    trainer_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        learning_rate=args.lr,
        save_strategy='epoch',
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
    our_callback = CustomWandbCallback()
    trainer.remove_callback(transformers.integrations.WandbCallback)
    trainer.add_callback(our_callback)
    trainer_output = trainer.train()
    trainer.save_model(args.output_dir)
    our_callback._wandb.log(
        {'generations': our_callback.table})
    print(trainer_output)


class CustomWandbCallback(transformers.integrations.WandbCallback):
    def __init__(self) -> None:
        super().__init__()
        # no eos postprocessing here
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        self.table = self._wandb.Table(columns=['text', 'rhymes', 'eos_prob'])

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        '''We sample some 4 lines based on hardcoded prompts and check if they rhyme, and how likely eos probs was'''
        super().on_evaluate(args, state, control, model=model, **kwargs)
        assert model is not None

        texts, eos_probs = return_four_lines(model, self.tokenizer)
        texts = [text.split('\n') for text in texts]
        processed_texts = [[process_line(line, strict=True)
                            for line in text] for text in texts]

        rhymes = [str(stanza_rhymes(text)) if len(
            text) == 4 and None not in text else 'less than 4' for text in processed_texts]

        for log_text, log_rhyme, eos_prob in zip(texts, rhymes, eos_probs):
            self.table.add_data(log_text, log_rhyme, eos_prob)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str,
                        default="data/stanzas_train.txt")
    parser.add_argument("--test-file", type=str,
                        default="data/stanzas_test.txt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--no-track", action="store_true")
    parser.add_argument("--output-dir", type=str,
                        default="output/supervised_eos_10_epoch")
    args = parser.parse_args()

    train(args)

# %%
