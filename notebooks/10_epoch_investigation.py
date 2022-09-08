# %%
import transformers
MODEL = "../output/supervised_eos_10_epoch/"
model = transformers.AutoModelForCausalLM.from_pretrained(
    'MODEL')
# %%
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
pipe = transformers.pipeline(
    model=model, task='text-generation', tokenizer=tokenizer, device=0)

output = pipe('Poetry is like the dawn\n',
              do_sample=True, temperature=0.7, top_p=0.9, no_repeat_ngram_size=2)[0]['generated_text']
print(output)


# %%
