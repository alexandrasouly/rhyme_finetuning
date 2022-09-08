#%%
import torch as t
import transformers

#%%
device = "cuda" if t.cuda.is_available() else "cpu"
ref_model = transformers.AutoModelForCausalLM.from_pretrained("../../output/supervised_eos_10_epoch").to(device)
# %%
model = transformers.AutoModelForCausalLM.from_pretrained("../../output/rl/overnight/checkpoint_1350").to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
#%%
def kl_div(prompt):
    # input_ids, attention_mask = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=15, truncation=True)
    # input_ids, attention_mask = t.tensor(input_ids, device=device).unsqueeze(dim=0), t.tensor(attention_mask, device=device).unsqueeze(dim=0)
    # response = model.generate(attention_mask=attention_mask, input_ids=input_ids, max_new_tokens=50, output_attentions=True, do_sample=True, return_dict_in_generate=True)
    # tokens = response["sequences"]
    # attention_mask = response["attentions"]
    # output = model(tokens, attention_mask=attention_mask)
    # ref_output = ref_model(tokens, attention_mask=attention_mask)

    # logprobs = output.logits.log_softmax(dim=-1)[tokens]
    # ref_logprobs = ref_output.logits.log_softmax(dim=-1)[tokens]

    # return (logprobs - ref_logprobs)
    with t.no_grad():
        logits, _, v = model(input_ids)
        ref_logits, _, _ = ref_model(input_ids)
    logprobs = logprobs_from_logits(logits[:,:-1,:], input_ids[:,1:])
    ref_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], input_ids[:,1:])





# %%
pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
ref_pipe = transformers.pipeline("text-generation", model=ref_model, tokenizer=tokenizer, device=0)
# %%

input_ids = t.tensor(tokenizer('She walks in beauty, like the night\n')[
                     'input_ids'], device=device).unsqueeze(dim=0)
attention_mask = t.tensor(tokenizer('She walks in beauty, like the night\n')[
                          'attention_mask'], device=device).unsqueeze(dim=0)

result = model.generate(attention_mask=attention_mask, input_ids=input_ids,
                        output_scores=True, return_dict_in_generate=True, do_early_stopping=True, max_new_tokens=50, no_repeat_ngram_size=2)
# %%
sequences = result.sequences.squeeze(dim=0)
scores = result.scores
print(tokenizer.decode(sequences))
probs = t.softmax(t.stack(scores, dim=0).squeeze(dim=1), dim=-1)
# the output sequence without the prompt
output_sequences = sequences.squeeze(dim=0)[len(input_ids[0]):]
assert len(probs) == len(output_sequences)
# %%
scores = t.stack(scores, dim=0).squeeze(dim=1)
# %%
