# %%
import transformers
import torch as t
'''This notebook investigates why we never output eos despite always training on 4 line poems that have eos at the end. This is based on training data that doesn't have newline at the end of line 4, just eos. '''

MODEL = "../output/supervised_eos_10_epoch/"
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL)
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
# %%
input_ids = t.tensor(tokenizer('She walks in beauty, like the night\n')[
                     'input_ids']).unsqueeze(dim=0)
attention_mask = t.tensor(tokenizer('She walks in beauty, like the night\n')[
                          'attention_mask']).unsqueeze(dim=0)

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
eos_outputted = (output_sequences == 50256).nonzero(as_tuple=True)
print('Generated eos indexes are (probably empty):', eos_outputted)
# sad times, no eos

# %%
# eos should be instead of 3rd new_line
last_new_line_idx = (output_sequences == 198).nonzero()[2]
print('3rd new lineoutput idx:', last_new_line_idx)
# %%

probs_of_eos_at_last_newline = probs[last_new_line_idx, 50256]
print('probs of eos where it should be:', probs_of_eos_at_last_newline)
probs_of_newline_at_last_newline = probs[last_new_line_idx,  198]
print('probs of putting new line where eos should be:',
      probs_of_newline_at_last_newline)

highest_eos_probs = t.argmax(probs[:, 50256])
print('highest eos probs are at index:', highest_eos_probs)
print('This hopefully matches the 3rd linenew output index.')

# %%
'''
Conclusions:
- observation: training data doesn’t have new lines at the end of line 4 before eos, model is getting confused whether to go with eos or newlines. It always goes with new lines, never eos. EOS have 0 prob basically everywhere, except at the end of line 4, where it is 2-5% (newline is 90%+). EOS does have the highest value at the right place though.
- Possibly putting newline before eos would help.
- Maybe model doesn’t know how to count to 4?
- Maybe more training would help.
- Maybe a larger model would be less dumb. Duh.

'''

# %%
