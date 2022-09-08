# %%
from unittest import result
from einops import repeat
import torch as t
import transformers

from rhyme_finetuning.rhyming import stanza_rhymes
from rhyme_finetuning.process_data import process_line_return_empty

# %%


def return_four_lines(model, tokenizer):
    '''Samples a stanza for three example prompts'''
    prompt1 = 'Three rings for the elven kings high under the sky\n'
    prompt2 = 'To be or not to be: that is the question\n'
    prompt3 = 'His palms are sweaty, knees weak, arms heavy\n'
    results = []
    eos_probs = []
    for prompt in [prompt1, prompt2, prompt3]:
        input_ids = t.tensor(
            tokenizer(prompt)['input_ids'], device=model.device).unsqueeze(dim=0)
        attention_mask = t.tensor(
            tokenizer(prompt)['attention_mask'], device=model.device).unsqueeze(dim=0)
        model_output = model.generate(input_ids, attention_mask=attention_mask, output_scores=True, return_dict_in_generate=True,
                                      do_early_stopping=True, max_new_tokens=50, no_repeat_ngram_size=2)
        sequences = model_output.sequences.squeeze(dim=0).to(model.device)
        gen_text = tokenizer.decode(sequences)
        output = gen_text.split('\n')[:4]
        output = '\n'.join(output)
        print('\n'+output)
        results.append(output)
        probs = t.softmax(t.stack(model_output.scores, dim=0), dim=-1)
        if len((sequences == 198).nonzero()) >= 4:
            last_new_line_idx = (sequences == 198).nonzero()[3]
            probs_of_eos_at_last_newline = probs[last_new_line_idx, :, 50256]
        else:
            probs_of_eos_at_last_newline = 0
        eos_probs.append(probs_of_eos_at_last_newline)

    return results, eos_probs


def get_samples(model, tokenizer, gen_kwargs, device):
    queries = [
        'Three rings for the elven kings high under the sky\n',
        'To be or not to be: that is the question\n',
        'His palms are sweaty, knees weak, arms heavy\n'
    ]
    query_tensors = t.tensor(tokenizer(
        queries, padding="max_length", max_length=15, truncation=True)["input_ids"]).to(device)
    response_tensors = model.generate(query_tensors, **gen_kwargs)
    response_tensors = response_tensors[:, query_tensors.shape[1]:]

    response_strs = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    response_strs = [[q.rstrip()] + r.split('\n')[:3]
                     for r, q in zip(response_strs, queries)]
    response_strs = ['\n'.join(r) for r in response_strs]
    return response_strs

#%%
def sample(prompt, model, tokenizer, gen_kwargs, device, rejection=True, allow_repeats=False, rhyme_with=-1, batch_size=128, max_steps=750):
    """Generate a response using rejection sampling."""
    # While the response doesn't rhyme, sample a new batch:
    if not rejection:
        batch_size = 1
    
    n_prompt_lines = len(prompt.split('\n'))
    for _ in range(max_steps // batch_size):
        if not prompt.endswith("\n"):
            prompt = prompt + "\n"
        query_tensor = t.tensor(tokenizer(prompt)["input_ids"], dtype=t.long, device=device)
        query_tensor = repeat(query_tensor, "s -> b s", b=batch_size)
        response_tensors = model.generate(query_tensor, **gen_kwargs)
        response_strs = [tokenizer.decode(r) for r in response_tensors]

        for response in response_strs:
            lines = response.split("\n")
            # Line the new one should rhyme with:
            first_line = lines[n_prompt_lines + rhyme_with]
            # New line:
            second_line = lines[n_prompt_lines]
            processed_first_line, processed_second_line = process_line_return_empty(first_line, strict=True), process_line_return_empty(second_line, strict=True)
            if not allow_repeats:
                first_line_end = processed_first_line.split(" ")[-1]
                second_line_end = processed_second_line.split(" ")[-1]
                if first_line_end.lower() == second_line_end.lower():
                    continue
            if not rejection:
                return second_line
            if stanza_rhymes([processed_first_line, processed_second_line]):
                return second_line
    return second_line
            
GEN_KWARGS = {
    "min_length":-1,
    "top_k": 0.0,
    # "top_p": 1.0,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "no_repeat_ngram_size": 2,
    "max_new_tokens": 40,
}

def complete_stanza(prompt, model, tokenizer=None, gen_kwargs=GEN_KWARGS, device=None, schema="aabb"):
    prompt = prompt.rstrip("\n")
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    if device is None:
        device = next(model.parameters()).device
    stanza = [prompt]
    if schema == "aabb":
        stanza.append(sample(prompt, model, tokenizer, gen_kwargs, device))
        stanza.append(sample("\n".join(stanza), model, tokenizer, gen_kwargs, device, rejection=False))
        stanza.append(sample("\n".join(stanza), model, tokenizer, gen_kwargs, device))
    elif schema == "abab":
        stanza.append(sample(prompt, model, tokenizer, gen_kwargs, device, rejection=False))
        stanza.append(sample("\n".join(stanza), model, tokenizer, gen_kwargs, device, rhyme_with=-2))
        stanza.append(sample("\n".join(stanza), model, tokenizer, gen_kwargs, device, rhyme_with=-2))
    return "\n".join(stanza)

#%%
if __name__ == "__main__":
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = transformers.AutoModelForCausalLM.from_pretrained("../../output/supervised_eos_10_epoch").to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
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

# %%
if __name__ == '__main__':
    # uncomment next line to set which model to load
    # MODEL = '../../output/supervised_eos/checkpoint-209'

    pipe = transformers.pipeline(
        model=MODEL, task='text-generation')
    output = return_four_lines(pipe)
