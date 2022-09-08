import os
import pwd
import streamlit as st
import pandas as pd
import numpy as np
import transformers

print(os.getcwd())
MODEL = "/home/ubuntu/rhyme_finetuning/output/supervised_eos_10_epoch"
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL)
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
pipe = transformers.pipeline(
    model=model, task='text-generation', tokenizer=tokenizer, device=0)

st.title('GPT-2 fine-tuned for poetry')
desc = 'Please input the first line of the poem you would like to complete...'
st.write(desc)

prompt = st.text_input('First line prompt')
if prompt[-2:] != '\n':
    prompt += '\n'

if st.button('Generate Poem'):
    generated_text = pipe(prompt,
                          do_sample=True, temperature=0.7, top_p=0.9, no_repeat_ngram_size=2)[0]['generated_text']
    lines = generated_text.splitlines()[:4]
    for line in lines:
        st.write(line)
