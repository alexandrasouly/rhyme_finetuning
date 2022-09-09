
import streamlit as st
import transformers
from rhyme_finetuning.predict import complete_stanza
from PIL import Image
from streamlit.components import v1 as components

st.set_page_config(layout="wide")


@st.cache
def load_images():
    rej_sample_img = Image.open(
        'streamlit_demo/rejection_sampling.png')
    rl_img = Image.open('streamlit_demo/rl.png')
    return rl_img, rej_sample_img


@st.cache(allow_output_mutation=True)
def load_models():

    model_og = transformers.AutoModelForCausalLM.from_pretrained('gpt2')
    MODEL_finetuned = "/home/ubuntu/rhyme_finetuning/output/supervised_eos_10_epoch"
    model_finetuned = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_finetuned)
    MODEL_rl = "/home/ubuntu/rhyme_finetuning/output/rl/overnight/checkpoint_1500"
    model_rl = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_rl)
    return model_og, model_finetuned, model_rl


rl_img, rej_sample_img = load_images()
model_og, model_finetuned, model_rl = load_models()


tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
pipe_og = transformers.pipeline(
    model=model_og, task='text-generation', tokenizer=tokenizer, device=0)
pipe_finetuned = transformers.pipeline(
    model=model_finetuned, task='text-generation', tokenizer=tokenizer, device=0)
pipe_rl = transformers.pipeline(
    model=model_rl, task='text-generation', tokenizer=tokenizer, device=0)

# Rended starting setup
st.markdown("<h1 style='text-align: center; color: white;'>GPT-2 fine-tuned for poetry</h1>",
            unsafe_allow_html=True)

run = False  # whether we want to generate poetry

col1, col2, col3 = st.columns(3)
with col1:
    st.image(rl_img, caption="RL trying its best", width=400)
with col2:
    st.markdown(
        """
    You can generate a poem with the following models:
    - GPT-2
    - GPT-2 **finetuned on poems** and using **rejection sampling** to rhyme
    - GPT-2 **finetuned on poems** and **trained via RL** to rhyme

    _Please proceed with maximal caution when using the RL one._
    """
    )
    st.markdown(' ')
    desc = '  \n  **Please input the first line of the poem you would like to complete...**'
    st.markdown(desc)

    prompt = st.text_input('Submit First line prompt')
    if prompt[-2:] != '\n':
        prompt += '\n'
    print(prompt)
    if st.button('Generate') or prompt != '\n':
        run = True

with col3:
    st.image(rej_sample_img,
             caption='Rejection sampling producing a masterpiece', width=400)
# Generate poems
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """,
            unsafe_allow_html=True)
print(run)
if run == True:
    col4, col5, col6 = st.columns(3)
    with col4:
        st.subheader('RL mess that forgot English')
        generated_text = pipe_rl(prompt,
                                 do_sample=True, temperature=0.7, top_p=0.9, no_repeat_ngram_size=2)[0]['generated_text']
        lines = generated_text.splitlines()[:4]
        for line in lines:
            st.markdown(line)
    with col5:
        st.subheader('Basic GPT-2 (small)')
        generated_text = pipe_og(prompt,
                                 do_sample=True, temperature=0.7, top_p=0.9, no_repeat_ngram_size=2)[0]['generated_text']
        lines = generated_text.splitlines()[:4]
        for line in lines:
            st.markdown(line)
    with col6:
        st.subheader('Finetuned plus rejection sampling')
        generated_text = complete_stanza(prompt, model_finetuned, tokenizer)
        lines = generated_text.splitlines()[:4]
        for line in lines:
            st.markdown(line)
