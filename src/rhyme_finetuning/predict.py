# %%
import transformers

# %%


def return_four_lines(pipe: transformers.pipeline):
    '''Samples a stanza for our prompts from a text-gen pipeline'''
    output = pipe('Three rings for the elven kings high under the sky\n',
                  do_sample=True, temperature=0.7, top_p=0.9, no_repeat_ngram_size=2, min_lenght=40)[0]['generated_text']
    output = output.split('\n')[:4]
    output1 = '\n'.join(output)
    print('\n'+output1)

    output = pipe('To be or not to be: that is the question\n',
                  do_sample=True, temperature=0.7, top_p=0.9, no_repeat_ngram_size=2, min_lenght=40)[0]['generated_text']
    output = output.split('\n')[:4]
    output2 = '\n'.join(output)
    print('\n'+output2)

    output = pipe('His palms are sweaty, knees weak, arms heavy\n',
                  do_sample=True, temperature=0.7, top_p=0.9, no_repeat_ngram_size=2, min_lenght=40)[0]['generated_text']
    output = output.split('\n')[:4]
    output3 = '\n'.join(output)
    print('\n'+output3)
    return output1, output2, output3


# %%
if __name__ == '__main__':
    # uncomment next line to set which model to load
    # MODEL = '../../output/supervised_eos/checkpoint-209'

    pipe = transformers.pipeline(
        model=MODEL, task='text-generation')
    output = return_four_lines(pipe)
