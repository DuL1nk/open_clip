import pdb
import random

import torch
from transformers import ElectraTokenizerFast, ElectraForMaskedLM
from open_clip import MLMLoss
from open_clip.mask_tokens import MaskTokens
from open_clip.mask_tokens import SelectMaskTokensFromText
import time
import numpy as np


PRETRAINED_ELECTRA_GENERATORS = {
    'small': "../electra_configs/electra-small-generator/",
    'base': "../electra_configs/electra-base-generator/",
    'large': "../electra_configs/electra-large-generator/"
}

ELECTRA_TOKENIZER_VOCAB = "../electra_configs/vocab/"

tokenizer = ElectraTokenizerFast.from_pretrained(ELECTRA_TOKENIZER_VOCAB)
# generator = ElectraForMaskedLM.from_pretrained(PRETRAINED_ELECTRA_GENERATORS['small'])



def parse_text_and_mask(text, mask_prob=0.3, mask='[MASK]'):

    mask_pos = ['NN', 'NNS', 'NNP', 'NNPS',
                'JJ', 'JJR', 'JJS']

    import nltk
    text = nltk.wordpunct_tokenize(text)
    pos = nltk.pos_tag(text)
    words = []
    # t1 = time.time()
    for pair in pos:
        if pair[1] in mask_pos and np.random.random() < mask_prob:
            words.append(mask)
        else:
            words.append(pair[0])
    # t2 = time.time()
    if mask not in words:
        num_indices = random.randint(1, len(words))
        indices = random.sample(range(0, len(words)), num_indices)
        for index in indices:
            words[index] = mask
    # t3 = time.time()
    # print(f'mask costs{t2-t1}')
    # print(f'random costs{t3-t2}')
    return ' '.join(words)



def truncate_tokens(all_tokens, truncate_length, eot_token):
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > truncate_length:
            tokens = tokens[:truncate_length]  # Truncate
            tokens[-1] = eot_token
        all_tokens[i] = torch.tensor(tokens)
    return all_tokens





def tokenize(texts, context_length=77, mask_prob=0, word_parsing_mask=False, generator=None, gumbel_t=1., device='cpu', return_generation=False):

    # Receive a list of sentences

    assert mask_prob or not generator, 'mask is required for enabling generation!'
    assert mask_prob or not word_parsing_mask, 'mask is required for enabling word parsing mask!'
    assert generator or not return_generation, 'generator is required for returning generation results!'

    unmask_flag = -1
    sot_token = tokenizer.encode('[CLS]')[1]
    eot_token = tokenizer.encode('[SEP]')[1]
    pad_token = tokenizer.encode('[PAD]')[1]
    mask_token = tokenizer.encode('[MASK]')[1]

    if word_parsing_mask:

        all_tokens = []
        all_labels = []

        for text in texts:
            tokens, labels = SelectMaskTokensFromText(text, tokenizer, unmask_flag, mask_prob=mask_prob, sot_token=sot_token, eot_token=eot_token, context_length=context_length)
            all_tokens.append(tokens)
            all_labels.append(torch.tensor(labels))
        all_tokens = truncate_tokens(all_tokens, context_length, eot_token)
    else:
        all_tokens = [tokenizer.encode(text) for text in texts]
        all_tokens = truncate_tokens(all_tokens, context_length, eot_token)
        all_labels = []
        if mask_prob:
            import copy
            masked_tokens = [MaskTokens(copy.deepcopy(tokens), mask_type='MLM', mask_token=mask_token,
                                        special_tokens=[sot_token, eot_token, mask_token],
                                        tokenizer_length=tokenizer.vocab_size, mlm_probability=mask_prob,
                                        unmask_flag=unmask_flag) for tokens in all_tokens]
            all_tokens = [item[0] for item in masked_tokens]
            all_labels = [item[1] for item in masked_tokens]

    # all_tokens = [tokenizer.encode(text) for text in texts]
    # all_labels = []
    #
    #
    # # all_tokens = truncate_tokens(all_tokens, context_length, eot_token)
    # for i, tokens in enumerate(all_tokens):
    #     if len(tokens) > context_length:
    #         tokens = tokens[:context_length]  # Truncate
    #         tokens[-1] = eot_token
    #     all_tokens[i] = torch.tensor(tokens)
    # t2 = time.time()
    #
    # if mask:
    #     import copy
    #     special_tokens = [sot_token, eot_token, mask_token]
    #     masked_tokens = [MaskTokens(copy.deepcopy(all_tokens[i]), mask_type='MLM', mask_token=mask_token, special_tokens=special_tokens,
    #                                 tokenizer_length=tokenizer.vocab_size, mlm_probability=0.3, unmask_flag=unmask_flag) for i in range(len(all_tokens))]
    #     all_tokens = [item[0] for item in masked_tokens]
    #     all_labels = [item[1] for item in masked_tokens]


    # pad_token is 0
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    labels = torch.ones(len(all_tokens), context_length, dtype=torch.long) * unmask_flag
    token_lengths = torch.ones(len(all_tokens), dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        result[i, :len(tokens)] = tokens
        token_lengths[i] = min(len(tokens), context_length)
        if mask_prob:
            labels[i, :len(tokens)] = all_labels[i]

    result = result.to(device, non_blocking=True)

    if generator:
        pad_mask = result != pad_token
        logits = generator(result, pad_mask)[0]
        sampled_logits = logits[labels != unmask_flag]
        sampled_tokens = gumbel_sample(sampled_logits, temperature=gumbel_t)
        generate = result.clone()
        generate[labels != unmask_flag] = sampled_tokens


        if return_generation:
            # pdb.set_trace()
            de_texts = [tokenizer.decode(tmp).replace('[PAD]', '').replace('[SEP]', '').replace('[CLS]', '').strip(' ') for tmp in generate]
            # if word_parsing_mask:
            #     texts = [texts[i] for i in range(len(mask_texts)) if mask_texts[i]]
            # for i in range(len(texts)):
            #     print(texts[i])
            #     print(de_texts[i])
            #     print()
            return de_texts, generate, logits, labels, pad_token


        return generate
    if mask_prob:
        return result, labels, token_lengths
    else:
        return result



def gumbel_sample(t, temperature = 1.):
    def log(t, eps=1e-9):
        return torch.log(t + eps)
    def gumbel_noise(t):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -log(-log(noise))
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)
