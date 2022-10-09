import pdb
import numpy as np
import random
import torch
from typing import Tuple, List


def mask_tokens(inputs, special_tokens, mask_token, tokenizer_length, mlm_probability, unmask_flag, special_tokens_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [1 if val in special_tokens else 0 for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    # if tokenizer._pad_token is not None:
    #     padding_mask = labels.eq(tokenizer.pad_token_id)
    #     probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = unmask_flag  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
    inputs[indices_replaced] = mask_token

    # 10% of the time, we replace masked input tokens with random word
    # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    # random_words = torch.randint(tokenizer_length, labels.shape, dtype=torch.long)
    # inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def MaskTokens(tokens, mask_type, mask_token, special_tokens=None, tokenizer_length=None, mlm_probability=0.15, unmask_flag=-1, special_tokens_mask=None):
    if mask_type == 'MLM':
        tokens, labels = mask_tokens(inputs=tokens, special_tokens=special_tokens, mask_token=mask_token, tokenizer_length=tokenizer_length, mlm_probability=mlm_probability, unmask_flag=unmask_flag, special_tokens_mask=special_tokens_mask)
    else:
        raise NotImplementedError(mask_type)
    return tokens, labels


def SelectMaskTokensFromText(text, tokenizer, unmask_flag, mask_prob=0.3, sot_token=1, eot_token=2, mask='[MASK]', context_length=77):

    mask_pos = ['NN', 'NNS', 'NNP', 'NNPS',
                'JJ', 'JJR', 'JJS']

    import nltk
    text = nltk.wordpunct_tokenize(text)
    pos = nltk.pos_tag(text)
    raw_words = np.array(pos)[:, 0]
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

    tokens = [sot_token]
    labels = [unmask_flag]
    assert len(words) == len(raw_words)
    for i in range(len(words)):
        if words[i] == mask:
            raw_token = tokenizer.encode(raw_words[i])[1:-1]
            tokens += tokenizer.encode(mask)[1:-1] * len(raw_token)
            labels += raw_token
        else:
            token = tokenizer.encode(words[i])[1:-1]
            tokens += token
            labels += [unmask_flag] * len(token)
        if len(tokens) >= context_length - 1:
            tokens = tokens[:context_length - 1]
            labels = labels[:context_length - 1]
            break

    tokens += [eot_token]
    labels += [unmask_flag]

    return tokens, labels


