import torch
from transformers import ElectraTokenizerFast, ElectraForMaskedLM
from mask_tokens import MaskTokens


PRETRAINED_ELECTRA_GENERATORS = {
    'small': "../electra_configs/electra-small-generator/",
    'base': "../electra_configs/electra-base-generator/",
    'large': "../electra_configs/electra-large-generator/"
}

ELECTRA_TOKENIZER_VOCAB = "../electra_configs/vocab/"

tokenizer = ElectraTokenizerFast.from_pretrained(ELECTRA_TOKENIZER_VOCAB)
generator = ElectraForMaskedLM.from_pretrained(PRETRAINED_ELECTRA_GENERATORS['small'])


def tokenize(texts, context_length=77, mask=False, resample=False, gumbel_t=1.):
    assert mask and ~resample, 'mask is required for enabling resample!'

    unmask_flag = -1
    sot_token = tokenizer.encode('[CLS]')[1]
    eot_token = tokenizer.encode('[SEP]')[1]
    pad_token = tokenizer.encode('[PAD]')[1]
    mask_token = tokenizer.encode('[MASK]')[1]
    all_tokens = [tokenizer.encode(text) for text in texts]
    all_labels = []

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eot_token
        all_tokens[i, :len(tokens)] = torch.tensor(tokens)

    if mask:
        special_tokens = [sot_token, eot_token, mask_token]
        masked_tokens = [MaskTokens(tokens, mask_type='MLM', mask_token=mask_token, special_tokens=special_tokens,
                                    tokenizer_length=tokenizer.vocab_size, unmask_flag=unmask_flag) for tokens in all_tokens]
        all_tokens = [item[0] for item in masked_tokens]
        all_labels = [item[1] for item in masked_tokens]

    # pad_token is 0
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    labels = torch.ones(len(all_tokens), context_length, dtype=torch.long) * unmask_flag
    token_lengths = torch.ones(len(all_tokens), dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        result[i, :len(tokens)] = tokens
        token_lengths[i] = min(len(tokens), context_length)
        if mask:
            labels[i, :len(tokens)] = all_labels[i]

    if resample:
        pad_mask = result != pad_token
        logits = generator(result, pad_mask)[0]
        sampled_logits = logits[labels == mask_token]
        sampled_tokens = gumbel_sample(sampled_logits, temperature=gumbel_t)
        generate = result.clone()
        generate[labels == mask_token] = sampled_tokens.detach()
        return generate
    if mask:
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