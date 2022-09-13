""" CLIP tokenizer

Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import gzip
import html
import os
from functools import lru_cache
from typing import Union, List

import ftfy
import regex as re
import torch

from transformers import ElectraConfig, ElectraForMaskedLM

@lru_cache()
def default_bpe():
   return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
   """
   Returns list of utf-8 byte and a corresponding list of unicode strings.
   The reversible bpe codes work on unicode strings.
   This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
   When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
   This is a signficant percentage of your normal, say, 32K bpe vocab.
   To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
   And avoids mapping to whitespace/control characters the bpe code barfs on.
   """
   bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
   cs = bs[:]
   n = 0
   for b in range(2**8):
       if b not in bs:
           bs.append(b)
           cs.append(2**8+n)
           n += 1
   cs = [chr(n) for n in cs]
   return dict(zip(bs, cs))


def get_pairs(word):
   """Return set of symbol pairs in a word.
   Word is represented as tuple of symbols (symbols being variable-length strings).
   """
   pairs = set()
   prev_char = word[0]
   for char in word[1:]:
       pairs.add((prev_char, char))
       prev_char = char
   return pairs


def basic_clean(text):
   text = ftfy.fix_text(text)
   text = html.unescape(html.unescape(text))
   return text.strip()


def whitespace_clean(text):
   text = re.sub(r'\s+', ' ', text)
   text = text.strip()
   return text


class SimpleTokenizer(object):
   def __init__(self, bpe_path: str = default_bpe(), special_tokens=None):
       self.byte_encoder = bytes_to_unicode()
       self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
       merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
       merges = merges[1:49152-256-2+1]
       merges = [tuple(merge.split()) for merge in merges]
       vocab = list(bytes_to_unicode().values())
       vocab = vocab + [v+'</w>' for v in vocab]
       for merge in merges:
           vocab.append(''.join(merge))
       if not special_tokens:
           special_tokens = ['<start_of_text>', '<end_of_text>']
       else:
           special_tokens = ['<start_of_text>', '<end_of_text>'] + special_tokens
       vocab.extend(special_tokens)
       self.encoder = dict(zip(vocab, range(len(vocab))))
       self.decoder = {v: k for k, v in self.encoder.items()}
       self.bpe_ranks = dict(zip(merges, range(len(merges))))
       self.cache = {t:t for t in special_tokens}
       special = "|".join(special_tokens)
       self.pat = re.compile(special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

       self.vocab_size = len(self.encoder)
       self.all_special_ids = [self.encoder[t] for t in special_tokens]

   def bpe(self, token):
       if token in self.cache:
           return self.cache[token]
       word = tuple(token[:-1]) + ( token[-1] + '</w>',)
       pairs = get_pairs(word)

       if not pairs:
           return token+'</w>'

       while True:
           bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
           if bigram not in self.bpe_ranks:
               break
           first, second = bigram
           new_word = []
           i = 0
           while i < len(word):
               try:
                   j = word.index(first, i)
                   new_word.extend(word[i:j])
                   i = j
               except:
                   new_word.extend(word[i:])
                   break

               if word[i] == first and i < len(word)-1 and word[i+1] == second:
                   new_word.append(first+second)
                   i += 2
               else:
                   new_word.append(word[i])
                   i += 1
           new_word = tuple(new_word)
           word = new_word
           if len(word) == 1:
               break
           else:
               pairs = get_pairs(word)
       word = ' '.join(word)
       self.cache[token] = word
       return word

   def encode(self, text):
       bpe_tokens = []
       text = whitespace_clean(basic_clean(text)).lower()
       for token in re.findall(self.pat, text):
           token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
           bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
       return bpe_tokens

   def decode(self, tokens):
       text = ''.join([self.decoder[token] for token in tokens])
       text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
       return text


_tokenizer = SimpleTokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, mask_type=None) -> torch.LongTensor:
   """
   Returns the tokenized representation of given input string(s)

   Parameters
   ----------
   texts : Union[str, List[str]]
       An input string or a list of input strings to tokenize
   context_length : int
       The context length to use; all CLIP models use 77 as the context length

   Returns
   -------
   A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
   """
   if isinstance(texts, str):
       texts = [texts]

   sot_token = _tokenizer.encoder["<start_of_text>"]
   eot_token = _tokenizer.encoder["<end_of_text>"]
   all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
   all_labels = []
   result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

   for i, tokens in enumerate(all_tokens):
       if len(tokens) > context_length:
           tokens = tokens[:context_length]  # Truncate
           tokens[-1] = eot_token
       result[i, :len(tokens)] = torch.tensor(tokens)

   if mask_type is not None:
       mask_token = _tokenizer.encoder["<|mask|>"]
       special_tokens = [sot_token, eot_token, mask_token]
       masked_tokens = [MaskTokens(tokens, mask_type=mask_type, mask_token=mask_token, special_tokens=special_tokens,
                                   tokenizer_length=len(_tokenizer.encoder)) for tokens in all_tokens]
       all_tokens = [item[0] for item in masked_tokens]
       all_labels = [item[1] for item in masked_tokens]

   result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
   labels = torch.ones(len(all_tokens), context_length, dtype=torch.long) * -100
   token_lengths = torch.ones(len(all_tokens), dtype=torch.long)

   for i, tokens in enumerate(all_tokens):
       result[i, :len(tokens)] = tokens
       token_lengths[i] = min(len(tokens), context_length)
       if mask_type is not None:
           labels[i, :len(tokens)] = all_labels[i]

   if mask_type:
       # print(result[0], labels[0], '<< masking', flush=True)
       return result, labels, token_lengths
   else:
       return result

def mask_tokens(inputs, special_tokens, mask_token, tokenizer_length, mlm_probability=0.15, special_tokens_mask=None):
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
   labels[~masked_indices] = -100  # We only compute loss on masked tokens

   # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
   indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
   inputs[indices_replaced] = mask_token

   # 10% of the time, we replace masked input tokens with random word
   indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
   random_words = torch.randint(tokenizer_length, labels.shape, dtype=torch.long)
   inputs[indices_random] = random_words[indices_random]

   # The rest of the time (10% of the time) we keep the masked input tokens unchanged
   return inputs, labels


def MaskTokens(tokens, mask_type, mask_token, special_tokens=None, tokenizer_length=None, sepcial_tokens_mask=None, special_tokens_mask=None):
   if mask_type == 'MLM':
       tokens, labels = mask_tokens(inputs=tokens, special_tokens=special_tokens, mask_token=mask_token, tokenizer_length=tokenizer_length, special_tokens_mask=special_tokens_mask)
   else:
       raise NotImplementedError(mask_type)
   return tokens, labels


