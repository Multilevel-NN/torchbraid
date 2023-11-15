## Partially taken from Karpathy's github: [url]

import os
import torch
# from transformers import GPT2Tokenizer, GPT2Model  <-- below

def obtain_data(data_dir, input_text, tokenization):
  data_path = os.path.join(data_dir, input_text + '.txt')

  print('1.1 Reading text')
  with open(data_path, 'r', encoding='utf-8') as f:
      text = f.read()

  if tokenization == 'character':
    print('1.2 Building character-level tokenizer')
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    print('1.3 Encoding data')
    data = torch.tensor(encode(text), dtype=torch.long)

  elif tokenization == 'gpt2':
    from transformers import GPT2Tokenizer, GPT2Model

    print('1.2 Obtaining gpt2 tokenizer')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # tokenizer.pad_token = '<pad>'
    decode = tokenizer.decode
    vocab_size = tokenizer.vocab_size

    print('1.3 Encoding data')
    data = tokenizer(text)['input_ids']

  else: raise Exception()

  print('1.4 Splitting data into training and validation data')
  n = int(.9*len(data))
  train_data, val_data = data[:n], data[n:]

  return train_data, val_data, decode, vocab_size
















