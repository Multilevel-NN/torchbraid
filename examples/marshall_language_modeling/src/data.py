## Partially taken from Karpathy's github: [url]

import os
import torch
# from transformers import GPT2Tokenizer  <-- below
from tqdm import tqdm
from transformers import GPT2TokenizerFast

## Following taken from nanoGPT for openwebtext
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

def obtain_data(data_dir, input_text, tokenization, percent_data=1):
  # Honestly should've rewrote this to just download from huggingface...
  if input_text == 'openwebtext':
    # If using openwebtext, please create the correct preprocess data using the notebook first
    train_data_path = os.path.join(data_dir, 'train.bin')
    val_data_path = os.path.join(data_dir, 'val.bin')
    enc = tiktoken.get_encoding("gpt2")
    train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r').astype(np.int64)
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r').astype(np.int64)
    print(f"Loaded from {train_data_path} and {val_data_path}")
    print(f'Truncating {len(train_data)=}, {len(val_data)=} {percent_data=}')
    train_data  = train_data[:int(percent_data * len(train_data))]
    val_data    = val_data[:int(percent_data * len(val_data))]
    print(f'{len(train_data)=}, {len(val_data)=}')
    return train_data, val_data, enc.decode, enc.n_vocab




  data_path = os.path.join(data_dir, input_text + '.txt')
  data_file = os.path.join(data_dir, input_text + '.data')
  data = []
  counter = 0 

  # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-tokenizer')
  decode = tokenizer.decode
  vocab_size = tokenizer.vocab_size

  # Counted number of lines on linux to be 2966378
  try:
    # Attempt to load the tensor
    data = torch.load(data_file)
    print(f"Loaded tensor from {data_file}")
    
  except FileNotFoundError:
    # File not found, save the tensor
    print(f"Tokenized data not found; creating and saving for future.")
    print('Wikipedia takes roughly 5 minutes to load on A100')
    with open(data_path, 'r', encoding='utf-8') as f:
      # Use tqdm to iterate through lines with a description
      for line in tqdm(f, desc="Tokenizing", total=2966378):
        text = line.strip()  # Strip whitespace from each line

        # Check if it's a blank line (after stripping)
        if not text:
          continue

        # Tokenize and process the text
        data += tokenizer(text)['input_ids']
    
    data = torch.tensor(data, dtype=torch.long)
    torch.save(data, data_file)
    print(f'Total number of tokens: {len(data)}')
    print(f"Saved tensor to {data_file}")

  print('1.4 Splitting data into training and validation data')
  n = int(.9*len(data))
  train_data, val_data = data[:n], data[n:]
  print(f'{len(train_data)=}, {len(val_data)=} {percent_data=}')
  train_data  = train_data[:int(percent_data * len(train_data))]
  val_data    = val_data[:int(percent_data * len(val_data))]
  print(f'{len(train_data)=}, {len(val_data)=}')

  return train_data, val_data, decode, vocab_size















