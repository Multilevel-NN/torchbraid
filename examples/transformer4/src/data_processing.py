## Example of de-en translation taken from PyTorch website: https://torchtutorialstaging.z5.web.core.windows.net/beginner/translation_transformer.html

import math
import torchtext
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
from torch import Tensor
import io
import time

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def build_vocab(filepath, tokenizer):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
    for string_ in f:
      counter.update(tokenizer(string_))
  return Vocab(counter)#, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

def data_process(filepaths):
  raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
  raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
  data = []
  for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
    de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de.rstrip("\n"))],
                            dtype=torch.long)
    en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en.rstrip("\n"))],
                            dtype=torch.long)
    data.append((de_tensor_, en_tensor_))
  return data


train_data = data_process(train_filepaths)
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)

train_data_aux, val_data_aux, test_data_aux = train_data, val_data, test_data
train_data, val_data, test_data = [], [], []

max_len_de, max_len_en = -1, -1
for tensor_de, tensor_en in train_data_aux + val_data_aux + test_data_aux:
  max_len_de = max(max_len_de, tensor_de.shape[0])
  max_len_en = max(max_len_en, tensor_en.shape[0])

for data_aux, data in [(train_data_aux, train_data), (val_data_aux, val_data), (test_data_aux, test_data)]:
  for tensor_de, tensor_en in data_aux:
    pad_de = torch.zeros(size=(max_len_de - tensor_de.shape[0],), dtype=torch.long)
    pad_en = torch.zeros(size=(max_len_en - tensor_en.shape[0],), dtype=torch.long)
    tensor_de = torch.cat((tensor_de, pad_de), axis=0)
    tensor_en = torch.cat((tensor_en, pad_en), axis=0)
    data.append((tensor_de, tensor_en))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 2#128
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']













