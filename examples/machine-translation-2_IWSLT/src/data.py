from   datasets           import load_dataset
import inspect
import os
import pickle
import torch
from   torch.nn.utils.rnn import pad_sequence
from   torch.utils.data   import Dataset as tDataset, DataLoader
from   tqdm               import tqdm
import sys

from _utils               import Timeit
from vocabulary           import Vocabulary

DATASET_NAME = ('iwslt2017', 'iwslt2017-de-en')
DOWNLOADED_DATASET_DIR  = os.path.join('..', 'data')
DOWNLOADED_DATASET_DIR2 = '/Users/marcsalvado/Desktop/Aux-Scripts-python/85_cheap-Transformer-IWSLT/data'
DOWNLOADED_DATASET_FNM_ = lambda lang, split: f'{lang}-{split}-dataset.txt'
DOWNLOADED_TOKENIZERS_DIR  = os.path.join('..', 'tokenizers')
DOWNLOADED_TOKENIZERS_DIR2 = '/Users/marcsalvado/Desktop/Aux-Scripts-python/85_cheap-Transformer-IWSLT/tokenizers'
LANGUAGES    = ('de', 'en')
SPLITS       = ('train', 'validation', 'test')
SEP          = '!@#$\n'

class Dataset(tDataset):
  def __init__(
    self, raw_dataset, tokenizers, vocabs, extend_vocabulary, debug,
  ): 
    de_tokenizer, en_tokenizer = (tokenizers[lang] for lang in LANGUAGES)
    de_vocab    , en_vocab     = (    vocabs[lang] for lang in LANGUAGES)

    tokenized_sentences = []
    # max_de_length, max_en_length = -1, -1

    de_sentences = raw_dataset['de']
    en_sentences = raw_dataset['en']

    for k, (de_sentence, en_sentence) in enumerate(
      tqdm(zip(de_sentences, en_sentences))
    ):
      de_tokens = [token.text for token in de_tokenizer(de_sentence)]
      en_tokens = [token.text for token in en_tokenizer(en_sentence)]
      en_tokens.insert(0, en_vocab.bos_token)
      en_tokens.append(en_vocab.eos_token)

      # max_de_length = max(max_de_length, len(de_tokens))
      # max_en_length = max(max_en_length, len(en_tokens))

      tokenized_sentences.append((de_tokens, en_tokens))

      if debug and k == 20: break  # debug

    max_sequence_length = max(max(len(de_tokens), len(en_tokens)-1) 
                            for (de_tokens, en_tokens) in tokenized_sentences)

    batch_size = len(tokenized_sentences)

    self.data = {
      'de_ids': torch.empty((batch_size, max_sequence_length), 
                                     dtype=torch.long).fill_(de_vocab.pad_id),
      'en_ids': torch.empty((batch_size, max_sequence_length+1), 
                                     dtype=torch.long).fill_(en_vocab.pad_id),
    }

    for idx, (de_tokens, en_tokens) in enumerate(tokenized_sentences):
      # de_tokens.extend([de_vocab.pad_token] * (max_de_length - len(de_tokens)))
      # en_tokens.extend([en_vocab.pad_token] * (max_en_length - len(en_tokens)))

      de_ids = de_vocab.encode(de_tokens, extend_vocabulary)
      en_ids = en_vocab.encode(en_tokens, extend_vocabulary)

      de_ids = torch.LongTensor(de_ids)
      en_ids = torch.LongTensor(en_ids)

      self.data['de_ids'][idx, :len(de_ids)] = de_ids
      self.data['en_ids'][idx, :len(en_ids)] = en_ids

    # self.data['de_ids'] = torch.LongTensor(self.data['de_ids'])
    # self.data['en_ids'] = torch.LongTensor(self.data['en_ids'])

  def __getitem__(self, idx): 
    return self.data['de_ids'][idx], self.data['en_ids'][idx]

  def __len__(self): return len(self.data['de_ids'])

  # def transpose(self, x):
  #   if   type(x) == torch.Tensor: return x.T
  #   elif type(x) == list        : return list(map(list, zip(*x)))
  #   else: raise Exception("Type unknown")

def download_dataset(): 
  return load_dataset(*DATASET_NAME, trust_remote_code=True)  # needed for Colab

def get_collate_fn(device, vocabs):
  def padding_collate_fn(batch):
    batch_size = len(batch)

    max_sequence_length = max(max(len(de_tokens), len(en_tokens)-1) 
                              for (de_tokens, en_tokens) in batch)

    de_batch = torch.empty((batch_size, max_sequence_length), 
                           dtype=torch.long).fill_(de_vocab.pad_id)
    en_batch = torch.empty((batch_size, max_sequence_length+1), 
                           dtype=torch.long).fill_(en_vocab.pad_id)

    for batch_idx, (de_tokens, en_tokens) in enumerate(batch):
      de_batch[batch_idx, :len(de_tokens)] = de_tokens
      en_batch[batch_idx, :len(en_tokens)] = en_tokens

    # for de_tokens, en_tokens in batch:
    #   de_batch.append(torch.LongTensor(de_tokens))
    #   en_batch.append(torch.LongTensor(en_tokens))

    # de_batch = pad_sequence(
    #   de_batch, batch_first=True, padding_value=de_vocab.pad_id,
    # )
    # en_batch = pad_sequence(
    #   en_batch, batch_first=True, padding_value=en_vocab.pad_id,
    # )

    return (de_batch, en_batch)
 
  de_vocab, en_vocab = vocabs['de'], vocabs['en']
  return padding_collate_fn
  
@Timeit
def get_data(device, debug, num_beams, download_tokenizers, 
             **data_loader_config_dict):
  # raw_datasets = download_dataset()
  raw_datasets = get_raw_datasets()
  tokenizers = get_tokenizers(download_tokenizers)
  vocabs = {lang: Vocabulary() for lang in LANGUAGES}
  datasets, data_loaders = {}, {}

  # data_loader_config_dict['collate_fn'] = get_collate_fn(device, vocabs)

  for split in SPLITS:
    extend_vocabulary = shuffle_data = (split == 'train')
    dataset = Dataset(
      raw_datasets[split], tokenizers, vocabs, extend_vocabulary, debug,
    )
    data_loader_config_dict_copy = \
       {k: v for (k, v) in data_loader_config_dict.items()
             if k in inspect.signature(DataLoader).parameters} \
     | {'shuffle': shuffle_data}

    if split == 'validation': 
      data_loader_config_dict_copy['batch_size'] //= num_beams

    data_loader = DataLoader(dataset, **data_loader_config_dict_copy)
    datasets    [split] = dataset
    data_loaders[split] = data_loader

  return datasets, data_loaders, vocabs

def get_raw_datasets(download=False):
  # return download_dataset() if download else \
  #        load_and_process_downloaded_data()
  if download: raise NotImplementedError()
  return load_and_process_downloaded_data()

def get_tokenizers(download):
  if download: 
    import spacy
    return {
      'de': spacy.load('de_core_news_sm').tokenizer,#de_core_news_sm.load().tokenizer,
      'en': spacy.load('en_core_web_sm' ).tokenizer,#en_core_web_sm .load().tokenizer,
    }
  else: 
    for _dir in [DOWNLOADED_TOKENIZERS_DIR, DOWNLOADED_TOKENIZERS_DIR2]:
      try: 
        with open(os.path.join(_dir, 'de_tokenizer.pkl'), 'rb') as f: 
          de_tokenizer = pickle.load(f)
        with open(os.path.join(_dir, 'en_tokenizer.pkl'), 'rb') as f: 
          en_tokenizer = pickle.load(f)
      except: pass
      else  : break
    return dict(zip(('de', 'en'), (de_tokenizer, en_tokenizer)))

def load_and_process_downloaded_data():
  raw_datasets = {}

  for split in SPLITS:
    split_raw_datasets = {}

    for lang in LANGUAGES:
      downloaded_dataset_fnm = DOWNLOADED_DATASET_FNM_(lang, split)
      for downloaded_dataset_dir in [
        DOWNLOADED_DATASET_DIR, DOWNLOADED_DATASET_DIR2
      ]:
        try:
          downloaded_dataset_path = os.path.join(
            downloaded_dataset_dir, downloaded_dataset_fnm
          )
          with open(downloaded_dataset_path, 'r') as f:
            split_raw_datasets[lang] = f.read().split(SEP)[:-1]
        except: pass
        else  : break

    raw_datasets[split] = split_raw_datasets

  return raw_datasets




