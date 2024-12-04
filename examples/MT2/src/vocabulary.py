import os
import sentencepiece
import torch

MODEL_PREFIX_ = lambda tokenization, vocab_size_abbrev: \
                                   f'de-en_{tokenization}_{vocab_size_abbrev}'

class Vocabulary:
  sep = ' '

  def __init__(self):
    self.special_tokens = ('<pad>', '<unk>', '<bos>', '<eos>')
    self.id2tok = dict(enumerate(self.special_tokens))
    self.tok2id = {v: k for (k, v) in self.id2tok.items()}

    for token in self.special_tokens:
      setattr(self, f'{token[1:-1]}_token', token)
      setattr(self, f'{token[1:-1]}_id'   , self.tok2id[token])

  def __getitem__(self, x):
    if   type(x) == str: return self.tok2id[x]
    elif type(x) == int: return self.id2tok[x]
    else: raise Exception('element must be of type int or str.')

  def __len__(self): return len(self.tok2id)

  def add_token(self, token):
    if token not in self.tok2id:
      self.tok2id[token] = len(self.tok2id)
      self.id2tok[self.tok2id[token]] = token

  def decode(self, id_list):
    if type(id_list) == list: assert type(id_list[0]) == int
    if type(id_list) == torch.Tensor: 
      assert id_list.ndim == 1
      id_list = id_list.tolist()

    # return list(map(lambda id: self.decode_id(id), id_list))
    tokens_list = []
    for id in id_list:
      if   id == self.eos_id: break
      elif id == self.pad_id: break
      else: tokens_list.append(self.decode_id(id))
      # elif id == self.pad_id: raise Exception(f'{self.pad_token} should not be given any probability.')
      # elif id == self.bos_id: raise Exception(f'{self.bos_token} should not be given any probability.')

    return tokens_list

  def decode_id(self, id):
    if id not in self.id2tok: raise Exception(f'Unknown token id: {id}')
    return self.id2tok[id]

  def encode(self, token_list, extend_vocab=False):
    return list(
      map(lambda token: self.encode_token(token, extend_vocab), token_list)
    )

  def encode_token(self, token, extend_vocab=False):
    if token in self.tok2id or extend_vocab: 
      if token not in self.tok2id: self.add_token(token)
      return self.tok2id[token]

    else: return self.unk_id

class SPVocabulary:
  sep = ''
  
  pad_token = '<pad>'
  unk_token = '<unk>'
  bos_token = '<s>' #'<bos>'
  eos_token = '</s>'#'<eos>'

  pad_id = 0#-1#-2
  unk_id = 1# 0
  bos_id = 2# 1#-1
  eos_id = 3# 2# 0

  def __init__(self, tokenization, vocab_size):#vocab_size_abbrev):
    assert tokenization in ['bpe', 'unigram']
    assert (vocab_size_abbrev := str(vocab_size).replace('000', 'k')) \
                                        in ['4k', '8k', '16k', '32k']
    processor_fnm = f'{MODEL_PREFIX_(tokenization, vocab_size_abbrev)}.model'

    try: 
        processor_path = os.path.join('..', 'tokenizers', 
                                             tokenization, processor_fnm)
        self.processor = sentencepiece.SentencePieceProcessor(processor_path)
    except:
      try:
        processor_path = os.path.join(
          '/Users/marcsalvado/Desktop/Aux-Scripts-python/85_cheap-Transformer-IWSLT/tokenizers',
          tokenization, processor_fnm,
        )
        self.processor = sentencepiece.SentencePieceProcessor(processor_path)
      except: raise Exception()
    print(f'Vocab processor loaded successfully. Processor: {self.processor}')

  def __len__(self): return self.processor.vocab_size()

  def decode(self, id_list):
    if type(id_list) == list: assert all(type(id) == int for id in id_list)
    if type(id_list) == torch.Tensor: 
      assert id_list.ndim == 1
      id_list = id_list.tolist()

    if self.eos_id in id_list: id_list = id_list[:id_list.index(self.eos_id)]
    # id_list = list(filter(self.pad_id.__ne__, id_list))  # should not be necessary

    return self.processor.decode(id_list)

  def encode(self, sentence): return self.processor.encode(sentence)




