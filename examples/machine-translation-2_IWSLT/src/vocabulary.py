import torch

class Vocabulary:
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

  def decode(self, id_list, break_at_pad=False):
    if type(id_list) == list: assert type(id_list[0]) == int
    if type(id_list) == torch.Tensor: 
      assert id_list.ndim == 1
      id_list = id_list.tolist()

    # return list(map(lambda id: self.decode_id(id), id_list))
    tokens_list = []
    for id in id_list:
      if   id == self.eos_id: break
      elif id == self.pad_id and break_at_pad: break
      elif id == self.pad_id: pass
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



