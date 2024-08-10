## Taken from https://pytorch.org/tutorials/beginner/translation_transformer.html
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout, batch_first, max_length=5000):
    super().__init__()
    den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
    pos = torch.arange(0, max_length).reshape(max_length, 1)
    pos_embedding = torch.zeros((max_length, d_model))
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    pos_embedding = pos_embedding.unsqueeze(0) if batch_first else \
                    pos_embedding.unsqueeze(1)  # [1, Lmax, d] | [Lmax, 1, d]
  
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('pos_embedding', pos_embedding)

    self.batch_first = batch_first

  def forward(self, token_embedding):
    if self.batch_first:  # token_embedding: [b, L, d]
      L = token_embedding.shape[1]
      return self.dropout(token_embedding + self.pos_embedding[:, :L])

    else:                 # token_embedding: [L, b, d]
      L = token_embedding.shape[0]
      return self.dropout(token_embedding + self.pos_embedding[:L])  # [L, b, d]




