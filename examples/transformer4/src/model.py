import math
import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
  ## Taken from Pytorch: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: Tensor) -> Tensor:
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)

class Transformer(nn.Module):
  def __init__(self, dev, voc_src, voc_tgt, h, n_lays_enc, n_lays_dec):
    super(Transformer, self).__init__()
    nclasses = len(voc_tgt)
    self.emb_src = nn.Embedding(len(voc_src), h)
    self.emb_tgt = nn.Embedding(nclasses, h)
    self.posenc = PositionalEncoding(d_model=h)
    self.transformer = nn.Transformer(d_model=h, nhead=8, dim_feedforward=1024,
      num_encoder_layers=n_lays_enc, num_decoder_layers=n_lays_dec, 
      dropout=.1, batch_first=False)
    self.fc = nn.Linear(h, nclasses)
    self.idx_pad_src, self.idx_pad_tgt = voc_src['<pad>'], voc_tgt['<pad>']
    self.dev = dev
    self.to(dev)

  def forward(self, src, tgt):
    n, L = tgt.shape
    mask_pad_src = (src == self.idx_pad_src)
    mask_pad_tgt = (tgt == self.idx_pad_tgt)
    mask_pad_mem = mask_pad_src 
    mask_tgt = nn.Transformer.generate_square_subsequent_mask(L).to(self.dev)
    src, tgt = src.T, tgt.T
    src, tgt = self.emb_src(src), self.emb_tgt(tgt)
    src, tgt = self.posenc(src), self.posenc(tgt)
    out = self.transformer(src=src, tgt=tgt, tgt_mask=mask_tgt, 
      src_key_padding_mask=mask_pad_src, tgt_key_padding_mask=mask_pad_tgt, 
      memory_key_padding_mask=mask_pad_mem)
    out = self.fc(out).transpose(0, 1)
    return out































