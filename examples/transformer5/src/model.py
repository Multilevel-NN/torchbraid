import math
import torch
import torch.nn as nn
from torch import Tensor

## Taken from PyTorch website: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
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
###

class Transformer(nn.Module):
  def __init__(self, dim_voc_src, dim_voc_tgt, h, idx_pad_src, idx_pad_tgt,
                                                            max_len_tgt, dev):
    super(Transformer, self).__init__()
    self.emb_src = nn.Embedding(dim_voc_src, h)
    self.emb_tgt = nn.Embedding(dim_voc_tgt, h)
    self.posenc = PositionalEncoding(h)
    self.transformer = nn.Transformer(d_model=h, nhead=8, num_encoder_layers=3, 
      num_decoder_layers=3, dim_feedforward=1024, dropout=.1, batch_first=False)#True)
    self.fc = nn.Linear(h, dim_voc_tgt)
    self.idx_pad_src, self.idx_pad_tgt = idx_pad_src, idx_pad_tgt
    self.h, self.L = h, max_len_tgt-1
    self.dev = dev

  def forward(self, src, tgt):
    mask_pad_src = (src == self.idx_pad_src)
    mask_pad_tgt = (tgt == self.idx_pad_tgt)
    mask_pad_mem = mask_pad_src
    src, tgt = src.T, tgt.T
    mask_tgt = nn.Transformer.generate_square_subsequent_mask(self.L).to(self.dev)
    src, tgt = self.emb_src(src), self.emb_tgt(tgt)
    src, tgt = self.posenc(src), self.posenc(tgt)
    # with open('log.txt', 'w') as f: f.write(' '.join([str(x) for x in (src.shape, tgt.shape, mask_pad_src.shape, mask_pad_tgt.shape, mask_tgt.shape)]))
    output = self.transformer(src=src, tgt=tgt, tgt_mask=mask_tgt,
      src_key_padding_mask=mask_pad_src, tgt_key_padding_mask=mask_pad_tgt, 
      memory_key_padding_mask=mask_pad_mem)
    # output = self.fc(output.reshape(-1, self.h)).reshape(-1, self.L, self.fc.out_features)
    output = self.fc(output).transpose(0, 1)
    return output



































