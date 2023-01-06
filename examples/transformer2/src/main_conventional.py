import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import preproc
from models import PositionalEncoding, PE_Alternative

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, metavar='N',    # 32; 128, 256, 512, ... until it crashes
                  help='input batch size for training (default: 50)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                  help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                  help='learning rate (default: 0.01)')
parser.add_argument('--encoding', type=str, default='Torch',
                  help='which positional encoding will be used for the attention')
parser.add_argument('--output_fn',type=str, required=True, 
                  help='Output filename (for model saving)')
parser.add_argument('--models_dir',type=str, required=True, 
                  help='Models directory (for model saving)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                  help='random seed (default: 783253419)')
args = parser.parse_args()

class Transformer(nn.Module):
  def __init__(self, dim_src, dim_tgt, encoding):
    super(Transformer, self).__init__()
    self.encoding = encoding
    self.emb_src = nn.Embedding(dim_src, 128)
    self.emb_tgt = nn.Embedding(dim_tgt, 128)
    # self.dropout = nn.Dropout(p=.1)
    self.posenc = PositionalEncoding(128) if encoding == 'Torch'\
      else PE_Alternative(128) if encoding == 'Alternative'\
      else Exception('encoding unknown')

    self.transformer = nn.Transformer(
      d_model=128, nhead=8, num_encoder_layers=3,
      num_decoder_layers=3, dim_feedforward=1024,
      dropout=.3, batch_first=True)

    self.fc = nn.Linear(128, dim_tgt)

  def forward(self, src, tgt):
    msk_tgt = nn.Transformer.generate_square_subsequent_mask(
      sz=tgt.shape[1]).to(src.device)
    msk_pad_src = (src == 0)
    msk_pad_mem = msk_pad_mem
    msk_pad_tgt = (tgt == 0)

    src = self.emb_src(src)
    tgt = self.emb_tgt(tgt)

    out = self.transformer(src=src, tgt=tgt, tgt_mask=msk_tgt, 
      src_key_padding_mask=msk_pad_src,
      tgt_key_padding_mask=msk_pad_tgt, 
      memory_key_padding_mask=msk_pad_mem,)

    out = self.fc(out)

    return out


def main():
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  torch.manual_seed(args.seed)

  vocs, sents = preproc.main(small=True)
  voc_de, voc_en = vocs
  sents_de_tr, sents_en_tr, sents_de_te, sents_en_te = sents
  # ds_tr, ds_te = (tuple(zip(sents)) for sents in [(sents_de_tr, sents_en_tr),
  #                                                 (sents_de_te, sents_en_te)])
  ds_tr, ds_te = [(i, j) for (i, j) in zip(sents_en_tr, sents_de_tr)], [(i, j) for (i, j) in zip(sents_en_te, sents_de_te)]
  dl_tr, dl_te = (DataLoader(ds, batch_size=args.batch_size, shuffle=True, 
                           drop_last=True) for ds in (ds_tr, ds_te))

  dim_voc_de, dim_voc_en = len(list(voc_de.keys())), len(list(voc_en.keys()))

  model = Transformer(dim_voc_en, dim_voc_de, args.encoding).to(dev)
  opt = torch.optim.Adam(model.parameters(), lr=args.lr)
  crit = nn.CrossEntropyLoss(ignore_index=0)

  for epoch in range(args.epochs):
    for (src, tgt) in dl_tr:
      tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
      src, tgt_in = src.to(dev), tgt_in.to(dev)
      src, tgt_in, tgt_out = src.long(), tgt_in.long(), tgt_out.long()

      out = model(src, tgt_in).cpu()
      loss = crit(out.transpose(1, 2), tgt_out)

      opt.zero_grad()
      loss.backward()
      opt.step()

      print(f'Loss_tr: {loss.item() :.4f}')
      

if __name__ == '__main__':
    main()





































