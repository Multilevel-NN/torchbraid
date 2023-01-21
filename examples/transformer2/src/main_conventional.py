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
parser.add_argument('--steps', type=int, required=True)
args = parser.parse_args()


class Transformer(nn.Module):
  def __init__(self, dim_src, dim_tgt, encoding, nlayers_enc, nlayers_dec):
    super(Transformer, self).__init__()
    self.encoding = encoding
    self.nlayers_enc = nlayers_enc
    self.nlayers_dec = nlayers_dec

    self.emb_src = nn.Embedding(dim_src, 128)
    self.emb_tgt = nn.Embedding(dim_tgt, 128)
    self.dout_src = nn.Dropout(p=.1)
    self.dout_tgt = nn.Dropout(p=.1)
    self.posenc = PositionalEncoding(128) if encoding == 'Torch'\
      else PE_Alternative(128) if encoding == 'Alternative'\
      else Exception('encoding unknown')

    # self.transformer = nn.Transformer(
    #   d_model=128, nhead=8, num_encoder_layers=3,
    #   num_decoder_layers=3, dim_feedforward=1024,
    #   dropout=.3, batch_first=True)

    ## Encoder    
    self.enc_att = nn.MultiheadAttention(
      embed_dim=128, 
      num_heads=8, 
      dropout=.3, 
      batch_first=True
    )

    self.enc_fc1 = nn.Linear(128, 1024)
    self.enc_dout = nn.Dropout(.1)
    self.enc_fc2 = nn.Linear(1024, 128)

    self.enc_ln1 = nn.LayerNorm(128, eps=1e-5)
    self.enc_ln2 = nn.LayerNorm(128, eps=1e-5)
    self.enc_dout1 = nn.Dropout(.1)
    self.enc_dout2 = nn.Dropout(.1)

    ## Decoder
    self.dec_att_tgt = nn.MultiheadAttention(
      embed_dim=128, 
      num_heads=8, 
      dropout=.3, 
      batch_first=True
    )
    self.dec_att_mha = nn.MultiheadAttention(
      embed_dim=128, 
      num_heads=8, 
      dropout=.3, 
      batch_first=True
    )
    self.dec_fc1 = nn.Linear(128, 1024)
    self.dec_dout = nn.Dropout(.1)
    self.dec_fc2 = nn.Linear(1024, 128)

    self.dec_ln1 = nn.LayerNorm(128, eps=1e-5)
    self.dec_ln2 = nn.LayerNorm(128, eps=1e-5)
    self.dec_ln3 = nn.LayerNorm(128, eps=1e-5)
    self.dec_dout1 = nn.Dropout(.1)
    self.dec_dout2 = nn.Dropout(.1)
    self.dec_dout3 = nn.Dropout(.1)

    ## Classifier
    self.fc = nn.Linear(128, dim_tgt)

  def forward(self, src, tgt):
    msk_tgt = nn.Transformer.generate_square_subsequent_mask(
      sz=tgt.shape[1]).to(src.device)
    msk_pad_src = (src == 0)
    msk_pad_mem = msk_pad_src
    msk_pad_tgt = (tgt == 0)

    src = self.emb_src(src)
    tgt = self.emb_tgt(tgt)

    # out = self.transformer(src=src, tgt=tgt, tgt_mask=msk_tgt, 
    #   src_key_padding_mask=msk_pad_src,
    #   tgt_key_padding_mask=msk_pad_tgt, 
    #   memory_key_padding_mask=msk_pad_mem,)

    ## Encoder
    x = src
    for _ in range(self.nlayers_enc):
      x = self.encode(x, msk_pad_src)

    mem = x

    ## Decoder
    x = tgt

    for _ in range(self.nlayers_dec):
      x = self.decode(x, mem, msk_tgt, msk_pad_mem, msk_pad_tgt)

    ## Classifier
    out = self.fc(x)

    return out

  def encode(self, src, msk_pad_src):
    x = src
    x_sa, _ = self.enc_att(x, x, x, key_padding_mask=msk_pad_src)
    x_sa = self.enc_dout1(x_sa)
    x = self.enc_ln1(x + x_sa)

    x_ff = self.enc_fc1(x).relu()
    x_ff = self.enc_dout(x_ff)
    x_ff = self.enc_fc2(x_ff)
    x_ff = self.enc_dout2(x_ff)
    x = self.enc_ln2(x + x_ff)

    return x

  def decode(self, tgt, mem, msk_tgt, msk_pad_mem, msk_pad_tgt):
    x = tgt
    x_sa, _ = self.dec_att_tgt(x, x, x, attn_mask=msk_tgt, 
                              key_padding_mask=msk_pad_tgt)
    x_sa = self.dec_dout1(x_sa)
    x = self.dec_ln1(x + x_sa)

    x_mha, _ = self.dec_att_mha(x, mem, mem, key_padding_mask=msk_pad_mem)
    x_mha = self.dec_dout2(x_mha)
    x = self.dec_ln2(x + x_mha)

    x_ff = self.dec_fc1(x).relu()
    x_ff = self.dec_dout(x_ff)
    x_ff = self.dec_fc2(x_ff)
    x_ff = self.dec_dout3(x_ff)
    x = self.dec_ln3(x + x_ff)

    return x


def main():
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  torch.manual_seed(args.seed)

  vocs, sents = preproc.main(small=False)#True)
  voc_de, voc_en = vocs
  sents_de_tr, sents_en_tr, sents_de_te, sents_en_te = sents
  # ds_tr, ds_te = (tuple(zip(sents)) for sents in [(sents_de_tr, sents_en_tr),
  #                                                 (sents_de_te, sents_en_te)])
  ds_tr, ds_te = [(i, j) for (i, j) in zip(sents_en_tr, sents_de_tr)], [(i, j) for (i, j) in zip(sents_en_te, sents_de_te)]
  dl_tr, dl_te = (DataLoader(ds, batch_size=args.batch_size, shuffle=True, 
                           drop_last=True) for ds in (ds_tr, ds_te))

  dim_voc_de, dim_voc_en = len(list(voc_de.keys())), len(list(voc_en.keys()))

  model = Transformer(dim_voc_en, dim_voc_de, args.encoding, 
    nlayers_enc=args.steps, nlayers_dec=args.steps).to(dev)
  opt = torch.optim.Adam(model.parameters(), lr=args.lr)
  crit = nn.CrossEntropyLoss(ignore_index=0)

  for epoch in range(args.epochs):
    batch_epochs, batch_ctr = 100, 0
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

      batch_ctr += 1
      if batch_ctr >= batch_epochs:
        break

    model.eval()
    with torch.no_grad():
      corr, tot = 0, 0
      for (src, tgt) in dl_te:
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        src, tgt_in = src.to(dev), tgt_in.to(dev)
        src, tgt_in, tgt_out = src.long(), tgt_in.long(), tgt_out.long()

        out = model(src, tgt_in).cpu()
        # loss = crit(out.transpose(1, 2), tgt_out)
        pred = out.argmax(axis=-1)
        corr += ((pred == tgt_out)*(tgt_out != 0)).sum()
        tot += (tgt_out != 0).sum()

        batch_ctr += 1
        if batch_ctr == 10000: break

      acc_te = corr/tot
      print(f'Acc_te: {acc_te.item() : .4f}')
    model.train()
      

if __name__ == '__main__':
    main()





































