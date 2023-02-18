import argparse
import ast
import numpy as np
import torch
import torch.nn as nn

import generate_db as gdb
import model

# fn_data_tr, fn_data_va = (f'../data/{s}.txt' for s in ['tr', 'va'])

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--max_len_src', type=int, required=True)   # 100
parser.add_argument('--max_len_tgt', type=int, required=True)   # 3
parser.add_argument('--h', type=int, required=True)   # 128
parser.add_argument('--num_epochs', type=int, required=True)   # 2
parser.add_argument('--lr', type=float, required=True)   # 1e-4
parser.add_argument('--n_lays_enc', type=float, required=True)   # 3
parser.add_argument('--n_lays_dec', type=float, required=True)   # 3
args = parser.parse_args()

def main():
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  ds_tr_raw, ds_va_raw, voc_src, voc_tgt = gdb.main()
  ds_tr, ds_va = [], []
  for ds_raw, ds in [(ds_tr_raw, ds_tr), (ds_va_raw, ds_va)]:
    for src, tgt in ds_raw:
      src, tgt = [torch.tensor(x).long() for x in (src, tgt)]
      src = torch.cat((src, torch.zeros(args.max_len_src-len(src))), axis=0).long().to(dev)
      tgt = torch.cat((tgt, torch.zeros(args.max_len_tgt-len(tgt))), axis=0).long().to(dev)
      ds.append((src, tgt))

  dl_tr, dl_va = [torch.utils.data.DataLoader(ds, batch_size=args.batch_size, 
                      shuffle=True, drop_last=False) for ds in (ds_tr, ds_va)]

  ## Setup
  m = model.Transformer(voc_src.ctr, voc_tgt.ctr, args.h, voc_src.voc['<pad>'], 
                          voc_tgt.voc['<pad>'], args.max_len_tgt, dev).to(dev)
  opt = torch.optim.Adam(m.parameters(), lr=1e-4)
  crit = nn.CrossEntropyLoss(ignore_index=voc_tgt.voc['<pad>'])

  ## Training
  ctr_grad = 0
  opt.zero_grad()
  for epoch in range(args.num_epochs):
    m.train()
    loss_epoch, corr, tot = [], 0, 0
    for (src, tgt) in dl_tr:
      tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
      ## Fwd
      out = m(src, tgt_inp)
      loss = crit(out.transpose(1, 2), tgt_out)
      # print(loss.item())
      
      ## Bwd
      loss.backward()
      ctr_grad += 1
      if ctr_grad%1 == 0:#%10
        # torch.nn.utils.clip_grad_norm_(m.parameters(), .1)
        opt.step()
        opt.zero_grad()
      
      with torch.no_grad():
        pred = torch.argmax(out, dim=-1)
        corr += ((pred == tgt_out)*(tgt_out != voc_tgt.voc['<pad>'])).sum().item()
        tot += (tgt_out != voc_tgt.voc['<pad>']).sum().item()
      loss_epoch.append(loss.item())
    print(f'Epoch {epoch} - Tr.Loss {np.mean(loss_epoch) : .6f} - Tr.Acc {corr/tot}')

if __name__ == '__main__': main()




































