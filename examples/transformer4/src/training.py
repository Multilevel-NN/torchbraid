import numpy as np
import torch

from bleu import Bleu

def train_epoch(model, train_iter, optimizer, loss_fn, device, create_mask, voc_tgt):
  model.train()
  losses = 0
  # corr, tot = 0, 0    # Opt 1
  bleus = []    # Opt 2
  voc_inv = {v: k for (k, v) in voc_tgt.items()}
  for idx, (src, tgt) in enumerate(train_iter):
    src = src.to(device).T
    tgt = tgt.to(device).T

    tgt_input = tgt[:-1, :]

    # print(src.shape, tgt.shape, tgt_input.shape)

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    logits = model(src, tgt_input, src_mask, tgt_mask,
                              src_padding_mask, tgt_padding_mask, src_padding_mask)

    optimizer.zero_grad()

    tgt_out = tgt[1:,:]
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

    loss.backward()
    optimizer.step()

    losses += loss.item()

    with torch.no_grad():
      preds = logits.argmax(dim=-1)
      ## Option 1: precision (p_1)
      # corr += ((preds == tgt_out)*(tgt_out != 0)).sum()
      # tot += (tgt_out != 0).sum()
      ## Option 2: bleu
      for pred, tgt in zip(preds.T, tgt_out.T):
        # tgt, pred = tgt.tolist(), pred.tolist()
        pred, tgt = (' '.join([voc_inv.get(i.item(), '<unk>') for i in tensor]) for tensor in (pred, tgt))
        # print(f'pred: {pred}')
        # print(f'tgt: {tgt}')
        bleu = Bleu(pred, tgt)
        if bleu > .8: f = open('goodbleus.txt', 'a'); f.write(';'.join([str(tgt), str(pred)])); f.close()
        bleus.append(bleu)

    ## 
    # with open('log.txt', 'a') as f: f.write(f'loss_tr {loss.item() :.2f}\n')

    # if idx > 10: break

  # acc = (corr/tot).item()   # Opt 1
  bleu = np.mean(bleus)   # Opt 2
  return losses / len(train_iter), bleu#acc


def evaluate(model, val_iter, loss_fn, device, create_mask, voc_tgt):
  model.eval()
  losses = 0
  # corr, tot = 0, 0    # Opt 1
  bleus = []    # Opt 2
  voc_inv = {v: k for (k, v) in voc_tgt.items()}
  with torch.no_grad():
    for idx, (src, tgt) in (enumerate(val_iter)):
      src = src.to(device).T
      tgt = tgt.to(device).T

      tgt_input = tgt[:-1, :]

      # print(src.shape, tgt.shape, tgt_input.shape)

      src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

      logits = model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
      tgt_out = tgt[1:,:]
      loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
      losses += loss.item()

      preds = logits.argmax(dim=-1)
      ## Option 1: precision (p_1)
      # corr += ((preds == tgt_out)*(tgt_out != 0)).sum()
      # tot += (tgt_out != 0).sum()
      ## Option 2: bleu
      for pred, tgt in zip(preds.T, tgt_out.T):
        pred, tgt = (' '.join([voc_inv.get(i.item(), '<unk>') for i in tensor]) for tensor in (pred, tgt))
        #print(f'pred: {pred}')
        #print(f'tgt: {tgt}')
        bleu = Bleu(pred, tgt)
        bleus.append(bleu)

      # if idx > 10: break

  # acc = (corr/tot).item()   # Opt 1
  bleu = np.mean(bleus)   # Opt 2
  return losses / len(val_iter), bleu#acc





