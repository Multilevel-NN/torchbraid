import numpy as np
import torch
from torchtext.data.metrics import bleu_score

def train_epoch(model, train_iter, optimizer, loss_fn, device, create_mask, voc_tgt):
  model.train()
  losses = 0
  candidate_corpus, reference_corpus = [], []
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
      for pred, tgt in zip(preds.T, tgt_out.T):
        pred, tgt = (' '.join([voc_inv.get(i.item(), '<unk>') for i in tensor]) for tensor in (pred, tgt))
        candidate_corpus.append(pred.split())
        reference_corpus.append([tgt.split()])

    # if idx > 10: break

  bleu = bleu_score(candidate_corpus, reference_corpus)
  return losses / len(train_iter), bleu


def evaluate(model, val_iter, loss_fn, device, create_mask, voc_tgt):
  model.eval()
  losses = 0
  candidate_corpus, reference_corpus = [], []
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
      for pred, tgt in zip(preds.T, tgt_out.T):
        pred, tgt = (' '.join([voc_inv.get(i.item(), '<unk>') for i in tensor]) for tensor in (pred, tgt))
        candidate_corpus.append(pred.split())
        reference_corpus.append([tgt.split()])

      # if idx > 10: break

  bleu = bleu_score(candidate_corpus, reference_corpus)
  return losses / len(val_iter), bleu





































