import numpy as np
import torch
from torchtext.data.metrics import bleu_score

def train_epoch(model, optim, criterion, dl, voc_tgt):
  model.train()
  losses = 0
  candidate_corpus, reference_corpus = [], []

  for i, (src, tgt) in enumerate(dl):
    tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:].cpu()

    logits = model(src, tgt_inp).cpu()
    loss = criterion(logits.transpose(1,2), tgt_out)

    optim.zero_grad()
    loss.backward()
    optim.step()

    losses += loss.item()

    with torch.no_grad():
      preds = logits.argmax(dim=-1)
      for pred, tgt in zip(preds, tgt_out):
        pred, tgt = [' '.join([voc_tgt[i.item()] for i in tensor]) for tensor in (pred, tgt)]
        candidate_corpus.append(pred.split())
        reference_corpus.append([tgt.split()])

    # if i > 10: break

  bleu = bleu_score(candidate_corpus, reference_corpus)
  return losses/len(dl), bleu

def eval_epoch(model, criterion, dl, voc_tgt):
  model.eval()
  losses = 0
  candidate_corpus, reference_corpus = [], []

  with torch.no_grad():
    for i, (src, tgt) in (enumerate(dl)):
      tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:].cpu()

      logits = model(src, tgt_inp).cpu()
      loss = criterion(logits.transpose(1,2), tgt_out)

      losses += loss.item()

      preds = logits.argmax(dim=-1)
      for pred, tgt in zip(preds, tgt_out):
        pred, tgt = (' '.join([voc_tgt[i.item()] for i in tensor]) for tensor in (pred, tgt))
        candidate_corpus.append(pred.split())
        reference_corpus.append([tgt.split()])

      # if i > 10: break

  bleu = bleu_score(candidate_corpus, reference_corpus)
  return losses/len(dl), bleu





































