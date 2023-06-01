import numpy as np
import torch
import torch.nn as nn
from torchtext.data.metrics import bleu_score

def train_epoch(model, optim, criterion, dl, voc_tgt):
  model.train()
  losses = 0
  candidate_corpus, reference_corpus = [], []
  optim.zero_grad(); ctr_gradients_accum = 0

  for i, (src, tgt) in enumerate(dl):
    tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:].cpu()

    logits = model(src, tgt_inp).cpu()
    loss = criterion(logits.transpose(1,2), tgt_out)

    loss.backward()
    ctr_gradients_accum += 1
    if ctr_gradients_accum%10 == 0: 
      nn.utils.clip_grad_norm_(model.parameters(), .1)
      optim.step()
      optim.zero_grad()

    losses += loss.item()

    with torch.no_grad():
      preds = logits.argmax(dim=-1)
      for pred, tgt in zip(preds, tgt_out):
        pred, tgt = [' '.join([voc_tgt[i.item()] for i in tensor]) for tensor in (pred, tgt)]
        pred, tgt = (pred[:pred.index('<pad>')] if '<pad>' in pred else pred, 
                      tgt[: tgt.index('<pad>')] if '<pad>' in tgt  else tgt )
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
        pred, tgt = (pred[:pred.index('<pad>')] if '<pad>' in pred else pred, 
                      tgt[: tgt.index('<pad>')] if '<pad>' in tgt  else tgt )
        candidate_corpus.append(pred.split())
        reference_corpus.append([tgt.split()])

      # if i > 10: break

  bleu = bleu_score(candidate_corpus, reference_corpus)
  return losses/len(dl), bleu


def print_example(model, dl, voc, forced_learning=False):
  voc_src, voc_tgt = voc['de'], voc['en']
  model.eval()
  src, tgt = next(iter(dl))
  src, tgt_inp, tgt_out = src[:1], tgt[:1, :-1], tgt[:1, 1:].cpu()

  if forced_learning: 
    output = model(src, tgt_inp).cpu()
    preds = output.argmax(dim=-1)
  else: 
    preds = []
    tgt_inp = torch.full_like(tgt_inp, voc_src['<sos>'])  # (b, L)
    max_len = tgt_out.shape[1]
    for step in range(max_len):
      output = model(src, tgt_inp)                        # (b, L, d)
      pred_sent = output.argmax(-1)                       # (b, L)
      pred_tok = pred_sent[:, step]                       # (b)
      preds.append(pred_tok[0].item())
      tgt_inp[:, step] = pred_tok                                    
    preds = torch.LongTensor(preds, device='cpu').unsqueeze(0)

  text = dict(zip('src pred tgt'.split(), 
                  'Original;Predicted;Correct'.split(';')))
  print(tgt_out[0][-1].item())
  orig, corr, pred = [' '.join([_voc[i.item()] for i in tensor]) \
                       for (tensor, _voc) in [(src[0], voc_src), \
                    (tgt_out[0], voc_tgt), (preds[0], voc_tgt)]]
  orig, corr, pred = (orig[:orig.index('<pad>')] if '<pad>' in orig else orig, 
                      corr[:corr.index('<eos>')+len('<eos>')] if '<eos>' in corr else corr,
                      pred[:pred.index('<eos>')+len('<eos>')] if '<eos>' in pred else pred)

  print(f'{text["src"]}: {orig}')
  print(f'{text["tgt"]}: {corr}')
  print(f'{text["pred"]}: {pred}')

  bleu = bleu_score([pred.split()], [[corr.split()]])
  print(f'Bleu: {bleu}')
  print('-'*50)

































