import numpy as np
from torchmetrics.functional import bleu_score

def p(n, tgt, pred):
  corr, tot = 0, 0
  for i in range(len(pred)-n+1):
    corr += (tgt[i:i+n] == pred[i:i+n]) if i < len(tgt) else 0 
    tot += 1
  return corr/tot if tot > 0 else 0

## Option 1
# def GAP(tgt, pred, N=4):   # Geometric Average Precision (N)
#   pn = np.array([p(n, tgt, pred) for n in range(1, N+1)])
#   return np.prod(pn**(1/N))

# def BrevPen(tgt, pred):   # Brevity Penalty
#   r, c = len(tgt), len(pred)
#   return 1 if c>r else np.exp(1-r/c)

# def Bleu(tgt, pred, N=4): return BrevPen(tgt, pred) * GAP(tgt, pred, N)

## Option 2
# def logBleu(tgt, pred, N=4):
#   r, c = len(tgt), len(pred)  
#   pn = np.array([p(n, tgt, pred) for n in range(1, N+1)])
#   return min(1-r/c, 0) + (np.log(pn)/N).sum()

# def Bleu(tgt, pred, N=4): return np.exp(logBleu(tgt, pred, N=4))

## Option 3
# def Bleu(tgt, pred, **irrel):
#   references = [[tgt]]
#   candidates = [pred]
#   from nltk.translate.bleu_score import corpus_bleu
#   score = corpus_bleu(references, candidates)
#   return score

## Option 4
def Bleu(pred, tgt, **irrel):
    score = bleu_score([pred], [tgt])
    return score

def main():
  tgt  = "The guard arrived late because it was raining"
  pred = "The guard arrived late because of the rain"
  print(Bleu(pred, tgt, N=4))

if __name__ == '__main__': main()


































