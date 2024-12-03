import sys
sys.path.append('/users/msalvado/anaconda3/envs/original-transf_MT_20240717/lib/python3.11/site-packages')

from nltk.translate.bleu_score    import corpus_bleu      as nltk_corpus_bleu
from sacrebleu                    import corpus_bleu      as   sb_corpus_bleu
from torchmetrics.functional.text import sacre_bleu_score as   sb_corpus_bleu2

def corpus_bleu(candidates, references):
  assert type(candidates) == list
  assert type(references) == list
  assert all(type(_references) == list for _references in references)

  if type(candidates[0]) == list:
    assert all(type(candidate) == list for candidate in candidates)
    assert all(all(type(token) == str  for token     in candidate ) 
               for candidate   in  candidates)
    assert all(all(type(reference) == list for reference in _references) 
               for _references in  references)
    assert all(all(all(type(token) == str  for   token   in  reference )
               for  reference  in _references)
               for _references in  references)

    bleu_score = nltk_corpus_bleu(hypotheses=candidates, 
                          list_of_references=references)
  elif type(candidates[0]) == str:
    assert all(type(candidate) == str for candidate in candidates)
    assert all(all(type(reference) == str for reference in _references)
               for _references in references)

    # bleu_score = sb_corpus_bleu(hypotheses=candidates,
    #                             references=references).score / 100
    bleu_score = sb_corpus_bleu2(preds=candidates,
                                target=references).item()
  else: raise Exception()

  return bleu_score





