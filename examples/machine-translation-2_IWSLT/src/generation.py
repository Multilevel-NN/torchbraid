## Based on https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py
import torch
from   tqdm   import tqdm
import sys

from autoinit import AutoInitializer

def compute_new_token_logits(model, tgt, memory, encoder_masks):  #/ memory: [b, Ls, d]
                                                                  #/    tgt: [b, Lt]
  decoder_masks = model.get_decoder_masks(tgt, encoder_masks)
  logits = model.compute_logits(tgt, memory, decoder_masks)[:, -1, :]  #/ [b, m]
  return logits  

def generate(model, src: torch.LongTensor, max_new_tokens: int):  #/ src: [b, Ls]
  model.eval()

  assert max_new_tokens >= 0

  pad_id, bos_id, eos_id = (vocab := model.target_vocabulary).pad_id, \
                                          vocab.bos_id, vocab.eos_id

  with torch.no_grad():
    tgt = generate_no_beam_search(model, src, max_new_tokens, pad_id, bos_id,
                                                               eos_id)[:, 1:]
    padding = tgt.new(tgt.shape[0], max_new_tokens - tgt.shape[1]) \
                 .fill_(pad_id)
    tgt = torch.cat((tgt, padding), axis=-1)
    return tgt

def generate_no_beam_search(
    model, src: torch.LongTensor, max_new_tokens: int, pad_id, bos_id, eos_id,
):
  batch_size, L = src.shape
  assert max_new_tokens <= L
  tgt = src.new(batch_size, L+1).fill_(pad_id)  #/ tgt: [b, L+1]
  finished_sentences = tgt.new(batch_size, 1).fill_(0).bool()  #/ [b, 1]

  for step in range(max_new_tokens):
    input_tgt = tgt[:, :-1]
    new_token_logits = model(src, input_tgt)[:, step, :]  #/ [b, m]
    new_token_scores = new_token_logits

    new_token = new_token_scores.argmax(dim=-1).unsqueeze(dim=-1) # greedy decoding

    new_token = ~finished_sentences * new_token              \
               + finished_sentences * pad_id
    # tgt = torch.cat((tgt, new_token), axis=-1)
    tgt[:, step+1] = new_token.ravel()

    eosed = (new_token == eos_id)
    finished_sentences += eosed
    if all(finished_sentences): break

  return tgt




