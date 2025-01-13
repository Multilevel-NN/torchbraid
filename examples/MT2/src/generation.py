## Based on https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py
import time
import torch
# from   tqdm   import tqdm
import sys

from autoinit import AutoInitializer

def generate(
  model, 
  src: torch.LongTensor, 
  max_new_tokens: int, 
  debug: bool=False,
):  #/ src: [b, Ls]
  model.eval()

  assert max_new_tokens >= 0

  pad_id, bos_id, eos_id = (vocab := model.target_vocabulary).pad_id, \
                                          vocab.bos_id, vocab.eos_id

  with torch.no_grad():
    tgt = generate_no_beam_search(
      model, src, max_new_tokens, pad_id, bos_id, eos_id, debug
    )[:, 1:]
    padding = tgt.new(tgt.shape[0], max_new_tokens - tgt.shape[1]) \
                 .fill_(pad_id)
    tgt = torch.cat((tgt, padding), axis=-1)
    return tgt

def generate_no_beam_search(
    model, 
    src: torch.LongTensor, 
    max_new_tokens: int, 
    pad_id, 
    bos_id, 
    eos_id,
    debug,
):
  batch_size, L = src.shape
  assert max_new_tokens <= L
  tgt = src.new(batch_size, L+1).fill_(pad_id)  #/ tgt: [b, L+1]
  tgt[:, 0] = bos_id
  finished_sentences = tgt.new(batch_size, 1).fill_(0).bool()  #/ [b, 1]

  for step in range(max_new_tokens):
    # t0 = time.time()
    input_tgt = tgt[:, :-1]
    new_token_logits = model(src, input_tgt)  #/ [b, L, m]
    
    # if model.comm_lp is not None:
    #   tcomm0 = time.time()
    #   new_token_logits = model.comm_lp.bcast(new_token_logits, root=0)
    #   tcomm1 = time.time()
    rank = comm_lp.Get_rank() if (comm_lp := model.comm_lp) is not None else 0
    
    if rank == 0:
      new_token_logits = new_token_logits[:, step, :]  #/ [b, m]
      new_token_scores = new_token_logits
  
      new_token = new_token_scores.argmax(dim=-1).unsqueeze(dim=-1) # greedy decoding  #/ [b, 1]
  
      new_token = ~finished_sentences * new_token              \
                 + finished_sentences * pad_id
  
      # print(f'new_token_logits: {new_token_logits}')
      # print(f'new_token: {new_token}')
      # print(f'finished_sentences: {finished_sentences}')
      # print(tgt[0])
      # sys.exit()
  
      # tgt = torch.cat((tgt, new_token), axis=-1)
      tgt[:, step+1] = new_token.ravel()
  
      eosed = (new_token == eos_id)
      finished_sentences += eosed

    if comm_lp is not None: 
      # tcomm0 = time.time()
      finished_sentences = comm_lp.bcast(finished_sentences, root=0)
      # tcomm1 = time.time()

    # t1 = time.time()
    # print(f'Rank: {rank}. Generation step(comm) time: {t1 - t0} ({tcomm1 - tcomm0}) seconds')

    if all(finished_sentences) or debug: break

  return tgt




