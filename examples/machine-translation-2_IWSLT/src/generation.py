## Based on https://github.com/huggingface/transformers/blob/c4d4e8bdbd25d9463d41de6398940329c89b7fb6/src/transformers/generation_utils.py
import torch
from   tqdm   import tqdm
import sys

from autoinit import AutoInitializer

inf = float('inf')

@AutoInitializer
class BeamHypotheses:
  def __init__(self, num_beams, length_penalty):
    self.beams = []
    self.worst_score_so_far = inf

  def __len__(self): return len(self.beams)

  def add(self, hypothesis, sum_logprobs):
    score = sum_logprobs / len(hypothesis)**self.length_penalty

    if len(self) < self.num_beams or score > self.worst_score_so_far:
      self.beams.append((score, hypothesis))

      if len(self) > self.num_beams:
        self.beams.sort(reverse=True, key=lambda x: x[0])
        _ = self.beams.pop()
        self.worst_score_so_far = self.beams[-1][0]
      else:
        self.worst_score_so_far = min(self.worst_score_so_far, score)

  def is_done(self, best_sum_logprobs, current_length):
    if len(self) < self.num_beams: return False
    else: 
      current_score = best_sum_logprobs / current_length**self.length_penalty
      return (self.worst_score_so_far >= current_score)

def compute_new_token_logits(model, tgt, memory, encoder_masks):  #/ memory: [b, Ls, d]
                                                                  #/    tgt: [b, Lt]
  decoder_masks = model.get_decoder_masks(tgt, encoder_masks)
  logits = model.compute_logits(tgt, memory, decoder_masks)[:, -1, :]  #/ [b, m]
  return logits  

def generate(
  model, src: torch.LongTensor, max_new_tokens: int, do_sample: bool, 
  length_penalty: float, num_beams: int, num_return_sequences: int, 
  top_k: int, top_p: float,
):  #/ src: [b, Ls]
  model.eval()

  assert max_new_tokens >= 0
  assert num_beams >= 1
  assert num_return_sequences >= 1
  assert top_k >= 1
  assert 0. < top_p <= 1.
  assert do_sample or num_beams >= num_return_sequences

  with torch.no_grad():
    if num_beams == 1:  # no beam search
      tgt = generate_no_beam_search(model, src, max_new_tokens, do_sample, 
                                    num_return_sequences, top_k, top_p)[:, 1:]
    else: 
      #raise NotImplementedError('Beam search not implemented.')
      tgt = generate_beam_search(
        model, src, max_new_tokens, do_sample, length_penalty, num_beams, 
        num_return_sequences, top_k, top_p
      )[:, 1:]
      # tgt = generate_beam_search2(
      #   model, src, max_new_tokens, do_sample, length_penalty, num_beams, 
      #   num_return_sequences, top_k, top_p
      # )[:, 1:]

    padding = tgt.new(tgt.shape[0], max_new_tokens - tgt.shape[1])\
                 .fill_(model.tgt_vocab.pad_id)
    tgt = torch.cat((tgt, padding), axis=-1)
    return tgt

def generate_beam_search(
  model, src: torch.LongTensor, max_new_tokens: int, do_sample: bool, 
  length_penalty: float, num_beams: int, num_return_sequences: int, 
  top_k: int, top_p: float,
):  #/ src: [b, Ls]
  assert num_return_sequences == 1

  batch_size = src.shape[0]
  src = src.repeat_interleave(num_beams, dim=0)           #/    src: [b*B, Ls]; B = num_beams
  memory, tgt, encoder_masks, expanded_batch_size, _ = \
    prepare_decoding(model, src)                          #/ memory: [b*B, Ls, d]
                                                          #/    tgt: [b*B,  1]
  generated_hypotheses = [BeamHypotheses(num_beams, length_penalty)
                          for _ in range(batch_size)]

  beam_scores = torch.zeros((batch_size, num_beams), 
                            dtype=torch.float, device=src.device)  #/ [b, B]

  if not do_sample: beam_scores[:, 1:] = -inf
  beam_scores = beam_scores.view(-1)  #/ [b*B,]
  done = [False for _ in range(batch_size)]

  for step in tqdm(range(max_new_tokens)):
    # if model.device == 'cuda':
    #   print(torch.cuda.mem_get_info())
    #   print(f"Memory allocated   : {torch.cuda.memory_allocated   (0)/1024**3} GB")
    #   print(f"Memory reserved    : {torch.cuda.memory_reserved    (0)/1024**3} GB")
    #   print(f"Max memory reserved: {torch.cuda.max_memory_reserved(0)/1024**3} GB")

    current_length = step + 1

    new_token_logits  = compute_new_token_logits(model, tgt, memory, 
                                                 encoder_masks)       #/ [b*B, m]
    new_token_scores  = new_token_logits.log_softmax(dim=-1)          #/ [b*B, m]
    new_token_scores += beam_scores.unsqueeze(dim=-1)                 #/ [b*B, m]
    tgt_vocab_size = new_token_scores.shape[-1]

    if do_sample:
      # new_token_scores = top_k_filtering(new_token_scores, top_k)               #/ [b*B,   m]
      # new_token_scores = top_p_filtering(new_token_scores, top_p)               #/ [b*B,   m]
      new_token_scores = new_token_scores.view(batch_size, -1)                  #/ [b  , B*m]
      new_token_probs  = new_token_scores.softmax(dim=-1)                       #/ [b  , B*m]
      new_token_ids = new_token_probs.multinomial(num_samples=2*num_beams)      #/ [b  , B*2]
      new_token_scores = new_token_scores.gather(dim=-1, index=new_token_ids)   #/ [b  , B*2]
      new_token_scores, new_token_scores_ids = \
        new_token_scores.sort(descending=True, dim=-1)    #/ new_token_scores    : [b  , B*2]
                                                          #/ new_token_scores_ids: [b  , B*2]
      new_token_ids = new_token_ids.gather(dim=-1, index=new_token_scores_ids)  #/ [b  , B*2]
    else:
      new_token_scores = new_token_scores.view(batch_size, -1)                  #/ [b, B*m]
      new_token_scores, new_token_ids = \
        torch.topk(new_token_scores, 2*num_beams, dim=-1, sorted=True)  #/ new_token_scores: [b, B*2]
                                                                        #/ new_token_ids   : [b, B*2]
    next_batch_beam = []

    for batch_idx in range(batch_size):
      if done[batch_idx]:
        assert len(generated_hypotheses[batch_idx]) >= num_beams
        next_batch_beam.extend(
          [(0, model.tgt_vocab.pad_id, 0)] * num_beams
        )
        continue

      next_sentence_beam = []

      for score_sorted_beam_idx, (extended_token_id, new_token_score) in \
        enumerate(zip(new_token_ids[batch_idx], 
                      new_token_scores[batch_idx])):
        beam_idx = extended_token_id // tgt_vocab_size + batch_idx*num_beams
        token_id = extended_token_id  % tgt_vocab_size

        if token_id.item() == (eos_id := model.tgt_vocab.eos_id):
          if score_sorted_beam_idx >= num_beams: continue
          generated_hypotheses[batch_idx].add(tgt[beam_idx].clone(), 
                                              new_token_score.item())
        else:
          next_sentence_beam.append((new_token_score, token_id, beam_idx))

        if len(next_sentence_beam) == num_beams: break

      if not done[batch_idx]: 
        done[batch_idx] = generated_hypotheses[batch_idx].is_done(
          new_token_scores[batch_idx].max().item(), current_length,
        )

      assert len(next_sentence_beam) == num_beams
      next_batch_beam.extend(next_sentence_beam)
      assert len(next_batch_beam) == num_beams * (batch_idx+1)
      
    if all(done): break

    assert len(next_batch_beam) == num_beams * batch_size
    beam_scores = beam_scores.new([x[0] for x in next_batch_beam])  #/ [b*B,]
    new_tokens  =     tgt    .new([x[1] for x in next_batch_beam])  #/ [b*B,]
    beam_idxs   =     tgt    .new([x[2] for x in next_batch_beam])  #/ [b*B,]

    tgt = tgt[beam_idxs]                                            #/ [b*B, current_length  ]
    tgt = torch.cat((tgt, new_tokens.unsqueeze(dim=-1)), dim=-1)    #/ [b*B, current_length+1]

  for batch_idx in range(batch_size):
    if done[batch_idx]: continue

    for local_beam_idx in range(num_beams):
      beam_idx = num_beams*batch_idx + local_beam_idx
      score  = beam_scores[beam_idx].item()
      tokens = tgt[beam_idx]
      generated_hypotheses[batch_idx].add(tokens, score)

  best_hypotheses = []

  for hypothesis_idx, hypothesis in enumerate(generated_hypotheses):
    sorted_hypothesis = sorted(hypothesis.beams, 
                               key=lambda x: x[0])
    best_hypothesis = sorted_hypothesis.pop()[1]
    best_hypotheses.append(best_hypothesis)

  max_sequence_length = max(map(len, best_hypotheses))
  decoded = tgt.new(batch_size, max_sequence_length)\
               .fill_(model.tgt_vocab.pad_id)
  for hypothesis_idx, hypothesis in enumerate(best_hypotheses):
    decoded[hypothesis_idx, :len(hypothesis)] = hypothesis
    if len(hypothesis) < max_sequence_length: 
      decoded[hypothesis_idx, len(hypothesis)] = eos_id

  return decoded

def generate_beam_search2(  # Marc
  model, src: torch.LongTensor, max_new_tokens: int, do_sample: bool, 
  length_penalty: float, num_beams: int, num_return_sequences: int, 
  top_k: int, top_p: float,
):  #/ src: [b, Ls]
  batch_size = src.shape[0]
  src = src.repeat_interleave(num_beams, dim=0)           #/    src: [b*B, Ls]; B = num_beams
  memory, tgt, encoder_masks, expanded_batch_size, _ = \
    prepare_decoding(model, src)                          #/ memory: [b*B, Ls, d]
                                                          #/    tgt: [b*B,  1]
  generated_hypotheses = [(-inf, None) for _ in range(batch_size)]
  beam_scores = torch.zeros((batch_size*num_beams,), 
                            dtype=torch.float, device=src.device)  #/ [b*B,]

  for step in range(max_new_tokens):
    current_length = step + 1
    is_last_step = (step == max_new_tokens - 1)

    new_token_logits  = compute_new_token_logits(model, tgt, memory, 
                                                 encoder_masks)       #/ [b*B, m]
    new_token_scores  = new_token_logits.log_softmax(dim=-1)          #/ [b*B, m]
    new_token_scores += beam_scores.unsqueeze(dim=-1)                 #/ [b*B, m]
    tgt_vocab_size = new_token_scores.shape[-1]

    if not is_last_step:
      if do_sample:
        new_token_scores = top_k_filtering(new_token_scores, top_k)               #/ [b*B,   m]
        new_token_scores = top_p_filtering(new_token_scores, top_p)               #/ [b*B,   m]
        new_token_scores = new_token_scores.view(batch_size, 
                                                 num_beams*tgt_vocab_size)        #/ [b  , B*m]
        new_token_probs  = new_token_scores.softmax(dim=-1)                       #/ [b  , B*m]
        new_token_ids = new_token_probs.multinomial(num_samples=2*num_beams)      #/ [b  , B*2]
        new_token_scores = new_token_scores.gather(dim=-1, 
                                                   index=new_token_ids)           #/ [b  , B*2]
        new_token_scores, new_token_scores_ids = \
          new_token_scores.sort(descending=True, dim=-1)    #/ new_token_scores    : [b  , B*2]
                                                            #/ new_token_scores_ids: [b  , B*2]
        new_token_ids = new_token_ids.gather(dim=-1, 
                                             index=new_token_scores_ids)          #/ [b  , B*2]
      else:
        new_token_scores = new_token_scores.view(batch_size, -1)                  #/ [b  , B*m]
        new_token_scores, new_token_ids = \
          torch.topk(new_token_scores, 2*num_beams, dim=-1, sorted=True)  
                                                                #/ new_token_scores: [b  , B*2]
                                                                #/ new_token_ids   : [b  , B*2]
    else:
      new_token_scores = new_token_scores[:, 
                                          (eos_id := model.tgt_vocab.eos_id)]     #/ [b*B,]
      new_token_scores = new_token_scores.view(batch_size, num_beams)             #/ [b  , B  ]
      new_token_ids = torch.arange(num_beams)*tgt_vocab_size + eos_id             #/ [B,]
      new_token_ids = new_token_ids.repeat(batch_size, 1)                         #/ [b  , B  ]

    next_batch_beam = []

    for batch_idx in range(batch_size):
      next_sentence_beam = []

      for score_sorted_beam_idx, (extended_token_id, new_token_score) in \
        enumerate(zip(new_token_ids[batch_idx], 
                      new_token_scores[batch_idx])):
        beam_idx = extended_token_id // tgt_vocab_size + batch_idx*num_beams
        token_id = extended_token_id  % tgt_vocab_size

        if token_id.item() == (eos_id := model.tgt_vocab.eos_id):
          hypothesis = torch.cat((tgt[beam_idx].clone(), 
                                  tgt.new(1).fill_(eos_id)), axis=0)  #/ [current_length + 1,]
          sum_logprobs = new_token_score.item()
          score = sum_logprobs / len(hypothesis)**length_penalty
          if score > generated_hypotheses[batch_idx][0]:
            generated_hypotheses[batch_idx] = (score, hypothesis)

        else:
          next_sentence_beam.append((new_token_score, token_id, beam_idx))

        if len(next_sentence_beam) == num_beams: break

      assert len(next_sentence_beam) == (num_beams
             if not is_last_step else 0)
      next_batch_beam.extend(next_sentence_beam)
      assert len(next_batch_beam) == (num_beams * (batch_idx+1)
             if not is_last_step else 0)
      
    assert len(next_batch_beam) == (num_beams * batch_size
           if not is_last_step else 0), \
      f'len(next_batch_beam): {len(next_batch_beam)}, num_beams: {num_beams}, batch_size: {batch_size}, is_last_step {is_last_step}'

    if not is_last_step:
      beam_scores = beam_scores.new([x[0] for x in next_batch_beam])  #/ [b*B,]
      new_tokens  =     tgt    .new([x[1] for x in next_batch_beam])  #/ [b*B,]
      beam_idxs   =     tgt    .new([x[2] for x in next_batch_beam])  #/ [b*B,]

      tgt = tgt[beam_idxs]                                            #/ [b*B, current_length  ]
      tgt = torch.cat((tgt, new_tokens.unsqueeze(dim=-1)), dim=-1)    #/ [b*B, current_length+1]

  max_sequence_length = max(map(lambda hypothesis: len(hypothesis[1]), 
                                generated_hypotheses))
  decoded = tgt.new(batch_size, max_sequence_length)\
               .fill_(model.tgt_vocab.pad_id)

  for batch_idx, (score, hypothesis) in enumerate(generated_hypotheses):
    decoded[batch_idx, :len(hypothesis)] = hypothesis

  return decoded

def generate_no_beam_search(
  model, src: torch.LongTensor, max_new_tokens: int, do_sample: bool, 
  num_return_sequences: int, top_k: int, top_p: float,
):
  memory, tgt, encoder_masks, batch_size, finished_sentences = \
    prepare_decoding(model, src)

  for step in range(max_new_tokens):
    new_token_logits = compute_new_token_logits(model, tgt, memory, 
                                                encoder_masks)
    new_token_scores = new_token_logits

    if do_sample: raise NotImplementedError('Sampling not implemented.')
    else: new_token = new_token_scores.argmax(dim=-1).unsqueeze(dim=-1) # greedy decoding

    new_token = ~finished_sentences * new_token              \
               + finished_sentences * model.tgt_vocab.pad_id
    tgt = torch.cat((tgt, new_token), axis=-1)

    eosed = (new_token == model.tgt_vocab.eos_id)
    finished_sentences += eosed
    if all(finished_sentences): break

  return tgt

def prepare_decoding(model, src):
  encoder_masks = model.open_nn.get_encoder_masks(src)
  memory = model.compute_memory(src, encoder_masks)
  tgt = src.new(batch_size := src.shape[0], 1).fill_(model.tgt_vocab.bos_id)
  finished_sentences = torch.zeros_like(tgt).bool()

  return memory, tgt, encoder_masks, batch_size, finished_sentences

def top_k_filtering(scores: torch.Tensor, top_k: int):  #/ scores: [b*B, m]
  # print('Warning: top_k filtering not implemented. Ingored.')
  if top_k > 0:
    idxs_to_remove = (scores < scores.topk(top_k)[0][:, -1, None])  #/ [b*B, m]
    scores[idxs_to_remove] = -inf                           #/ scores: [b*B, m]

  return scores

def top_p_filtering(scores: torch.Tensor, top_p: int):  #/ scores: [b*B, m]
  # print('Warning: top_k filtering not implemented. Ingored.')
  if top_p < 1.0:
    sorted_scores, sorted_idxs = scores.sort(descending=True)  #/ sorted_scores: [b*B, m]
                                                               #/ sorted_idxs  : [b*B, m]
    cum_probs = torch.cumsum(sorted_scores.softmax(dim=-1), dim=-1)           #/ [b*B, m]
    sorted_idxs_to_remove = (cum_probs > top_p)                               #/ [b*B, m]
    sorted_idxs_to_remove[:, 1:] = sorted_idxs_to_remove[:, :-1].clone()
    sorted_idxs_to_remove[:, 0 ] = False
    idxs_to_remove = sorted_idxs_to_remove.scatter(1, sorted_idxs, 
                                                   sorted_idxs_to_remove)     #/ [b*B, m]
    scores[idxs_to_remove] = -inf

  return scores



