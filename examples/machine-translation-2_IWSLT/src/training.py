## Inspired by: https://github.com/gordicaleksa/pytorch-original-transformer/blob/main/training_script.py
from   nltk.translate.bleu_score import corpus_bleu
import numpy as np
import time
import torch
from   tqdm  import tqdm
import sys

from _utils    import Chronometer, Ctr, Loop, Timeit

inf = float('inf')

def bwd(loss, optimizer, do_step):
  loss.backward()
  if do_step: optimizer.step(); optimizer.zero_grad()

def compute_accuracy(
  mode, preds, output_tgt, pad_id, correct_total_split=False,
):
  if mode in 'token-level':
    correct = ((preds == output_tgt) * (output_tgt != pad_id)).sum().item()
    total = ((output_tgt != pad_id)).sum().item()

  elif mode in 'sentence-level':
    correct = (
      (preds == output_tgt) + (output_tgt == pad_id)
    ).prod(axis=1).sum(axis=0).item()
    total = preds.shape[0]

  else: raise Exception(f'Unknown acc-computing mode "{mode}"')

  acc = correct/total if total > 0 else 0.
  return acc if not correct_total_split else (correct, total)

def extend_sentences(
  originals, references, candidates, src_vocab, tgt_vocab, src, tgt, preds
):
  originals .extend([ src_vocab.decode(x, break_at_pad=True) for x in  src  ])
  references.extend([[tgt_vocab.decode(x)]                   for x in  tgt  ])
  candidates.extend([ tgt_vocab.decode(x)                    for x in  preds])

def fwd(batch, model, criterion, label_smoother):
  src, tgt, num_nonpad_tokens, num_total_tokens = batch
  src, tgt = src.to(model.device), tgt.to(model.device)

  input_tgt, output_tgt = tgt[:, :-1], tgt[:, 1:]
  output_tgt_distribution = label_smoother(output_tgt.reshape(-1, 1))

  output = model(src, input_tgt)
  loss = criterion(
    output.reshape(-1, output.shape[-1]),
    output_tgt_distribution,
  )
  return output, loss, src, input_tgt, output_tgt, num_nonpad_tokens, \
         num_total_tokens

def get_monitor(magnitude, unit):
  assert unit in ['batches', 'minutes']
  if magnitude <= 0: magnitude = inf
  return Loop(frequency=magnitude) if unit == 'batches' else \
     Chronometer(period=magnitude)

def log_metrics(split, writer, processed_tokens_ctr, loss, bleu_score=None):
  for tag, num_processed_tokens in [
    ('nonpad', processed_tokens_ctr.non_pad),
    ('total' , processed_tokens_ctr.total  ),
  ]:
    writer.add_scalars(f'Loss/{tag}', {split: loss}, num_processed_tokens)
    if bleu_score is not None:
      writer.add_scalars(f'BLEU-score/{tag}', {split: bleu_score}, 
                                              num_processed_tokens)
    writer.flush()

def print_example(originals, references, candidates, src_vocab, tgt_vocab):
  print()
  print(f'Original : {retouch_sentence(originals[0]    , src_vocab)}')
  print(f'Reference: {retouch_sentence(references[0][0], tgt_vocab)}')
  print(f'Candidate: {retouch_sentence(candidates[0]   , tgt_vocab)}')
  print()

def retouch_sentence(tokens_list, vocab):
  # if tokens_list[-1] == vocab.pad_token: tokens_list = tokens_list[:tokens_list.index(vocab.pad_token)]
  return ' '.join(tokens_list)

@Timeit
def train(
  model, optimizer, criterion, training_data_loader, validation_data_loader,  
  generation_config, label_smoother, src_vocab, tgt_vocab, writer, debug,
  saving_path, monitoring_frequency, monitor_validation_bleu, num_epochs, 
  patience, 
):
  processed_tokens_ctr = Ctr('non_pad', 'total')
  monitor = get_monitor(*monitoring_frequency)
  best_loss = +inf
  current_training_losses = []
  early_stop_ctr = 0

  print(f'Starting training.\n')

  monitor.start()

  for epoch in range(1, num_epochs + 1):
    print(f'Epoch {epoch}\n')

    best_loss, current_training_losses, early_stop_ctr, done = train_epoch(
      model, optimizer, criterion, training_data_loader, 
      validation_data_loader, generation_config, label_smoother, src_vocab, 
      tgt_vocab, writer, debug, monitor_validation_bleu, saving_path, 
      patience, processed_tokens_ctr, monitor, best_loss, early_stop_ctr,
      current_training_losses,
    )
    if done: break

  print(f'Training finished.\n')

@Timeit
def train_epoch(
  model, optimizer, criterion, training_data_loader, validation_data_loader, 
  generation_config, label_smoother, src_vocab, tgt_vocab, writer, debug,
  monitor_validation_bleu, saving_path, patience, processed_tokens_ctr,
  monitor, best_loss, early_stop_ctr, current_training_losses,
):
  model.train()

  training_time = 0.
  last_training_time = 0.

  # wasted_time = 0.
  # t1 = time.time()

  done = False

  for batch_idx, batch in enumerate(training_data_loader):#enumerate(tqdm(training_data_loader)):
    t0 = time.time()
    output, loss, src, input_tgt, output_tgt, num_batch_nonpad_tokens, \
     num_batch_total_tokens = fwd(batch, model, criterion, label_smoother)
    bwd(loss, optimizer)
    training_time += time.time() - t0
    ## fwd + bwd: ~4.5e-2

    # wasted_time += t0 - t1
    # print(training_time, wasted_time)
    # t1 = time.time()

    with torch.no_grad():  # 5e-6
      processed_tokens_ctr.non_pad += num_batch_nonpad_tokens  # 7.2e-07 s
      processed_tokens_ctr.total   += num_batch_total_tokens   # 2.4e-07 s

      current_training_losses.append(loss.item())  # ~1e-5
      monitor.step()  # 7e-7

      if monitor.is_time():  # 6.5e-5? 
        print(f'Training time between validations: '
              f'{training_time - last_training_time} seconds')
        last_training_time = training_time

        print(f'# Processed non-pad tokens: {processed_tokens_ctr.non_pad}')
        print(f'# Processed total   tokens: {processed_tokens_ctr.total  }')

        # training_losses.append(np.mean(current_training_losses))
        mean_training_loss = np.mean(current_training_losses)
        current_training_losses = []

        print(f'Training loss: {mean_training_loss}')
        log_metrics('train', writer, processed_tokens_ctr, mean_training_loss)
        validation_loss = validate(
          model, criterion, validation_data_loader, generation_config, 
          label_smoother, src_vocab, tgt_vocab, writer, 
          monitor_validation_bleu, processed_tokens_ctr,
        )
        if validation_loss < best_loss:
          if not debug: 
            model.save(processed_tokens_ctr, saving_path)#, optimizer)
          best_loss = validation_loss
          early_stop_ctr = 0
        else:
          early_stop_ctr += 1

        if early_stop_ctr == patience: 
          print(f'EARLY STOP: run out of patience.\n')
          done = True
          break

  return best_loss, current_training_losses, early_stop_ctr, done

@Timeit
def validate(
  model, criterion, data_loader, generation_config, label_smoother, src_vocab,
  tgt_vocab, writer, monitor_validation_bleu, processed_tokens_ctr=None,
):
  model.eval()
  total_loss = None
  originals, references, candidates = [], [], []

  with torch.no_grad():
    for batch_idx, batch in enumerate(data_loader):
      # if model.device == 'cuda':
      #   print(torch.cuda.mem_get_info())
      #   print(f"Memory allocated   : {torch.cuda.memory_allocated   (0)/1024**3} GB")
      #   print(f"Memory reserved    : {torch.cuda.memory_reserved    (0)/1024**3} GB")
      #   print(f"Max memory reserved: {torch.cuda.max_memory_reserved(0)/1024**3} GB")

      output, loss, src, _, output_tgt, _, _ = fwd(
        batch, model, criterion, label_smoother
      )  # 1.6e-2 ~ 1.7e-2 s
      total_loss = total_loss + loss if total_loss != None else loss  # 6.5-05

      if monitor_validation_bleu: 
        preds = model.generate(src, output_tgt.shape[1], 
                               **generation_config.__dict__)  # .8 ~ 2.4 s (greedy)  <-- can this be improved?

        extend_sentences(originals, references, candidates, 
                         src_vocab, tgt_vocab, src, output_tgt, preds)  # 3e-3 s

    # post part: 0.07 ~ 0.1 s

    total_loss /= len(data_loader)
    total_loss = total_loss.item()
    print(f'Validation loss: {total_loss}')

    if monitor_validation_bleu: 
      bleu_score = corpus_bleu(references, candidates)
      print(f'Validation bleu score: {bleu_score}')
      print_example(originals, references, candidates, src_vocab, tgt_vocab)

    if processed_tokens_ctr is not None:
      log_metrics('validation', writer, processed_tokens_ctr, total_loss, 
                  *((bleu_score,) if monitor_validation_bleu else ()))

  return total_loss

def test(): pass



