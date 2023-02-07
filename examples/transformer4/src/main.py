## Example of de-en translation taken from PyTorch website: https://torchtutorialstaging.z5.web.core.windows.net/beginner/translation_transformer.html

# from data_processing import *
# from dataloader import *
# from training import *
# from transformer import *
# from translating import *

import argparse
import time

import data_processing as dp
import dataloader as dl
import transformer as transf
import training as train
import translating as transl

## Debugging
# import importlib
# r = importlib.reload
# r(dp); r(dl); r(transf); r(train); r(transl)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--n_lays_enc', type=int, required=True)
parser.add_argument('--n_lays_dec', type=int, required=True)
args = parser.parse_args()

def main():
  NUM_EPOCHS = 1000000000

  print(f'Loading data...', end=' ')
  t0 = time.time()
  train_data, val_data, test_data, de_vocab, en_vocab, device = dp.main()
  train_iter, valid_iter, test_iter = dl.main(train_data, val_data, test_data, de_vocab, en_vocab)
  print(f'Done ({time.time() - t0 :>5.2f}s)')

  print(f'Building model...', end=' ')
  t0 = time.time()
  (transformer, loss_fn, optimizer, create_mask, 
    generate_square_subsequent_mask) = transf.main(de_vocab, en_vocab, device, args)
  print(f'Done ({time.time() - t0 :>5.2f}s)')

  for epoch in range(1, NUM_EPOCHS+1):
    start_time = time.time()
    train_loss, train_bleu = train.train_epoch(transformer, train_iter,
                     optimizer, loss_fn, device, create_mask, en_vocab)
    end_time = time.time()

    val_loss, val_bleu = train.evaluate(transformer, valid_iter, loss_fn, 
                                           device, create_mask, en_vocab)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, " \
         + f"Train BLEU: {train_bleu :.4f}, Val BLEU: {val_bleu :.4f}, " \
         + f"Epoch time = {(end_time - start_time):.3f}s"))

    print(transl.translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu .", 
                                          de_vocab, en_vocab, device, generate_square_subsequent_mask))

if __name__ == '__main__':
  main()































