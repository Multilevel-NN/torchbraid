## Example of de-en translation taken from PyTorch website: https://torchtutorialstaging.z5.web.core.windows.net/beginner/translation_transformer.html

import argparse
import time
import torch
import torch.nn as nn

import data_processing as dp
import model as _model
import training as train
import translating as transl

## Debugging
import importlib
r = importlib.reload
mods = [dp, _model, train, transl]
for mod in mods: _ = r(mod)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--n_lays_enc', type=int, required=True)
parser.add_argument('--n_lays_dec', type=int, required=True)
args = parser.parse_args()

## Params
BATCH_SIZE = 64
H = 128
MAX_LEN = 50#100
NUM_EPOCHS = 1000000000

def report_time(f, *args, **kwargs):
  t0 = kwargs.get('t0', time.time())
  output = f(*args, **kwargs)
  print(f'Done ({time.time() - t0 :>5.2f}s)')
  return output

def main():
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print(f'Loading data...', end=' ')
  ds, dl, voc = report_time(dp.main, dev, MAX_LEN, BATCH_SIZE)

  print(f'Building model...', end=' ')
  t0 = time.time()
  model = _model.Transformer(dev, voc['de'], voc['en'], H, args.n_lays_enc, 
                                                           args.n_lays_dec)
  optim = torch.optim.Adam(model.parameters(), lr=args.lr)
  criterion = nn.CrossEntropyLoss(ignore_index=voc['en']['<pad>'])
  _ = report_time(lambda *args, **kwargs: None, t0=t0)

  for epoch in range(1, NUM_EPOCHS+1):
    start_time = time.time()
    train_loss, train_bleu = train.train_epoch(model, optim, criterion, 
                                                   dl['tr'], voc['en'])
    end_time = time.time()

    val_loss, val_bleu = train.eval_epoch(model, criterion, 
                                       dl['va'], voc['en'])

    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, " \
         + f"Train BLEU: {train_bleu :.4f}, Val BLEU: {val_bleu :.4f}, " \
         + f"Epoch time = {(end_time - start_time):.3f}s"))

    ## REMAINING!!!
    # prompt = "Eine Gruppe von Menschen steht vor einem Iglu ."
    # print(transl.translate(model, prompt, voc, dev))

if __name__ == '__main__':
  main()































