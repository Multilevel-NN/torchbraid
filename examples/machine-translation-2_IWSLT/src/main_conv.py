## Inspired by: https://github.com/gordicaleksa/pytorch-original-transformer/blob/main/utils/optimizers_and_distributions.py#L5

print('1/5. Importing modules\n')
import numpy as np
import time
import torch
import torch.nn as nn
from   torch.utils.tensorboard import SummaryWriter
# import os; os.system("conda env list")  # debugging spacy environment issue
import sys

from argument_parsing import get_config
from data             import get_data
from model            import get_model
from optimizer        import get_optimizer
from training         import train, validate, test
from _utils           import LabelSmoothingDistribution, FakeObject

def load_best_model_version_and_validate(
  model, criterion, validation_data_loader, label_smoother, src_vocab, 
  tgt_vocab, writer, skip_training, debug, config, loading_path, saving_path,
):
  ## Load best model version and compute its validation Bleu score
  if not skip_training and not debug:
    try:
      model.load(saving_path := config.other.saving_path)
    except: 
      print(f'WARNING: Could not load the last (best) saved version of the '
            f'model during training. '
             'Probably, the model was not validated during training, '
             'so no copy of it has been saved.\n'
             'As a consequence, the model to be validated is the one '
             'previously ' + ('loaded' if loading_path else 'initialized')
             + '.\n')
      model_loaded = False
    else: 
      print('The best model version during training has been successfully '
            'loaded.\n')
      model_loaded = True
  else:
    reason = 'training is skipped' if skip_training else 'debug is True'
    print(f'WARNING: Since {reason}, the model to be validated is '
           'the one previously ' 
           + ('loaded' if loading_path else 'initialized') + '.\n')
    model_loaded = False

  validate(model, criterion, validation_data_loader, config.generation, 
           label_smoother, src_vocab, tgt_vocab, writer, True)

def main():
  num_nodes = 1
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'device: {device}\n')

  ## Config, logging, and seed
  config = get_config()
  # config.tensorboard.filename_suffix = \
  #   '_n' + f'{num_nodes}'.zfill(2) + config.tensorboard.filename_suffix
  # print(f'config: {config}')
  writer = SummaryWriter(**config.tensorboard.__dict__) if not (
    skip_tensorboard := config.tensorboard.__dict__.pop('skip_tensorboard')
  ) else FakeObject()

  torch.manual_seed(config.other.seed)

  ## Data
  print('2/5. Loading data.\n')
  datasets, data_loaders, vocabs = get_data(
    device, debug := config.other.debug, config.generation.num_beams, 
    **config.data.__dict__, 
  )
  src_vocab, tgt_vocab = vocabs['de'], vocabs['en']
  print()
  print(f"# Training   samples: {len(datasets[  'train'   ])}")
  print(f"# Validation samples: {len(datasets['validation'])}")
  print(f"# Test       samples: {len(datasets[   'test'   ])}")
  print()
  print(f"# Training   batches: {len(data_loaders[  'train'   ])}")
  print(f"# Validation batches: {len(data_loaders['validation'])}")
  print(f"# Test       batches: {len(data_loaders[   'test'   ])}")
  print()
  print(f"# Training   batch-size: {data_loaders[  'train'   ].batch_size}")
  print(f"# Validation batch-size: {data_loaders['validation'].batch_size}")
  print()

  num_tokens_per_batch = []
  for batch in data_loaders['train']:
    src, tgt, numnonpad, nontotal = batch
    num_tokens_per_batch.append(src.numel() + tgt.numel())
  print(f'Average #tokens/batch: {np.mean(num_tokens_per_batch)}\n')

  ## Model, optimizer, and criterion
  print('3/5. Building model.\n')
  model = get_model(config.model, src_vocab, tgt_vocab, device)
  optimizer = get_optimizer(model, config.optimizer)
  # criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)
  criterion = nn.KLDivLoss(reduction='batchmean')
  label_smoother = LabelSmoothingDistribution(
    .1, tgt_vocab.pad_id, len(tgt_vocab), device,
  )
  print(f'model: {model}')
  print(f'optimizer: {optimizer}\n')

  if (loading_path := config.other.loading_path):
    model.load(loading_path, optimizer)

  print('4/5. Training model.\n')
  if not (skip_training := config.training.__dict__.pop('skip_training')):
    train(
      model, optimizer, criterion, data_loaders['train'], 
      data_loaders['validation'], config.generation, label_smoother, 
      src_vocab, tgt_vocab, writer, debug, 
      saving_path := config.other.saving_path, **config.training.__dict__, 
    )
  writer.flush()
  writer.close()

  print('5/5. Evaluating trained model.\n')
  load_best_model_version_and_validate(
    model, criterion, data_loaders['validation'], label_smoother, src_vocab, 
    tgt_vocab, writer, skip_training, debug, config, loading_path, 
    saving_path,
  )

  # test()

  print(f'Execution finished.\n')

if __name__ == '__main__': main()



