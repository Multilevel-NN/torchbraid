from __future__ import print_function

import numpy as np

import sys
import statistics as stats
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid
import torchbraid.utils

from torchbraid.mgopt import root_print, compute_levels

from timeit import default_timer as timer

from mpi4py import MPI

from model_utils.positional_encoding import PositionalEncoding
from model_utils.F_enc import F_enc
from model_utils.F_dec import F_dec

__all__ = [
  'OpenLayer', 'CloseLayer', 'StepLayer', 'parse_args', 'ParallelNet',
]

####################################################################################
####################################################################################
# Network architecture is Open + ResNet + Close
# The StepLayer defines the ResNet (ODENet)

class OpenLayer(nn.Module):
  def __init__(
    self, source_vocabulary, target_vocabulary, model_dimension, device, 
    **kwargs,
  ):
    super().__init__()

    ## Constants
    dim_alphabet_source = len(source_vocabulary)
    dim_alphabet_target = len(target_vocabulary)
    self.source_vocabulary = source_vocabulary
    self.target_vocabulary = target_vocabulary
    self.model_dimension = model_dimension
    self.device = device

    ## Embedding & Positional encoding
    self.embedding_encoder = nn.Embedding(
      num_embeddings=dim_alphabet_source, 
      embedding_dim=model_dimension,
      # padding_idx=source_vocabulary.pad_id,
    )
    self.embedding_decoder = nn.Embedding(
      num_embeddings=dim_alphabet_target, 
      embedding_dim=model_dimension,
      # padding_idx=target_vocabulary.pad_id,
    )
    self.positional_encoder = PositionalEncoding(model_dimension)

  def embed_src(self, src):  # src: [b, L]
    ## Padding masks for attention
    # src_padding_mask = torch.where(src.eq(self.pad_token_id), -np.inf, 0)  # src_padding_mask: [b, L]
    src_padding_mask = (src == self.source_vocabulary.pad_id)  # src_padding_mask: [b, L]
    mem_padding_mask = src_padding_mask                        # mem_padding_mask: [b, L]

    src = src.transpose(0, 1)   # (L, b)

    ## Embedding
    x = self.embedding_encoder(src)  # src: [L, b, d]

    ## Scaling
    # x *= np.sqrt(self.model_dimension)

    ## Positional encoding
    x = self.positional_encoder(x)  # x: [L, b, d]

    return x, src_padding_mask, mem_padding_mask

  def embed_tgt(self, tgt):  # y: [b, L']
    ## Causal mask for attention
    Lp = tgt.shape[1]
    tgt_attention_mask = nn.Transformer.generate_square_subsequent_mask(sz=Lp) \
                         .to(self.device)    # (Lp, Lp)

    ## Padding mask for attention
    # tgt_padding_mask = torch.where(tgt.eq(self.pad_token_id), -np.inf, 0)  # tgt_padding_mask: [b, L']
    tgt_padding_mask = (tgt == self.target_vocabulary.pad_id)  # tgt_padding_mask: [b, L']

    tgt = tgt.transpose(0, 1)   # (L', b)

    ## Embedding
    y = self.embedding_decoder(tgt)  # tgt: [L', b, d]

    ## Scaling
    # y *= np.sqrt(self.model_dimension)

    ## Positional encoding
    y = self.positional_encoder(y)  # y: [L', b, d]

    return y, tgt_attention_mask, tgt_padding_mask

  def forward(self, src, tgt):
    x, src_padding_mask, mem_padding_mask   = self.embed_src(src)
    y, tgt_attention_mask, tgt_padding_mask = self.embed_tgt(tgt)

    return (
      x, y, tgt_attention_mask, src_padding_mask, tgt_padding_mask, 
      mem_padding_mask,
    )
# end layer

class CloseLayer(nn.Module):
  def __init__(self, model_dimension, target_vocabulary):
    super().__init__()

    dim_alphabet_target = len(target_vocabulary)

    ## Language Modeling head
    self.LM_head = nn.Linear(model_dimension, dim_alphabet_target, bias=True)

  def forward(self, y):  # y: [L', b, d]
    y = y.transpose(0, 1)  # y: [b, L', d]
    logits = self.LM_head(input=y)  # logits: [b, L', m]

    return logits
# end layer

class StepLayer_enc(nn.Module):
  def __init__(
    self, model_dimension, num_heads, dim_ff, dropout, batch_size, 
    target_length, device,
  ):
    super().__init__()
    torch.manual_seed(0)
    self.F = F_enc(
      d_model=model_dimension, nhead=num_heads, dim_feedforward=dim_ff, 
      dropout=dropout, batch_first=False,
    )

    self.zeros_tensor = torch.zeros(
      size=(target_length, batch_size, model_dimension), device=device,
    )

  def forward(self, x):
    # t0 = time.time()
    x, y = x
    x = self.F(x=x,
      src_mask=None,
      src_key_padding_mask=src_padding_mask,
    )
    # x = torch.stack((x, torch.zeros_like(y)))
    x = torch.stack((x, self.zeros_tensor[:, :y.shape[1]]))
    # t1 = time.time()
    # print(f'Fwd time encoder layer: {t1 - t0} seconds')
    # print(f'x={x}')
    return x

class StepLayer_dec(nn.Module):
  def __init__(
    self, model_dimension, num_heads, dim_ff, dropout, batch_size, 
    source_length, device,
  ):
    super().__init__()
    torch.manual_seed(0)
    self.F = F_dec(model_dimension, num_heads, dim_ff, dropout, False)

    self.zeros_tensor = torch.zeros(
      size=(source_length, batch_size, model_dimension), device=device,
    )

  def forward(self, x):
    # t0 = time.time()
    mem, y = x
    y = self.F(
      x=y, 
      memory=mem,
      tgt_mask=tgt_attention_mask, 
      tgt_key_padding_mask=tgt_padding_mask, 
      mem_key_padding_mask=mem_padding_mask,
    )
    x = torch.stack((self.zeros_tensor[:, :mem.shape[1]], y))
    # t1 = time.time()
    # print(f'Fwd time decoder layer: {t1 - t0} seconds')
    print(f'y={x}')

    return x

####################################################################################
####################################################################################

# Parallel network class
# local_steps: number of ResNet layers per processor
# all other parameter definitions are in argument parser comments below
class ParallelNet(nn.Module):
  def __init__(
    self, model_dimension, num_heads, dim_ff, dropout, batch_size, source_vocabulary, 
    target_vocabulary, source_length, target_length, device, 
    local_steps=8, Tf=1.0, max_levels=1, bwd_max_iters=1, fwd_max_iters=2, 
    print_level=0, braid_print_level=0, cfactor=4, fine_fcf=False, 
    skip_downcycle=True, fmg=False, relax_only_cg=0, user_mpi_buf=False, 
    comm_lp=MPI.COMM_WORLD, comm_dp=None,
  ):
    super(ParallelNet, self).__init__()

    self.comm_dp = comm_dp  # M!

    self.comm_lp = comm_lp
    numprocs = self.comm_lp.Get_size()

    step_layer_enc = lambda: StepLayer_enc(
      model_dimension, num_heads, dim_ff, dropout, batch_size, target_length, 
      device,
    )
    step_layer_dec = lambda: StepLayer_dec(
      model_dimension=model_dimension, num_heads=num_heads, dim_ff=dim_ff, 
      dropout=dropout, batch_size=batch_size, source_length=source_length, 
      device=device,
    )

    # layers = [step_layer_enc, step_layer_dec]
    # num_steps = [local_steps*numprocs, local_steps*numprocs]#[num_encoder_layers, num_decoder_layers]
    # layers = [step_layer_enc]
    # num_steps = [local_steps*numprocs]
    layers = [step_layer_enc for _ in range(local_steps*numprocs)] \
           + [step_layer_dec for _ in range(local_steps*numprocs)]
    num_steps = [1 for _ in range(2*local_steps*numprocs)]

    # num_steps = [local_steps*numprocs * 5//4, local_steps*numprocs * 3//4]
    # layers, num_steps = [], []
    # for i in range(local_steps * numprocs): 
    #   layers.append(step_layer_enc)
    #   num_steps.append(1)
    # for i in range(local_steps * numprocs):
    #   layers.append(step_layer_dec_selfonly)
    #   num_steps.append(1)
    #   layers.append(step_layer_dec_crossmlponly)
    #   num_steps.append(1)

    self.parallel_nn = torchbraid.LayerParallel(
      comm_lp, 
      layers, num_steps, #step_layer, local_steps*numprocs, 
      Tf,
      max_fwd_levels=max_levels,#1,#max_levels, 
      max_bwd_levels=max_levels,
      max_iters=2, user_mpi_buf=user_mpi_buf,
    )
    self.parallel_nn.setBwdMaxIters(bwd_max_iters)
    self.parallel_nn.setFwdMaxIters(fwd_max_iters)
    self.parallel_nn.setPrintLevel(print_level, True)
    self.parallel_nn.setPrintLevel(braid_print_level, False)
    self.parallel_nn.setCFactor(cfactor)
    self.parallel_nn.setSkipDowncycle(skip_downcycle)
    self.parallel_nn.setBwdRelaxOnlyCG(relax_only_cg)
    self.parallel_nn.setFwdRelaxOnlyCG(relax_only_cg)
    if fmg:
      self.parallel_nn.setFMG()

    self.parallel_nn.setNumRelax(1)  # FCF relaxation default on coarse levels
    if not fine_fcf:
      self.parallel_nn.setNumRelax(0, level=0)  # Set F-Relaxation only on the fine grid
    else:
      self.parallel_nn.setNumRelax(1, level=0)  # Set FCF-Relaxation on the fine grid

    # this object ensures that only the LayerParallel code runs on ranks!=0
    compose = self.compose = self.parallel_nn.comp_op()

    # by passing this through 'compose' (mean composition: e.g. OpenLayer o channels)
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.open_nn = compose(
      OpenLayer, 
      source_vocabulary, target_vocabulary, model_dimension, device,
    )
    self.close_nn = compose(
      CloseLayer, 
      model_dimension, target_vocabulary,
    )

  def saveSerialNet(self, name):
    # Model can be reloaded in serial format with: model = torch.load(filename)
    serial_nn = self.parallel_nn.buildSequentialOnRoot()
    if self.comm_lp.Get_rank() == 0:
      s_net = SerialNet(
        -1, -1, -1, serial_nn=serial_nn, open_nn=self.open_nn, 
        close_nn=self.close_nn,
      )
      s_net.eval()
      torch.save(s_net, name)

  def forward(self, src, tgt):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x)
    # this makes sure this is run on only processor 0
    global tgt_attention_mask, src_padding_mask, tgt_padding_mask, \
           mem_padding_mask

    (x, y, tgt_attention_mask, src_padding_mask, tgt_padding_mask, 
    mem_padding_mask,) = self.compose(self.open_nn, src, tgt)
    x = torch.stack((x, y))
    t0_continuous_block_time = time.time()
    x = self.parallel_nn(x)
    t1_continuous_block_time = time.time()
    mem, y = x
    y = self.compose(self.close_nn, y)

    lp_rank = self.comm_lp.Get_rank()
    dp_rank = self.comm_dp.Get_rank() if self.comm_dp is not None else None
    if 0: print(f'''lp_rank={lp_rank}, dp_rank={dp_rank}: {
      t1_continuous_block_time - t0_continuous_block_time :.4f
    }''')

    return y

class SerialNet(nn.Module):
  def __init__(
    self, model_dimension, num_heads, dim_ff, dropout, batch_size, source_vocabulary, 
    target_vocabulary, source_length, target_length, device, 
    local_steps=8, Tf=1.0, serial_nn=None, open_nn=None, close_nn=None,
  ):
    super(SerialNet, self).__init__()

    step_layer_enc = lambda: StepLayer_enc(
      model_dimension, num_heads, dim_ff, dropout, batch_size, target_length, 
      device,
    )
    step_layer_dec_selfonly = lambda: StepLayer_dec_SA(
      model_dimension, num_heads, dropout, batch_size, source_length, device,
    )
    step_layer_dec_crossmlponly = lambda: StepLayer_dec_CA_MLP(
      model_dimension, num_heads, dim_ff, dropout, batch_size, source_length, 
      device,
    )

    numprocs = 1

    layers, num_steps = [], []
    for i in range(local_steps * numprocs): 
      layers.append(step_layer_enc)
      num_steps.append(1)
    for i in range(local_steps * numprocs):
      layers.append(step_layer_dec_selfonly)
      num_steps.append(1)
      layers.append(step_layer_dec_crossmlponly)
      num_steps.append(1)

    if serial_nn is None:
      parallel_nn = torchbraid.LayerParallel(
        MPI.COMM_SELF, layers, num_steps, Tf,
        max_fwd_levels=1, max_bwd_levels=1, max_iters=1
      )
      parallel_nn.setPrintLevel(0, True)
      self.serial_nn = parallel_nn.buildSequentialOnRoot()
    else:
      self.serial_nn = serial_nn

    if open_nn is None:
      self.open_nn = OpenLayer(
        source_vocabulary, target_vocabulary, model_dimension, device,
      )
    else:
      self.open_nn = open_nn

    if close_nn is None:
      self.close_nn = CloseLayer(
        model_dimension, target_vocabulary,
      )
    else:
      self.close_nn = close_nn

  def forward(self, src, tgt):
    global tgt_attention_mask, src_padding_mask, tgt_padding_mask, \
           mem_padding_mask

    (x, y, tgt_attention_mask, src_padding_mask, tgt_padding_mask, 
    mem_padding_mask,) = self.open_nn(src, tgt)
    x = torch.stack((x, y))
    x = self.serial_nn(x)
    mem, y = x
    y = self.close_nn(y)

    return y

####################################################################################
####################################################################################

# Parse command line 
def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """

  # Command line settings
  parser = argparse.ArgumentParser(description='MNIST example argument parser')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=32, metavar='N',
                      help='Number of times steps in the resnet layer (default: 32)')
  parser.add_argument('--Tf',type=float,default=1.0,
                      help='Final time for ResNet layer-parallel part')
  parser.add_argument('--serial-file', type=str, default=None,
                      help='Save network to file in serial (not parallel) format')

  # algorithmic settings (batching)
  parser.add_argument('--percent-data', type=float, default=0.05, metavar='N',
                      help='how much of the data to read in and use for training/testing')
  parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                      help='input batch size for training (default: 50)')
  parser.add_argument('--epochs', type=int, default=3, metavar='N',
                      help='number of epochs to train (default: 3)')
  parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                      help='learning rate (default: 0.01)')

  # algorithmic settings (layer-parallel)
  parser.add_argument('--lp-max-levels', type=int, default=3, metavar='N',
                      help='Layer parallel max number of levels (default: 3)')
  parser.add_argument('--lp-bwd-max-iters', type=int, default=1, metavar='N',
                      help='Layer parallel max backward iterations (default: 1)')
  parser.add_argument('--lp-fwd-max-iters', type=int, default=2, metavar='N',
                      help='Layer parallel max forward iterations (default: 2)')
  parser.add_argument('--lp-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel internal print level (default: 0)')
  parser.add_argument('--lp-braid-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel braid print level (default: 0)')
  parser.add_argument('--lp-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor (default: 4)')
  parser.add_argument('--lp-fine-fcf',action='store_true', default=False,
                      help='Layer parallel fine FCF for forward solve, on or off (default: False)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--warm-up', action='store_true', default=False,
                      help='Warm up for GPU timings (default: False)')
  parser.add_argument('--lp-user-mpi-buf',action='store_true', default=False,
                      help='Layer parallel use user-defined mpi buffers (default: False)')
  parser.add_argument('--lp-use-downcycle', action='store_true', default=False,
                      help='Layer parallel use downcycle on or off (default: False)')

  # data parallelism
  parser.add_argument('--dp-size', type=int, default=1, metavar='N',
                      help='Data parallelism (used if value != 1)')

  ## save model
  parser.add_argument('--output_fn',type=str, default=None,#required=True,
                      help='Output filename (for model saving)')
  parser.add_argument('--models_dir',type=str, default=None,#required=True,
                      help='Models directory (for model saving)')

  ## additional arguments
  parser.add_argument('--model_dimension', type=int, default=256)
  parser.add_argument('--num_heads', type=int, default=8)
  parser.add_argument('--dim_ff', type=int, default=1024)
  parser.add_argument('--dropout', type=float, default=0.)
  # parser.add_argument('--num_encoder_layers', type=int, default=6)
  # parser.add_argument('--num_decoder_layers', type=int, default=6)
  parser.add_argument('--scheduler', type=str, default=None)
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--enforce_serial', action='store_true')

  ##
  # Do some parameter checking
  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()

  if procs % args.dp_size != 0:
    root_print(
      rank, 1, 1, (
        'Data parallel size must be an even multiple of the number of ' 
      + 'processors: %d %d'
      ) % (procs, args.dp_size) 
    )
    sys.exit(0)
  else:
    procs_lp = int(procs / args.dp_size)

  ##
  # Compute number of parallel-in-time multigrid levels 
  def compute_levels(num_steps, min_coarse_size, cfactor):
    from math import log, floor
    # Find L such that ( max_L min_coarse_size*cfactor**L <= num_steps)
    levels = floor(log(float(num_steps) / min_coarse_size, cfactor)) + 1

    if levels < 1:
      levels = 1
    return levels

  if args.lp_max_levels < 1:
    min_coarse_size = 3
    args.lp_max_levels = compute_levels(
      args.steps, min_coarse_size, args.lp_cfactor,
    )

  if args.steps % procs_lp != 0:
    root_print(
      rank, 1, 1, (
        'Steps must be an even multiple of the number of layer parallel '
        'processors: %d %d'
      ) % (args.steps, procs_lp)
    )
    sys.exit(0)

  return args


####################################################################################
####################################################################################




