import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbraid
import torchbraid.utils
import sys

# from network_architecture import ParallelNet
from network_architecture import ParallelNet
from mpi4py import MPI
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import get_cosine_schedule_with_warmup

####################################################################################
####################################################################################

def int_or_tuple(value):
  try:
    # Check if the value is an integer
    return int(value)
  except ValueError:
    # Check if the value is a tuple-like object
    try:
      elements = tuple(map(int, value.strip('()').split(',')))
      return elements
    except ValueError:
      raise argparse.ArgumentTypeError("Invalid value. Must be an integer or a tuple-like object.")
        
# Parse command line 
def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """

  # Command line settings
  parser = argparse.ArgumentParser(description='Simple BERT training parser')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=32, metavar='N',
                      help='Number of times steps in the transformer layer (default: 32)')
  parser.add_argument('--Tf',type=float,default=1.0,
                      help='Final time for transformer layer-parallel part')
  parser.add_argument('--serial-file', type=str, default=None,
                      help='Save network to file in serial (not parallel) format')
  parser.add_argument('--seq-len', type=int, default=64,
                      help='Max sequence length')

  # algorithmic settings (batching)
  parser.add_argument('--percent-data', type=float, default=0.05, metavar='N',
                      help='how much of the data to read in and use for training/testing')
  parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                      help='input batch size for training (default: 32)')
  parser.add_argument('--epochs', type=int, default=3, metavar='N',
                      help='number of epochs to train (default: 3)')
  parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                      help='learning rate (default: 1e-4)')
  
  # algorithmic settings (layer-parallel)
  parser.add_argument('--lp-max-levels', type=int_or_tuple, default=(1,2), metavar='N',
                      help='Layer parallel max number of levels (default: (1,2) one forward, 2 backwards)')
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
  parser.add_argument('--model_dimension', type=int, default=128)
  parser.add_argument('--num_heads', type=int, default=1)
  parser.add_argument('--optimizer', type=str, default='SGD')#required=True)
  parser.add_argument('--momentum', type=float, default=.9)

  ##
  # Do some parameter checking
  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()

  if procs % args.dp_size != 0:
    root_print(rank, 1, 1, 'Data parallel size must be an even multiple of the number of processors: %d %d'
               % (procs, args.dp_size) )
    sys.exit(0)
  else:
    procs_lp = int(procs / args.dp_size)

  if args.steps % procs_lp != 0:
    root_print(rank, 1, 1, 'Steps must be an even multiple of the number of layer parallel processors: %d %d'
               % (args.steps, procs_lp) )
    sys.exit(0)

  return args


####################################################################################
####################################################################################

# Parallel printing helper function  
def root_print(rank, s):
  if rank == 0:
    print(s, flush='True')


def main():
  # Begin setting up run-time environment 
  # Initialize MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  procs = comm.Get_size()
  args = parse_args()

  # Use device or CPU
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device, host = torchbraid.utils.getDevice(comm=comm)
  if not use_cuda:
    device = torch.device("cuda" if use_cuda else "cpu")
  print(f'Run info rank: {rank}: Torch version: {torch.__version__} | Device: {device} | Host: {host}')

  # Set seed for reproducibility
  torch.manual_seed(args.seed)

  # Compute number of steps in ResNet per processor
  local_steps = int(args.steps / procs)

  # Get dataloader
  sequence_length = args.seq_len

  # Diagnostic information
  root_print(rank, '-- procs    = {}\n'
                '-- Tf       = {}\n'
                '-- steps    = {}\n'
                '-- max_levels     = {}\n'
                '-- max_bwd_iters  = {}\n'
                '-- max_fwd_iters  = {}\n'
                '-- cfactor        = {}\n'
                '-- fine fcf       = {}\n'
                '-- skip down      = {}\n'.format(procs, 
                    args.Tf, args.steps,
                    args.lp_max_levels,
                    args.lp_bwd_max_iters,
                    args.lp_fwd_max_iters,
                    args.lp_cfactor,
                    args.lp_fine_fcf,
                    not args.lp_use_downcycle)
    )
  
  # Create layer-parallel network
  model = ParallelNet(
                  local_steps=local_steps,
                  max_levels=args.lp_max_levels,
                  bwd_max_iters=args.lp_bwd_max_iters,
                  fwd_max_iters=args.lp_fwd_max_iters,
                  print_level=args.lp_print_level,
                  braid_print_level=args.lp_braid_print_level,
                  cfactor=args.lp_cfactor,
                  fine_fcf=args.lp_fine_fcf,
                  skip_downcycle=not args.lp_use_downcycle,
                  fmg=False, 
                  Tf=args.Tf,
                  relax_only_cg=False,
                  user_mpi_buf=args.lp_user_mpi_buf).to(device)

  model.parallel_nn.fwd_app.setBraidTimers(flag=1)
  model.parallel_nn.fwd_app.setTimerFile(
      f'timing_test_p_{procs}')

  # Load optimizer/scheduler
  betas=(0.9, 0.999)
  warmup_steps=10_000 #50000
  optimizer = optim.AdamW(
    model.parameters(), 
    lr=args.lr, 
    betas=betas, weight_decay=0.01
  )
  optim_schedule = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,  # Number of warmup steps
    num_training_steps=1000000   # Total number of training steps
)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(0, 10001, 1000):
    # Grab checkpoint 
    root_print(rank, f'Epoch: {epoch}')
    checkpoint = torch.load(f'vit-save-parallel-2/model_checkpoint_{rank}_batch_idx={epoch}')
    model.load_state_dict(checkpoint['model_state_dict'])  # Adjust the key if necessary
    root_print(rank, f'Model loaded. {epoch=}')

    # Grab optimizer/scheduler
    optimizer.load_state_dict(
      checkpoint['optimizer_state_dict']
    )
    optim_schedule.load_state_dict(
      checkpoint['scheduler_state_dict']
    )

    # Load the additional data
    images = checkpoint['images']
    labels = checkpoint['labels']
    images, labels = images.to(device), labels.to(device)
    
    model.parallel_nn.fwd_app.setBraidTimers(flag=1)
    model.parallel_nn.fwd_app.setTimerFile(
        f'timing_test_p_{procs}')
    
    # Train 
    torch.cuda.synchronize()
    batch_fwd_pass_start = time.time()
    output = model(images)
    torch.cuda.synchronize()
    batch_fwd_pass_end = time.time()
    loss = model.compose(
      criterion, 
      output.reshape(-1, output.shape[-1]), 
      labels.view(-1)
    )
    loss.backward()

if __name__ == '__main__':
  main()
  print('Finished.')
