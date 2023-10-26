from math import log, floor
import time, argparse, sys
from mpi4py import MPI
import torch
import torch.nn as nn
import torchbraid

from downloadModelNet import ModelNet
from network_architecture import ParallelNet
from ModelNet_script import root_print

# command line arguments
parser = argparse.ArgumentParser(description='ModelNet evaluation')
parser.add_argument('--filename', type=str, default=None,
                    help='filename of saved model to be loaded')

# algorithmic settings (batching)
parser.add_argument('--percent-data', type=float, default=0.1, metavar='N',
                    help='how much of the data to read in and use for training/testing')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')

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
                    help='disables running on GPU')
parser.add_argument('--warm-up', action='store_true', default=False,
                    help='Warm up for GPU timings (default: False)')
parser.add_argument('--lp-user-mpi-buf',action='store_true', default=False,
                    help='Layer parallel use user-defined mpi buffers (default: False)')
parser.add_argument('--lp-use-downcycle', action='store_true', default=False,
                    help='Layer parallel use downcycle on or off (default: False)')
parser.add_argument('--lp-sc-levels', type=int, nargs='+', default=None,
                    help='Layer parallel spatial coarsening levels (default: None)')

comm = MPI.COMM_WORLD
rank  = MPI.COMM_WORLD.Get_rank()
procs = MPI.COMM_WORLD.Get_size()
args = parser.parse_args()

# get device
device, host = torchbraid.utils.getDevice(comm=comm)
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
print(f'Run info rank: {rank}: Torch version: {torch.__version__} | Device: {device} | Host: {host}')

# Set seed for reproducibility
torch.manual_seed(1)

# load the model
if args.filename is None:
  channels, steps, state_dict = torch.load("models/nx31_nt128_ml1_scNone.pt", map_location=device)
else:
  channels, steps, state_dict = torch.load(f"models/{args.filename}.pt", map_location=device)

# Compute number of steps per processor
local_steps = int(steps / procs)

# Compute number of parallel-in-time multigrid levels 
def compute_levels(num_steps, min_coarse_size, cfactor):
# Find L such that ( max_L min_coarse_size*cfactor**L <= num_steps)
    levels = floor(log(float(num_steps) / min_coarse_size, cfactor)) + 1

    if levels < 1:
        levels = 1
    return levels

if args.lp_max_levels < 1:
    min_coarse_size = 3
    args.lp_max_levels = compute_levels(steps, min_coarse_size, args.lp_cfactor)

if steps % procs != 0:
    root_print(rank, 1, 1, 'Steps must be an even multiple of the number of layer parallel processors: %d %d'
            % (steps, procs) )

def time_test(rank, model, test_loader, compose, device):
  model.eval()
  test_loss = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()
  
  evaltime = 0.
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      timer = time.time()
      output = model(data)
      evaltime += time.time() - timer

      test_loss += compose(criterion, output, target).item()

      if rank == 0:
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  root_print(rank, f"\nTest set: Average loss: {test_loss:.4f} ")
  root_print(rank, f"Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%) ")
  root_print(rank, f"eval. time: {evaltime:.2E}\n")
#   root_print(rank, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), eval. time: {:.2E}\n'.format(
#     test_loss, correct, len(test_loader.dataset),
#     100. * correct / len(test_loader.dataset)), evaltime)
  return correct, len(test_loader.dataset), test_loss

# load test data
test_data = ModelNet(train=False)
test_indices = torch.randperm(len(test_data))[0:int(args.percent_data * len(test_data))]
test_set = torch.utils.data.Subset(test_data, test_indices)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)


# some logic to determine on which levels to apply the spatial coarsening
if args.lp_sc_levels == -1:
    sc_levels = list(range(args.lp_max_levels))
else:
    sc_levels = args.lp_sc_levels


# Create layer-parallel network
model = ParallelNet(
    channels=channels,
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
    Tf=1.0,
    relax_only_cg=False,
    user_mpi_buf=args.lp_user_mpi_buf,
    sc_levels=sc_levels
).to(device)

neednt_keys = []
# in parallel, each proc only needs part of the saved model
[neednt_keys.append(key) for key in state_dict if key not in model.state_dict()]
[state_dict.pop(key, None) for key in neednt_keys]
model.load_state_dict(state_dict)

time_test(rank, model, test_loader, model.compose, device)
