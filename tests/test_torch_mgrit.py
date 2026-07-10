"""Parity tests for the pure-PyTorch MGRIT backend (torch_mgrit) against the
XBraid-based LayerParallel path.

Run under MPI, e.g.:  mpirun -n 3 python tests/test_torch_mgrit.py
(global steps below are divisible by 1, 2, 3 and 4 ranks)
"""
import unittest

import torch
import torch.nn as nn
from mpi4py import MPI

import torchbraid
from torchbraid.torch_mgrit import TorchMGRIT

STEPS = 24
CHANNELS = 4
IMG = 8
BATCH = 3
TF = 2.0
CFACTOR = 4
FWD_ITERS = 2
BWD_ITERS = 1

# weights are seeded by GLOBAL layer index so both implementations build the
# identical network: torchbraid constructs each rank's local layers from the
# same RNG stream, and XBraid block-distributes steps+1 time points (uneven
# layer counts), so per-rank streams give distribution-dependent networks
LAYER_COUNTER = [0]


def xbraid_start_layer(steps, nprocs, rank):
  """First layer index owned by `rank` under XBraid's block distribution of
  steps+1 time points (_braid_GetBlockDistInterval)."""
  npoints = steps + 1
  quo, rem = npoints // nprocs, npoints % nprocs
  return rank * quo + min(rank, rem)


class StepLayer(nn.Module):
  def __init__(self):
    super().__init__()
    torch.manual_seed(5000 + LAYER_COUNTER[0])
    LAYER_COUNTER[0] += 1
    self.conv = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)

  def forward(self, x):
    return torch.relu(self.conv(x))


class TestTorchMGRIT(unittest.TestCase):

  def _input(self):
    torch.manual_seed(1234)
    return torch.randn(BATCH, CHANNELS, IMG, IMG)

  def _run_layer_parallel(self, comm, x):
    rank, nprocs = comm.Get_rank(), comm.Get_size()
    LAYER_COUNTER[0] = xbraid_start_layer(STEPS, nprocs, rank)
    model = torchbraid.LayerParallel(comm, StepLayer, STEPS, TF,
                                     max_fwd_levels=2, max_bwd_levels=2,
                                     max_iters=FWD_ITERS)
    model.setFwdMaxIters(FWD_ITERS)
    model.setBwdMaxIters(BWD_ITERS)
    model.setPrintLevel(0)
    model.setCFactor(CFACTOR)
    model.setSkipDowncycle(True)
    model.setNumRelax(1)
    model.setNumRelax(0, level=0)

    compose = model.comp_op()
    y = model(x)
    loss = compose(lambda t: t.square().mean(), y)
    model.zero_grad()
    loss.backward()

    local_sq = sum(float((p.grad**2).sum()) for p in model.parameters()
                   if p.grad is not None)
    total_sq = comm.allreduce(local_sq, op=MPI.SUM)
    loss_val = comm.bcast(loss.item() if rank == 0 else None, root=0)
    return loss_val, total_sq**0.5

  def _run_torch_mgrit(self, comm, x):
    rank, nprocs = comm.Get_rank(), comm.Get_size()
    LAYER_COUNTER[0] = rank * (STEPS // nprocs)
    model = TorchMGRIT(comm, StepLayer, STEPS, TF, cfactor=CFACTOR,
                       fwd_iters=FWD_ITERS, bwd_iters=BWD_ITERS,
                       device=torch.device('cpu'))
    y = model(x)
    loss = y.square().mean()
    model.zero_grad()
    loss.backward()

    local_sq = sum(float((p.grad**2).sum()) for p in model.parameters()
                   if p.grad is not None)
    total_sq = comm.allreduce(local_sq, op=MPI.SUM)
    return loss.item(), total_sq**0.5

  def test_parity_with_layer_parallel(self):
    """Same network, same MGRIT configuration: solutions must agree to
    floating-point roundoff (op-ordering differs slightly inside XBraid)."""
    comm = MPI.COMM_WORLD
    x = self._input()

    lp_loss, lp_gnorm = self._run_layer_parallel(comm, x)
    mg_loss, mg_gnorm = self._run_torch_mgrit(comm, x)

    self.assertGreater(lp_loss, 0.0)
    self.assertGreater(lp_gnorm, 0.0)
    self.assertAlmostEqual(mg_loss / lp_loss, 1.0, places=5)
    self.assertAlmostEqual(mg_gnorm / lp_gnorm, 1.0, places=3)

  def test_serial_exactness(self):
    """On one rank with one level, the backend is sequential integration and
    must match a hand-rolled forward Euler loop exactly."""
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 1:
      self.skipTest('serial test')

    x = self._input()

    LAYER_COUNTER[0] = 0
    model = TorchMGRIT(comm, StepLayer, STEPS, TF, levels=1,
                       fwd_iters=1, bwd_iters=1, device=torch.device('cpu'))
    y = model(x)
    loss = y.square().mean()
    model.zero_grad()
    loss.backward()
    mg_gsq = sum(float((p.grad**2).sum()) for p in model.parameters())

    LAYER_COUNTER[0] = 0
    layers = [StepLayer() for _ in range(STEPS)]
    dt = TF / STEPS
    yr = x
    for l in layers:
      yr = yr + dt * l(yr)
    loss_ref = yr.square().mean()
    loss_ref.backward()
    ref_gsq = sum(float((p.grad**2).sum()) for l in layers
                  for p in l.parameters())

    self.assertEqual(loss.item(), loss_ref.item())
    self.assertEqual(mg_gsq, ref_gsq)


if __name__ == '__main__':
  unittest.main()
