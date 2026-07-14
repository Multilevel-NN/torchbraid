# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TorchBraid implements layer-parallel training of neural networks (neural ODEs and GRUs) using multigrid-in-time (MGRIT). The frontend is PyTorch; the backend is XBraid, a C library, wrapped with Cython. MPI ranks partition the network's layers (time steps), so nearly everything meaningful runs under `mpirun`.

## Build

Recommended (pip, editable):

```
pip install -e .
```

- `setup.py` automatically clones XBraid into `src/xbraid` and builds it (`make debug=no braid`) before compiling the Cython extensions. Requires an MPI compiler (`mpicc` via mpi4py or `$PATH`).
- With `-e`, edits to `.py` files take effect immediately, but **changes to `.pyx` files require re-running `pip install -e .`** (or `make` in `src/torchbraid`).

Alternative Make-based build (see MAKEINSTRUCTIONS.md): copy `makefile.inc.example` to `makefile.inc`, edit `XBRAID_ROOT`/`CC`, run `make` at the repo root, and add `src/` to `PYTHONPATH`. Note: `make tests` also requires `makefile.inc` to exist (it defines `MPIRUN` and `PYTHON`).

## Tests

Tests live in `tests/` and use `unittest`. Most must run under MPI (typically 3 ranks) because they exercise the parallel decomposition.

```
make tests            # full suite under MPI (uses makefile.inc's MPIRUN/PYTHON)
make tests-serial     # same tests with -n 1
make tests-direct-gpu # 2-rank check that CUDA-aware MPI works (required for GPU runs)
tox                   # CI-style run in a fresh env; `tox --direct` reuses current env (needs tox-direct)
```

Single test file:

```
mpirun -n 3 python tests/test_layer_parallel.py
```

Single test case (standard unittest selection):

```
mpirun -n 3 python tests/test_layer_parallel.py TestTorchBraid.test_<name>
```

Quick end-to-end smoke test:

```
cd examples/mnist && mpirun -n 2 python mnist_script.py --percent-data 0.01
```

CI (`.github/workflows/test.yml`) runs `tox --direct` on Python 3.9–3.11 with OpenMPI.

## Architecture

Two-layer stack: a compiled Cython/C core and a Python layer on top.

**Cython core** — compiled into the single extension `torchbraid.torchbraid_app`:
- `src/torchbraid/torchbraid_app.pyx` — defines `BraidApp`, the base class that configures and drives an XBraid solve (levels, coarsening factors, iterations, buffer management). It textually `include`s the other two `.pyx` files, so they compile as one unit.
- `src/torchbraid/braid.pyx` — thin wrapper around the XBraid C core (`PyBraid_Core`); `braid_funcs.pxd` declares the C API.
- `src/torchbraid/torchbraid_callbacks.pyx` — the XBraid callbacks (step, clone, sum, norm, buffer pack/unpack, access) that operate on `BraidVector`s, including the MPI buffer packing of torch tensors (GPU-direct when available).
- `src/torchbraid/test_fixtures/` — separately compiled Cython helpers (`test_cbs`, `gpumpi_check`) used by tests.

**Python layer**:
- `braid_vector.py` — `BraidVector`, the state object (tuple of tensors) XBraid evolves through time/layers.
- `odenet_apps.py` — `ForwardODENetApp` / `BackwardODENetApp` (subclasses of `BraidApp`): forward propagation and the adjoint/backward solve for the ODE-net case. `gru_apps.py` is the analogue for GRUs (parallel over the sequence dimension).
- `braid_function.py` — `BraidFunction`, a `torch.autograd.Function` whose `forward` runs the forward MGRIT solve and whose `backward` runs the backward/adjoint solve. This is the bridge that makes the parallel-in-time solver look like an ordinary differentiable op to PyTorch.
- `layer_parallel.py` / `gru_layer_parallel.py` — the user-facing `nn.Module`s (`LayerParallel`, `GRU_Parallel`, `GRU_Serial`). `LayerParallel` takes layer-builder functors plus step counts and distributes the resulting layers across MPI ranks.
- `lp_module.py` — `LPModule`, shared base of both. Its `ExecLP` helper (obtained via `comp_op()`, conventionally named `compose`) is how user code composes serial layers (e.g. opening/closing layers that live only on rank 0) with the parallel block while keeping gradients consistent across ranks — you'll see `compose(op, ...)` throughout examples and drivers.
- `mgopt.py` — `mgopt_solver`: MG/Opt, a multigrid-in-optimization algorithm layered on top of `LayerParallel` (nested iteration, restriction/interpolation of network params and Adam state between coarse/fine models).
- `utils/` — supporting pieces: `data_parallel.py` (splitting a communicator to combine data-parallel with layer-parallel), buffer pack/unpack, context timers, layer-parallel-aware batchnorm/dropout variants.

**Entry points for usage patterns**: `examples/` (mnist scripts and notebooks) shows canonical usage; `drivers/` contains research drivers (mnist, cifar10, rnns, mgopt studies, scaling).

## GPU notes

GPU runs require CUDA-aware MPI (direct GPU communication); verify with `make tests-direct-gpu`. See README's "GPU direct communication" section for diagnosis.
