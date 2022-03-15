# Training GRU's

This directory contains an example for training GRU networks on the UCI HAR dataset.
For more details see [Parallel Training of GRU Networks with a Multi-Grid Solver for Long Sequences](https://arxiv.org/abs/2203.04738)
by G. E. Moon and E. C. Cyr.

## Instructions

Download the UCI HAR dataset:

```
  > sh ./download.sh
  > unzip UCI\ HAR\ Dataset.zip
  > rm -fr __MACOSX
  > rm -f UCI\ HAR\ Dataset.zip 
```
The last two lines are optional cleanup

## Running 

Here we train with parallel-in-time and show the output that was observed on one platform. Here is the command:

```
mpirun -n 4 python ./train.py --lp-levels 3 --lp-cfactor 4 --epochs=4 --lp-iters=1 --lp-fwd-finerelax=0 --lp-fwd-relax=1 --seed 3649905550453979348 --lp-fwd-iters 2
```

The output is below, notice that the repo SHA1 is included, the full set of input arguments, and the runtime for each function. 

In detail, a line indicating a training step with some details on the performance of the parallel-in-time cycle.
```
Train Epoch:  <EPOCHNUM> [<CURSAMP>/<SAMPS>]	Loss: <LOSS>	Time Per Batch <BATCHTIME> - F <FWDITER>/<FWDRES>, B <BWDITER>/<BWDRES>, fp=<FWDTIME>, cm=<LOSSTIME>, bp=<BWDTIME>
```
Here is the key:
* EPOCHNUM: Epoch number
* CURSAMP: Current sample number
* SAMPS: Number of Samples
* LOSS: Loss mesured on observed batch
* BATCHTIME: Time to do forward and backward propgation on a batch
* FWDITER: Number of forward iterations taken
* FWDRES: Braid residual at the end of the forward iterations
* BWDITER: Number of backward iterations taken
* BWDRES: Braid residual at the end of the backward iterations
* FWDTIME: Time to run the forward evaluation
* LOSSTIME: Time to compute the loss 
* BWDTIME: Time to run the backward evaluation

## Full output

```
TORCHBRAID REV: 55ad2c4516d24586d765cb2bcf16579ca3d3086e
DRIVER REV:     55ad2c4516d24586d765cb2bcf16579ca3d3086e

MPI: procs = 4
INPUT: Namespace(seed=3649905550453979348, log_interval=10, percent_data=1.0, ensemble_size=1, sequence_length=128, tf=128.0, input_size=9, hidden_size=100, num_layers=2, num_classes=6, implicit_serial=False, batch_size=100, batch_bug=False, epochs=4, lr=0.001, use_sgd=False, force_lp=False, lp_levels=3, lp_iters=1, lp_fwd_iters=2, lp_print=0, lp_braid_print=0, lp_cfactor=4, lp_use_downcycle=False, lp_fwd_finerelax=0, lp_fwd_relax=1, lp_fwd_tol=1e-16, lp_bwd_tol=1e-16)

USING SEED: 3649905550453979348

Loading UCI HAR Dataset:

Using ParallelNet:
-- max_levels = 3
-- max_iters  = 1
-- fwd_iters  = 2
-- cfactor    = 4
-- fwd0 relax = 0
-- fwd relax  = 1
-- fwd tol    = 1e-16
-- bwd tol    = 1e-16
-- skip down  = True

Train Epoch:  1 [ 100/7352]	Loss: 1.766864	Time Per Batch 0.149827 - F 2/5.29e+01, B 1/-1.00e+00
Train Epoch:  1 [1100/7352]	Loss: 1.466586	Time Per Batch 0.133223 - F 2/5.61e+01, B 1/-1.00e+00
Train Epoch:  1 [2100/7352]	Loss: 1.171703	Time Per Batch 0.132278 - F 2/6.69e+01, B 1/-1.00e+00
Train Epoch:  1 [3100/7352]	Loss: 1.054489	Time Per Batch 0.139676 - F 2/6.94e+01, B 1/-1.00e+00
Train Epoch:  1 [4100/7352]	Loss: 1.026831	Time Per Batch 0.137940 - F 2/7.57e+01, B 1/-1.00e+00
Train Epoch:  1 [5100/7352]	Loss: 0.968135	Time Per Batch 0.136613 - F 2/7.40e+01, B 1/-1.00e+00
Train Epoch:  1 [6100/7352]	Loss: 0.887377	Time Per Batch 0.139050 - F 2/7.54e+01, B 1/-1.00e+00
Train Epoch:  1 [7100/7352]	Loss: 0.804214	Time Per Batch 0.138413 - F 2/7.72e+01, B 1/-1.00e+00
Train Epoch:  1 [7352/7352]	Loss: 0.805688	Time Per Batch 0.137976 - F 2/7.93e+01, B 1/-1.00e+00, fp=0.000940, cm=0.000001, bp=0.000448

PARALLEL: Test set epoch  1: Accuracy: 1551/2947 (53%)	Time Per Batch 0.000828
SERIAL:   Test set epoch  1: Accuracy: 1516/2947 (51%)	Time Per Batch 0.000644

Train Epoch:  2 [ 100/7352]	Loss: 0.869030	Time Per Batch 0.135895 - F 2/7.89e+01, B 1/-1.00e+00
Train Epoch:  2 [1100/7352]	Loss: 0.758809	Time Per Batch 0.130933 - F 2/7.84e+01, B 1/-1.00e+00
Train Epoch:  2 [2100/7352]	Loss: 0.612238	Time Per Batch 0.138807 - F 2/7.81e+01, B 1/-1.00e+00
Train Epoch:  2 [3100/7352]	Loss: 0.758961	Time Per Batch 0.136781 - F 2/8.15e+01, B 1/-1.00e+00
Train Epoch:  2 [4100/7352]	Loss: 0.608233	Time Per Batch 0.136415 - F 2/8.13e+01, B 1/-1.00e+00
Train Epoch:  2 [5100/7352]	Loss: 0.706127	Time Per Batch 0.137540 - F 2/7.92e+01, B 1/-1.00e+00
Train Epoch:  2 [6100/7352]	Loss: 0.597811	Time Per Batch 0.136078 - F 2/7.89e+01, B 1/-1.00e+00
Train Epoch:  2 [7100/7352]	Loss: 0.615691	Time Per Batch 0.135629 - F 2/8.47e+01, B 1/-1.00e+00
Train Epoch:  2 [7352/7352]	Loss: 0.534003	Time Per Batch 0.135740 - F 2/8.14e+01, B 1/-1.00e+00, fp=0.000915, cm=0.000001, bp=0.000450

PARALLEL: Test set epoch  2: Accuracy: 2105/2947 (71%)	Time Per Batch 0.001103
SERIAL:   Test set epoch  2: Accuracy: 1966/2947 (67%)	Time Per Batch 0.000774

Train Epoch:  3 [ 100/7352]	Loss: 0.590576	Time Per Batch 0.129564 - F 2/8.50e+01, B 1/-1.00e+00
Train Epoch:  3 [1100/7352]	Loss: 0.547530	Time Per Batch 0.148106 - F 2/8.07e+01, B 1/-1.00e+00
Train Epoch:  3 [2100/7352]	Loss: 0.582506	Time Per Batch 0.148877 - F 2/8.50e+01, B 1/-1.00e+00
Train Epoch:  3 [3100/7352]	Loss: 0.590091	Time Per Batch 0.143815 - F 2/8.41e+01, B 1/-1.00e+00
Train Epoch:  3 [4100/7352]	Loss: 0.424599	Time Per Batch 0.148093 - F 2/8.29e+01, B 1/-1.00e+00
Train Epoch:  3 [5100/7352]	Loss: 0.498275	Time Per Batch 0.147872 - F 2/8.33e+01, B 1/-1.00e+00
Train Epoch:  3 [6100/7352]	Loss: 0.477795	Time Per Batch 0.147645 - F 2/8.69e+01, B 1/-1.00e+00
Train Epoch:  3 [7100/7352]	Loss: 0.459735	Time Per Batch 0.148802 - F 2/9.00e+01, B 1/-1.00e+00
Train Epoch:  3 [7352/7352]	Loss: 0.311050	Time Per Batch 0.148254 - F 2/8.36e+01, B 1/-1.00e+00, fp=0.000997, cm=0.000001, bp=0.000495

PARALLEL: Test set epoch  3: Accuracy: 2376/2947 (81%)	Time Per Batch 0.000999
SERIAL:   Test set epoch  3: Accuracy: 2223/2947 (75%)	Time Per Batch 0.001110

Train Epoch:  4 [ 100/7352]	Loss: 0.415718	Time Per Batch 0.131343 - F 2/8.43e+01, B 1/-1.00e+00
Train Epoch:  4 [1100/7352]	Loss: 0.349135	Time Per Batch 0.145750 - F 2/8.68e+01, B 1/-1.00e+00
Train Epoch:  4 [2100/7352]	Loss: 0.263656	Time Per Batch 0.147362 - F 2/8.65e+01, B 1/-1.00e+00
Train Epoch:  4 [3100/7352]	Loss: 0.260927	Time Per Batch 0.145915 - F 2/8.72e+01, B 1/-1.00e+00
Train Epoch:  4 [4100/7352]	Loss: 0.393104	Time Per Batch 0.144999 - F 2/8.54e+01, B 1/-1.00e+00
Train Epoch:  4 [5100/7352]	Loss: 0.283164	Time Per Batch 0.145269 - F 2/9.33e+01, B 1/-1.00e+00
Train Epoch:  4 [6100/7352]	Loss: 0.305417	Time Per Batch 0.143394 - F 2/8.81e+01, B 1/-1.00e+00
Train Epoch:  4 [7100/7352]	Loss: 0.182982	Time Per Batch 0.143370 - F 2/8.86e+01, B 1/-1.00e+00
Train Epoch:  4 [7352/7352]	Loss: 0.207726	Time Per Batch 0.145311 - F 2/8.59e+01, B 1/-1.00e+00, fp=0.000980, cm=0.000001, bp=0.000481

PARALLEL: Test set epoch  4: Accuracy: 2576/2947 (87%)	Time Per Batch 0.000978
SERIAL:   Test set epoch  4: Accuracy: 2480/2947 (84%)	Time Per Batch 0.000913


   *** Proc = 0        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9768       |    1.2101e-02    |    1.2388e-06    |    2.9875e-06    |
  BckWD::bufsize                 ||       592        |    2.7285e-03    |    4.6090e-06    |    1.8187e-06    |
  BckWD::bufunpack               ||       592        |    2.3696e-02    |    4.0028e-05    |    1.7510e-05    |
  BckWD::clone                   ||      26936       |    3.6880e-01    |    1.3692e-05    |    1.9363e-05    |
  BckWD::eval                    ||      12432       |    1.0229e+01    |    8.2280e-04    |    5.8614e-04    |
  BckWD::free                    ||      27504       |    5.6062e-02    |    2.0383e-06    |    1.1730e-05    |
  BckWD::func:postrun            ||       296        |    1.9758e-01    |    6.6749e-04    |    2.7099e-04    |
  BckWD::func:precomm            ||       296        |    7.6278e-03    |    2.5770e-05    |    1.1259e-05    |
  BckWD::func:run                ||       296        |    1.3506e+01    |    4.5629e-02    |    8.6389e-03    |
  BckWD::getUVector              ||       9735       |    2.6434e-03    |    2.7153e-07    |    2.5636e-07    |
  BckWD::init                    ||        9         |    1.2500e-04    |    1.3889e-05    |    4.9595e-06    |
  BckWD::step                    ||      12432       |    1.0471e+01    |    8.4222e-04    |    6.0131e-04    |
  BckWD::sum                     ||       7104       |    9.1972e-02    |    1.2947e-05    |    6.3084e-06    |
  ForWD::access                  ||      13728       |    8.0711e-03    |    5.8793e-07    |    6.5392e-07    |
  ForWD::bufpack                 ||       4160       |    1.2774e-01    |    3.0707e-05    |    2.2623e-05    |
  ForWD::bufsize                 ||       4160       |    1.5879e-02    |    3.8170e-06    |    2.2690e-06    |
  ForWD::clone                   ||      84032       |    1.2058e+00    |    1.4349e-05    |    1.6704e-05    |
  ForWD::eval                    ||      40768       |    2.8212e+01    |    6.9201e-04    |    2.3421e-04    |
  ForWD::free                    ||      84008       |    2.1403e-01    |    2.5478e-06    |    1.2483e-05    |
  ForWD::func:precomm            ||       416        |    1.2969e-02    |    3.1175e-05    |    1.2099e-05    |
  ForWD::func:run                ||       416        |    3.9610e+01    |    9.5216e-02    |    1.7483e-02    |
  ForWD::getPrimalWithGrad-long  ||       5328       |    4.2311e+00    |    7.9413e-04    |    5.9774e-04    |
  ForWD::getPrimalWithGrad-short ||       7104       |    3.1022e-03    |    4.3668e-07    |    3.3139e-07    |
  ForWD::getUVector              ||      19023       |    1.9009e-02    |    9.9927e-07    |    1.3939e-06    |
  ForWD::init                    ||        9         |    6.9688e-04    |    7.7431e-05    |    2.4802e-05    |
  ForWD::norm                    ||       3328       |    1.3917e-01    |    4.1817e-05    |    1.1256e-05    |
  ForWD::run:postcomm            ||       416        |    1.0575e-01    |    2.5420e-04    |    1.2550e-04    |
  ForWD::run:precomm             ||       416        |    4.1283e-02    |    9.9238e-05    |    1.4707e-04    |
  ForWD::run:run                 ||       416        |    3.9406e+01    |    9.4725e-02    |    1.7355e-02    |
  ForWD::step                    ||      40768       |    2.8463e+01    |    6.9816e-04    |    2.3539e-04    |
  ForWD::sum                     ||      42432       |    4.6315e-01    |    1.0915e-05    |    8.6534e-06    |

   *** Proc = 1        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9472       |    7.6326e-03    |    8.0581e-07    |    1.1689e-06    |
  BckWD::bufpack                 ||       592        |    1.7397e-02    |    2.9386e-05    |    9.5447e-06    |
  BckWD::bufsize                 ||       1184       |    4.8188e-03    |    4.0700e-06    |    1.5063e-06    |
  BckWD::bufunpack               ||       592        |    2.1949e-02    |    3.7077e-05    |    2.2117e-05    |
  BckWD::clone                   ||      24272       |    3.5147e-01    |    1.4481e-05    |    1.1586e-05    |
  BckWD::eval                    ||      11840       |    9.8498e+00    |    8.3191e-04    |    4.8761e-04    |
  BckWD::free                    ||      24840       |    5.1170e-02    |    2.0600e-06    |    5.4733e-06    |
  BckWD::func:postrun            ||       296        |    1.9332e-01    |    6.5310e-04    |    2.4274e-04    |
  BckWD::func:precomm            ||       296        |    9.6867e-04    |    3.2725e-06    |    2.5992e-06    |
  BckWD::func:run                ||       296        |    1.3528e+01    |    4.5701e-02    |    8.6856e-03    |
  BckWD::getUVector              ||       9735       |    2.9638e-03    |    3.0445e-07    |    6.6975e-07    |
  BckWD::init                    ||        8         |    1.0612e-04    |    1.3266e-05    |    3.6916e-06    |
  BckWD::step                    ||      11840       |    1.0082e+01    |    8.5154e-04    |    5.0441e-04    |
  BckWD::sum                     ||       5920       |    7.4890e-02    |    1.2650e-05    |    5.7258e-06    |
  ForWD::access                  ||      13312       |    8.0238e-03    |    6.0275e-07    |    2.5608e-06    |
  ForWD::bufpack                 ||       4160       |    1.1462e-01    |    2.7553e-05    |    1.6647e-05    |
  ForWD::bufsize                 ||       8320       |    2.1238e-02    |    2.5527e-06    |    2.8578e-06    |
  ForWD::bufunpack               ||       4160       |    1.5319e-01    |    3.6824e-05    |    2.2235e-05    |
  ForWD::clone                   ||      76544       |    1.1267e+00    |    1.4720e-05    |    1.0581e-05    |
  ForWD::eval                    ||      40768       |    2.8118e+01    |    6.8970e-04    |    2.1136e-04    |
  ForWD::free                    ||      80680       |    1.6912e-01    |    2.0962e-06    |    7.3276e-06    |
  ForWD::func:precomm            ||       416        |    6.4512e-02    |    1.5508e-04    |    4.1057e-04    |
  ForWD::func:run                ||       416        |    3.9583e+01    |    9.5152e-02    |    1.7410e-02    |
  ForWD::getPrimalWithGrad-long  ||       5032       |    3.9820e+00    |    7.9133e-04    |    4.0513e-04    |
  ForWD::getPrimalWithGrad-short ||       6808       |    3.2239e-03    |    4.7355e-07    |    1.8089e-07    |
  ForWD::getUVector              ||      18727       |    1.8482e-02    |    9.8690e-07    |    2.0643e-06    |
  ForWD::init                    ||        8         |    6.5096e-04    |    8.1370e-05    |    3.8694e-05    |
  ForWD::norm                    ||       3328       |    1.3681e-01    |    4.1108e-05    |    9.4595e-06    |
  ForWD::run:postcomm            ||       416        |    1.0543e-01    |    2.5343e-04    |    6.7858e-05    |
  ForWD::run:precomm             ||       416        |    5.9644e-02    |    1.4338e-04    |    5.0430e-04    |
  ForWD::run:run                 ||       416        |    3.9370e+01    |    9.4639e-02    |    1.7158e-02    |
  ForWD::step                    ||      40768       |    2.8261e+01    |    6.9323e-04    |    2.1198e-04    |
  ForWD::sum                     ||      42432       |    4.6721e-01    |    1.1011e-05    |    7.7737e-06    |

   *** Proc = 2        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9472       |    8.2791e-03    |    8.7406e-07    |    6.9415e-07    |
  BckWD::bufpack                 ||       592        |    1.5179e-02    |    2.5640e-05    |    1.1472e-05    |
  BckWD::bufsize                 ||       1184       |    4.6682e-03    |    3.9427e-06    |    1.7718e-06    |
  BckWD::bufunpack               ||       592        |    2.3083e-02    |    3.8992e-05    |    1.5275e-05    |
  BckWD::clone                   ||      24272       |    3.6750e-01    |    1.5141e-05    |    1.2501e-04    |
  BckWD::eval                    ||      11840       |    9.8327e+00    |    8.3047e-04    |    4.2683e-04    |
  BckWD::free                    ||      24840       |    5.5000e-02    |    2.2142e-06    |    8.9372e-06    |
  BckWD::func:postrun            ||       296        |    1.8951e-01    |    6.4025e-04    |    2.7018e-04    |
  BckWD::func:precomm            ||       296        |    1.0007e-03    |    3.3808e-06    |    1.4418e-06    |
  BckWD::func:run                ||       296        |    1.3530e+01    |    4.5711e-02    |    8.6609e-03    |
  BckWD::getUVector              ||       9735       |    2.8790e-03    |    2.9574e-07    |    3.0140e-07    |
  BckWD::init                    ||        8         |    1.0679e-04    |    1.3349e-05    |    3.9933e-06    |
  BckWD::step                    ||      11840       |    1.0083e+01    |    8.5158e-04    |    4.4615e-04    |
  BckWD::sum                     ||       5920       |    7.0483e-02    |    1.1906e-05    |    5.2479e-06    |
  ForWD::access                  ||      13312       |    7.8830e-03    |    5.9217e-07    |    1.3768e-06    |
  ForWD::bufpack                 ||       4160       |    1.1876e-01    |    2.8549e-05    |    1.3614e-05    |
  ForWD::bufsize                 ||       8320       |    2.0530e-02    |    2.4676e-06    |    2.2833e-06    |
  ForWD::bufunpack               ||       4160       |    1.5280e-01    |    3.6730e-05    |    2.0186e-05    |
  ForWD::clone                   ||      76544       |    1.1279e+00    |    1.4735e-05    |    9.6744e-06    |
  ForWD::eval                    ||      40768       |    2.8311e+01    |    6.9445e-04    |    2.3385e-04    |
  ForWD::free                    ||      80680       |    1.8704e-01    |    2.3183e-06    |    9.3886e-06    |
  ForWD::func:precomm            ||       416        |    5.7716e-02    |    1.3874e-04    |    3.9908e-04    |
  ForWD::func:run                ||       416        |    3.9575e+01    |    9.5133e-02    |    1.7339e-02    |
  ForWD::getPrimalWithGrad-long  ||       5032       |    3.9163e+00    |    7.7828e-04    |    2.4336e-04    |
  ForWD::getPrimalWithGrad-short ||       6808       |    3.2679e-03    |    4.8002e-07    |    3.9520e-07    |
  ForWD::getUVector              ||      18727       |    1.8134e-02    |    9.6833e-07    |    9.0297e-07    |
  ForWD::init                    ||        8         |    6.1733e-04    |    7.7167e-05    |    3.0606e-05    |
  ForWD::norm                    ||       3328       |    1.3709e-01    |    4.1193e-05    |    1.0265e-05    |
  ForWD::run:postcomm            ||       416        |    1.1010e-01    |    2.6466e-04    |    1.6980e-04    |
  ForWD::run:precomm             ||       416        |    6.2221e-02    |    1.4957e-04    |    2.6462e-04    |
  ForWD::run:run                 ||       416        |    3.9352e+01    |    9.4597e-02    |    1.7157e-02    |
  ForWD::step                    ||      40768       |    2.8489e+01    |    6.9882e-04    |    2.3556e-04    |
  ForWD::sum                     ||      42432       |    4.7723e-01    |    1.1247e-05    |    1.3227e-05    |

   *** Proc = 3        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9472       |    8.0588e-03    |    8.5080e-07    |    8.0818e-07    |
  BckWD::bufpack                 ||       592        |    1.7848e-02    |    3.0148e-05    |    9.4725e-06    |
  BckWD::bufsize                 ||       592        |    2.4700e-03    |    4.1722e-06    |    1.8643e-06    |
  BckWD::clone                   ||      23680       |    3.3532e-01    |    1.4160e-05    |    1.0180e-05    |
  BckWD::eval                    ||      11248       |    9.1862e+00    |    8.1670e-04    |    4.6969e-04    |
  BckWD::free                    ||      23656       |    5.4312e-02    |    2.2959e-06    |    1.0048e-05    |
  BckWD::func:postrun            ||       296        |    1.8511e-01    |    6.2538e-04    |    2.1747e-04    |
  BckWD::func:precomm            ||       296        |    2.1098e-01    |    7.1279e-04    |    3.0801e-04    |
  BckWD::func:run                ||       296        |    1.3319e+01    |    4.4998e-02    |    8.5939e-03    |
  BckWD::getUVector              ||       9735       |    2.7742e-03    |    2.8497e-07    |    6.3850e-07    |
  BckWD::init                    ||        8         |    1.1229e-04    |    1.4037e-05    |    4.5986e-06    |
  BckWD::step                    ||      11248       |    9.3960e+00    |    8.3535e-04    |    4.8583e-04    |
  BckWD::sum                     ||       4736       |    5.5605e-02    |    1.1741e-05    |    4.9793e-06    |
  ForWD::access                  ||      13312       |    1.3789e-02    |    1.0359e-06    |    3.8046e-06    |
  ForWD::bufsize                 ||       4160       |    1.6435e-02    |    3.9508e-06    |    4.0962e-06    |
  ForWD::bufunpack               ||       4160       |    1.6107e-01    |    3.8719e-05    |    1.7883e-05    |
  ForWD::clone                   ||      76544       |    1.1274e+00    |    1.4728e-05    |    1.4321e-05    |
  ForWD::eval                    ||      40768       |    2.8438e+01    |    6.9756e-04    |    2.8904e-04    |
  ForWD::free                    ||      80680       |    2.0201e-01    |    2.5038e-06    |    1.6525e-05    |
  ForWD::func:precomm            ||       416        |    6.2255e-02    |    1.4965e-04    |    3.6166e-04    |
  ForWD::func:run                ||       416        |    3.9553e+01    |    9.5079e-02    |    1.7402e-02    |
  ForWD::getPrimalWithGrad-long  ||       4440       |    3.4495e+00    |    7.7691e-04    |    4.1684e-04    |
  ForWD::getPrimalWithGrad-short ||       6808       |    3.1876e-03    |    4.6821e-07    |    2.8718e-07    |
  ForWD::getUVector              ||      18135       |    1.6704e-02    |    9.2112e-07    |    9.6590e-07    |
  ForWD::init                    ||        8         |    6.1996e-04    |    7.7495e-05    |    3.1578e-05    |
  ForWD::norm                    ||       3328       |    1.4055e-01    |    4.2233e-05    |    1.7956e-05    |
  ForWD::run:postcomm            ||       416        |    1.0409e-01    |    2.5021e-04    |    1.9770e-04    |
  ForWD::run:precomm             ||       416        |    2.7120e-02    |    6.5193e-05    |    1.7556e-04    |
  ForWD::run:run                 ||       416        |    3.9370e+01    |    9.4639e-02    |    1.7220e-02    |
  ForWD::step                    ||      40768       |    2.8597e+01    |    7.0145e-04    |    2.8990e-04    |
  ForWD::sum                     ||      42432       |    4.8638e-01    |    1.1463e-05    |    6.9808e-06    |

TIME PER EPOCH: 1.06e+01
TIME PER TEST:  2.89e+00
```
