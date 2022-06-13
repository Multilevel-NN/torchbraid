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
TORCHBRAID REV: cd10414b6871a830bc6e4f3ff2abe47e2128b5c2
DRIVER REV:     cd10414b6871a830bc6e4f3ff2abe47e2128b5c2

MPI: procs = 4
INPUT: Namespace(seed=3649905550453979348, log_interval=10, percent_data=1.0, ensemble_size=1, sequence_length=128, tf=128.0, input_size=9, hidden_size=100, num_layers=2, num_classes=6, implicit_serial=False, batch_size=100, batch_bug=False, epochs=4, lr=0.001, use_sgd=False, force_lp=False, lp_levels=3, lp_iters=1, lp_fwd_iters=2, lp_print=0, lp_cfactor=4, lp_use_downcycle=False, lp_fwd_finerelax=0, lp_fwd_relax=1, lp_fwd_tol=1e-16, lp_bwd_tol=1e-16)

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

Train Epoch:  1 [ 100/7352]	Loss: 1.766864	Time Per Batch 0.143474 - F 2/5.29e+01, B 1/-1.00e+00
Train Epoch:  1 [1100/7352]	Loss: 1.466586	Time Per Batch 0.134139 - F 2/5.61e+01, B 1/-1.00e+00
Train Epoch:  1 [2100/7352]	Loss: 1.171703	Time Per Batch 0.133799 - F 2/6.69e+01, B 1/-1.00e+00
Train Epoch:  1 [3100/7352]	Loss: 1.054489	Time Per Batch 0.132480 - F 2/6.94e+01, B 1/-1.00e+00
Train Epoch:  1 [4100/7352]	Loss: 1.026831	Time Per Batch 0.134472 - F 2/7.57e+01, B 1/-1.00e+00
Train Epoch:  1 [5100/7352]	Loss: 0.968135	Time Per Batch 0.135276 - F 2/7.40e+01, B 1/-1.00e+00
Train Epoch:  1 [6100/7352]	Loss: 0.887377	Time Per Batch 0.136603 - F 2/7.54e+01, B 1/-1.00e+00
Train Epoch:  1 [7100/7352]	Loss: 0.804214	Time Per Batch 0.136240 - F 2/7.72e+01, B 1/-1.00e+00
Train Epoch:  1 [7352/7352]	Loss: 0.805688	Time Per Batch 0.136589 - F 2/7.93e+01, B 1/-1.00e+00, fp=0.000921, cm=0.000001, bp=0.000453

PARALLEL: Test set epoch  1: Accuracy: 1551/2947 (53%)	Time Per Batch 0.000925
SERIAL:   Test set epoch  1: Accuracy: 1516/2947 (51%)	Time Per Batch 0.000678

Train Epoch:  2 [ 100/7352]	Loss: 0.869030	Time Per Batch 0.134031 - F 2/7.89e+01, B 1/-1.00e+00
Train Epoch:  2 [1100/7352]	Loss: 0.758809	Time Per Batch 0.131779 - F 2/7.84e+01, B 1/-1.00e+00
Train Epoch:  2 [2100/7352]	Loss: 0.612238	Time Per Batch 0.131384 - F 2/7.81e+01, B 1/-1.00e+00
Train Epoch:  2 [3100/7352]	Loss: 0.758961	Time Per Batch 0.131536 - F 2/8.15e+01, B 1/-1.00e+00
Train Epoch:  2 [4100/7352]	Loss: 0.608233	Time Per Batch 0.131637 - F 2/8.13e+01, B 1/-1.00e+00
Train Epoch:  2 [5100/7352]	Loss: 0.706127	Time Per Batch 0.134737 - F 2/7.92e+01, B 1/-1.00e+00
Train Epoch:  2 [6100/7352]	Loss: 0.597811	Time Per Batch 0.134179 - F 2/7.89e+01, B 1/-1.00e+00
Train Epoch:  2 [7100/7352]	Loss: 0.615691	Time Per Batch 0.133899 - F 2/8.47e+01, B 1/-1.00e+00
Train Epoch:  2 [7352/7352]	Loss: 0.534003	Time Per Batch 0.133714 - F 2/8.14e+01, B 1/-1.00e+00, fp=0.000903, cm=0.000001, bp=0.000442

PARALLEL: Test set epoch  2: Accuracy: 2105/2947 (71%)	Time Per Batch 0.000849
SERIAL:   Test set epoch  2: Accuracy: 1966/2947 (67%)	Time Per Batch 0.000658

Train Epoch:  3 [ 100/7352]	Loss: 0.590576	Time Per Batch 0.129652 - F 2/8.50e+01, B 1/-1.00e+00
Train Epoch:  3 [1100/7352]	Loss: 0.547530	Time Per Batch 0.131458 - F 2/8.07e+01, B 1/-1.00e+00
Train Epoch:  3 [2100/7352]	Loss: 0.582506	Time Per Batch 0.132750 - F 2/8.50e+01, B 1/-1.00e+00
Train Epoch:  3 [3100/7352]	Loss: 0.590091	Time Per Batch 0.132956 - F 2/8.41e+01, B 1/-1.00e+00
Train Epoch:  3 [4100/7352]	Loss: 0.424599	Time Per Batch 0.132737 - F 2/8.29e+01, B 1/-1.00e+00
Train Epoch:  3 [5100/7352]	Loss: 0.498275	Time Per Batch 0.132803 - F 2/8.33e+01, B 1/-1.00e+00
Train Epoch:  3 [6100/7352]	Loss: 0.477795	Time Per Batch 0.132599 - F 2/8.69e+01, B 1/-1.00e+00
Train Epoch:  3 [7100/7352]	Loss: 0.459735	Time Per Batch 0.132715 - F 2/9.00e+01, B 1/-1.00e+00
Train Epoch:  3 [7352/7352]	Loss: 0.311050	Time Per Batch 0.132740 - F 2/8.36e+01, B 1/-1.00e+00, fp=0.000898, cm=0.000001, bp=0.000437

PARALLEL: Test set epoch  3: Accuracy: 2376/2947 (81%)	Time Per Batch 0.000884
SERIAL:   Test set epoch  3: Accuracy: 2223/2947 (75%)	Time Per Batch 0.000675

Train Epoch:  4 [ 100/7352]	Loss: 0.415718	Time Per Batch 0.133481 - F 2/8.43e+01, B 1/-1.00e+00
Train Epoch:  4 [1100/7352]	Loss: 0.349135	Time Per Batch 0.134925 - F 2/8.68e+01, B 1/-1.00e+00
Train Epoch:  4 [2100/7352]	Loss: 0.263656	Time Per Batch 0.134086 - F 2/8.65e+01, B 1/-1.00e+00
Train Epoch:  4 [3100/7352]	Loss: 0.260927	Time Per Batch 0.133223 - F 2/8.72e+01, B 1/-1.00e+00
Train Epoch:  4 [4100/7352]	Loss: 0.393104	Time Per Batch 0.132588 - F 2/8.54e+01, B 1/-1.00e+00
Train Epoch:  4 [5100/7352]	Loss: 0.283164	Time Per Batch 0.132942 - F 2/9.33e+01, B 1/-1.00e+00
Train Epoch:  4 [6100/7352]	Loss: 0.305417	Time Per Batch 0.132770 - F 2/8.81e+01, B 1/-1.00e+00
Train Epoch:  4 [7100/7352]	Loss: 0.182982	Time Per Batch 0.132842 - F 2/8.86e+01, B 1/-1.00e+00
Train Epoch:  4 [7352/7352]	Loss: 0.207726	Time Per Batch 0.132704 - F 2/8.59e+01, B 1/-1.00e+00, fp=0.000900, cm=0.000001, bp=0.000435

PARALLEL: Test set epoch  4: Accuracy: 2576/2947 (87%)	Time Per Batch 0.000857
SERIAL:   Test set epoch  4: Accuracy: 2480/2947 (84%)	Time Per Batch 0.000690


   *** Proc = 0        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9768       |    1.0835e-02    |    1.1092e-06    |    2.7661e-06    |
  BckWD::bufsize                 ||       592        |    2.5308e-03    |    4.2750e-06    |    1.2901e-06    |
  BckWD::bufunpack               ||       592        |    2.1008e-02    |    3.5486e-05    |    9.8397e-06    |
  BckWD::clone                   ||      26936       |    3.3947e-01    |    1.2603e-05    |    6.6533e-06    |
  BckWD::eval                    ||      12432       |    9.7125e+00    |    7.8125e-04    |    3.7344e-04    |
  BckWD::free                    ||      27504       |    5.7077e-02    |    2.0752e-06    |    8.1054e-06    |
  BckWD::func:postrun            ||       296        |    1.8082e-01    |    6.1089e-04    |    1.7147e-04    |
  BckWD::func:precomm            ||       296        |    7.0192e-03    |    2.3713e-05    |    4.0443e-06    |
  BckWD::func:run                ||       296        |    1.2731e+01    |    4.3010e-02    |    3.3899e-03    |
  BckWD::getUVector              ||       9735       |    2.4608e-03    |    2.5278e-07    |    2.6674e-07    |
  BckWD::init                    ||        9         |    1.1996e-04    |    1.3329e-05    |    4.0933e-06    |
  BckWD::step                    ||      12432       |    9.9462e+00    |    8.0005e-04    |    3.9325e-04    |
  BckWD::sum                     ||       7104       |    9.0558e-02    |    1.2747e-05    |    2.2975e-05    |
  ForWD::access                  ||      13728       |    6.8855e-03    |    5.0156e-07    |    2.8573e-07    |
  ForWD::bufpack                 ||       4160       |    1.1624e-01    |    2.7941e-05    |    4.2906e-05    |
  ForWD::bufsize                 ||       4160       |    1.4313e-02    |    3.4407e-06    |    1.6808e-06    |
  ForWD::clone                   ||      84032       |    1.1110e+00    |    1.3221e-05    |    6.1148e-06    |
  ForWD::eval                    ||      40768       |    2.6573e+01    |    6.5181e-04    |    1.1767e-04    |
  ForWD::free                    ||      84008       |    2.0791e-01    |    2.4749e-06    |    1.3320e-05    |
  ForWD::func:precomm            ||       416        |    1.1695e-02    |    2.8113e-05    |    1.0315e-05    |
  ForWD::func:run                ||       416        |    3.6919e+01    |    8.8748e-02    |    7.4198e-03    |
  ForWD::getPrimalWithGrad-long  ||       5328       |    3.8463e+00    |    7.2190e-04    |    1.2594e-04    |
  ForWD::getPrimalWithGrad-short ||       7104       |    2.9378e-03    |    4.1354e-07    |    3.1883e-07    |
  ForWD::getUVector              ||      19023       |    1.7037e-02    |    8.9558e-07    |    8.4500e-07    |
  ForWD::init                    ||        9         |    6.6975e-04    |    7.4417e-05    |    2.0183e-05    |
  ForWD::norm                    ||       3328       |    1.3242e-01    |    3.9790e-05    |    6.7496e-06    |
  ForWD::run:postcomm            ||       416        |    9.7557e-02    |    2.3451e-04    |    5.0231e-05    |
  ForWD::run:precomm             ||       416        |    3.3366e-02    |    8.0208e-05    |    1.1909e-04    |
  ForWD::run:run                 ||       416        |    3.6737e+01    |    8.8310e-02    |    7.3651e-03    |
  ForWD::step                    ||      40768       |    2.6819e+01    |    6.5784e-04    |    1.2000e-04    |
  ForWD::sum                     ||      42432       |    4.3816e-01    |    1.0326e-05    |    3.9750e-06    |

   *** Proc = 1        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9472       |    7.0073e-03    |    7.3979e-07    |    1.6040e-06    |
  BckWD::bufpack                 ||       592        |    1.6109e-02    |    2.7211e-05    |    6.2940e-06    |
  BckWD::bufsize                 ||       1184       |    4.6002e-03    |    3.8853e-06    |    1.4894e-06    |
  BckWD::bufunpack               ||       592        |    1.8854e-02    |    3.1849e-05    |    5.7606e-06    |
  BckWD::clone                   ||      24272       |    3.2854e-01    |    1.3536e-05    |    6.4532e-06    |
  BckWD::eval                    ||      11840       |    9.4630e+00    |    7.9924e-04    |    3.5938e-04    |
  BckWD::free                    ||      24840       |    4.9232e-02    |    1.9820e-06    |    5.5012e-06    |
  BckWD::func:postrun            ||       296        |    1.7785e-01    |    6.0085e-04    |    1.6431e-04    |
  BckWD::func:precomm            ||       296        |    8.5387e-04    |    2.8847e-06    |    4.6081e-07    |
  BckWD::func:run                ||       296        |    1.2750e+01    |    4.3074e-02    |    3.4225e-03    |
  BckWD::getUVector              ||       9735       |    2.6318e-03    |    2.7034e-07    |    2.4752e-07    |
  BckWD::init                    ||        8         |    1.0392e-04    |    1.2989e-05    |    4.1615e-06    |
  BckWD::step                    ||      11840       |    9.6892e+00    |    8.1835e-04    |    3.7674e-04    |
  BckWD::sum                     ||       5920       |    7.2045e-02    |    1.2170e-05    |    6.5081e-06    |
  ForWD::access                  ||      13312       |    7.0876e-03    |    5.3242e-07    |    5.8179e-07    |
  ForWD::bufpack                 ||       4160       |    1.0149e-01    |    2.4397e-05    |    9.7950e-06    |
  ForWD::bufsize                 ||       8320       |    1.8325e-02    |    2.2025e-06    |    6.7044e-06    |
  ForWD::bufunpack               ||       4160       |    1.3636e-01    |    3.2778e-05    |    1.2842e-05    |
  ForWD::clone                   ||      76544       |    1.0589e+00    |    1.3833e-05    |    6.0812e-06    |
  ForWD::eval                    ||      40768       |    2.6511e+01    |    6.5029e-04    |    1.1466e-04    |
  ForWD::free                    ||      80680       |    1.7250e-01    |    2.1381e-06    |    7.9553e-06    |
  ForWD::func:precomm            ||       416        |    4.2944e-02    |    1.0323e-04    |    1.7577e-04    |
  ForWD::func:run                ||       416        |    3.6906e+01    |    8.8716e-02    |    7.3917e-03    |
  ForWD::getPrimalWithGrad-long  ||       5032       |    3.6527e+00    |    7.2590e-04    |    1.0993e-04    |
  ForWD::getPrimalWithGrad-short ||       6808       |    3.0883e-03    |    4.5363e-07    |    1.2134e-07    |
  ForWD::getUVector              ||      18727       |    1.7806e-02    |    9.5082e-07    |    1.0643e-06    |
  ForWD::init                    ||        8         |    6.3304e-04    |    7.9130e-05    |    3.3530e-05    |
  ForWD::norm                    ||       3328       |    1.3013e-01    |    3.9102e-05    |    8.2044e-06    |
  ForWD::run:postcomm            ||       416        |    1.0060e-01    |    2.4181e-04    |    3.3959e-05    |
  ForWD::run:precomm             ||       416        |    3.5750e-02    |    8.5938e-05    |    9.0240e-05    |
  ForWD::run:run                 ||       416        |    3.6720e+01    |    8.8269e-02    |    7.3552e-03    |
  ForWD::step                    ||      40768       |    2.6660e+01    |    6.5395e-04    |    1.1507e-04    |
  ForWD::sum                     ||      42432       |    4.4273e-01    |    1.0434e-05    |    4.1093e-06    |

   *** Proc = 2        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9472       |    7.3962e-03    |    7.8085e-07    |    5.0127e-07    |
  BckWD::bufpack                 ||       592        |    1.4262e-02    |    2.4091e-05    |    7.7282e-06    |
  BckWD::bufsize                 ||       1184       |    4.3769e-03    |    3.6967e-06    |    1.0335e-06    |
  BckWD::bufunpack               ||       592        |    2.1643e-02    |    3.6560e-05    |    1.0288e-05    |
  BckWD::clone                   ||      24272       |    3.2361e-01    |    1.3333e-05    |    6.0531e-06    |
  BckWD::eval                    ||      11840       |    9.4569e+00    |    7.9872e-04    |    3.5395e-04    |
  BckWD::free                    ||      24840       |    5.4715e-02    |    2.2027e-06    |    1.0483e-05    |
  BckWD::func:postrun            ||       296        |    1.7191e-01    |    5.8077e-04    |    1.5396e-04    |
  BckWD::func:precomm            ||       296        |    9.3641e-04    |    3.1635e-06    |    7.8470e-07    |
  BckWD::func:run                ||       296        |    1.2753e+01    |    4.3086e-02    |    3.4186e-03    |
  BckWD::getUVector              ||       9735       |    2.6979e-03    |    2.7714e-07    |    2.1065e-07    |
  BckWD::init                    ||        8         |    1.0654e-04    |    1.3318e-05    |    4.1904e-06    |
  BckWD::step                    ||      11840       |    9.6876e+00    |    8.1821e-04    |    3.7256e-04    |
  BckWD::sum                     ||       5920       |    6.9117e-02    |    1.1675e-05    |    3.5362e-06    |
  ForWD::access                  ||      13312       |    7.1607e-03    |    5.3791e-07    |    6.8766e-07    |
  ForWD::bufpack                 ||       4160       |    1.0764e-01    |    2.5876e-05    |    1.0591e-05    |
  ForWD::bufsize                 ||       8320       |    1.8508e-02    |    2.2245e-06    |    2.8134e-06    |
  ForWD::bufunpack               ||       4160       |    1.3735e-01    |    3.3017e-05    |    1.0308e-05    |
  ForWD::clone                   ||      76544       |    1.0654e+00    |    1.3919e-05    |    6.3153e-06    |
  ForWD::eval                    ||      40768       |    2.6608e+01    |    6.5267e-04    |    1.2485e-04    |
  ForWD::free                    ||      80680       |    1.7851e-01    |    2.2125e-06    |    8.5811e-06    |
  ForWD::func:precomm            ||       416        |    4.2873e-02    |    1.0306e-04    |    1.8265e-04    |
  ForWD::func:run                ||       416        |    3.6904e+01    |    8.8710e-02    |    7.4032e-03    |
  ForWD::getPrimalWithGrad-long  ||       5032       |    3.6418e+00    |    7.2372e-04    |    1.0572e-04    |
  ForWD::getPrimalWithGrad-short ||       6808       |    3.1933e-03    |    4.6905e-07    |    1.9071e-07    |
  ForWD::getUVector              ||      18727       |    1.7103e-02    |    9.1329e-07    |    7.6978e-07    |
  ForWD::init                    ||        8         |    6.1442e-04    |    7.6802e-05    |    3.2326e-05    |
  ForWD::norm                    ||       3328       |    1.3163e-01    |    3.9553e-05    |    6.6041e-06    |
  ForWD::run:postcomm            ||       416        |    9.9769e-02    |    2.3983e-04    |    4.4011e-05    |
  ForWD::run:precomm             ||       416        |    4.0076e-02    |    9.6336e-05    |    1.5207e-04    |
  ForWD::run:run                 ||       416        |    3.6713e+01    |    8.8251e-02    |    7.3426e-03    |
  ForWD::step                    ||      40768       |    2.6770e+01    |    6.5664e-04    |    1.2599e-04    |
  ForWD::sum                     ||      42432       |    4.5480e-01    |    1.0718e-05    |    4.2135e-06    |

   *** Proc = 3        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9472       |    7.6567e-03    |    8.0835e-07    |    5.5118e-07    |
  BckWD::bufpack                 ||       592        |    1.6580e-02    |    2.8007e-05    |    5.8236e-06    |
  BckWD::bufsize                 ||       592        |    2.2340e-03    |    3.7737e-06    |    1.1247e-06    |
  BckWD::clone                   ||      23680       |    3.0867e-01    |    1.3035e-05    |    6.4780e-06    |
  BckWD::eval                    ||      11248       |    8.6711e+00    |    7.7090e-04    |    3.5149e-04    |
  BckWD::free                    ||      23656       |    4.9811e-02    |    2.1057e-06    |    7.2342e-06    |
  BckWD::func:postrun            ||       296        |    1.7198e-01    |    5.8101e-04    |    1.6540e-04    |
  BckWD::func:precomm            ||       296        |    1.9801e-01    |    6.6895e-04    |    1.2758e-04    |
  BckWD::func:run                ||       296        |    1.2554e+01    |    4.2413e-02    |    3.3642e-03    |
  BckWD::getUVector              ||       9735       |    2.5112e-03    |    2.5796e-07    |    4.2522e-07    |
  BckWD::init                    ||        8         |    1.0638e-04    |    1.3297e-05    |    4.1827e-06    |
  BckWD::step                    ||      11248       |    8.8780e+00    |    7.8929e-04    |    3.6894e-04    |
  BckWD::sum                     ||       4736       |    5.2090e-02    |    1.0999e-05    |    3.2039e-06    |
  ForWD::access                  ||      13312       |    1.2819e-02    |    9.6294e-07    |    2.5535e-06    |
  ForWD::bufsize                 ||       4160       |    1.5107e-02    |    3.6314e-06    |    1.5464e-06    |
  ForWD::bufunpack               ||       4160       |    1.4200e-01    |    3.4136e-05    |    1.1084e-05    |
  ForWD::clone                   ||      76544       |    1.0578e+00    |    1.3820e-05    |    6.5373e-06    |
  ForWD::eval                    ||      40768       |    2.6553e+01    |    6.5131e-04    |    1.3746e-04    |
  ForWD::free                    ||      80680       |    1.7816e-01    |    2.2082e-06    |    9.7717e-06    |
  ForWD::func:precomm            ||       416        |    5.8479e-02    |    1.4057e-04    |    1.9916e-04    |
  ForWD::func:run                ||       416        |    3.6887e+01    |    8.8670e-02    |    7.3818e-03    |
  ForWD::getPrimalWithGrad-long  ||       4440       |    3.1812e+00    |    7.1650e-04    |    1.1313e-04    |
  ForWD::getPrimalWithGrad-short ||       6808       |    3.1200e-03    |    4.5828e-07    |    2.3770e-07    |
  ForWD::getUVector              ||      18135       |    1.8765e-02    |    1.0347e-06    |    1.1036e-06    |
  ForWD::init                    ||        8         |    5.4416e-04    |    6.8021e-05    |    3.6724e-05    |
  ForWD::norm                    ||       3328       |    1.3705e-01    |    4.1182e-05    |    8.7071e-06    |
  ForWD::run:postcomm            ||       416        |    9.3558e-02    |    2.2490e-04    |    7.5412e-05    |
  ForWD::run:precomm             ||       416        |    2.0835e-02    |    5.0085e-05    |    6.1615e-05    |
  ForWD::run:run                 ||       416        |    3.6725e+01    |    8.8282e-02    |    7.3499e-03    |
  ForWD::step                    ||      40768       |    2.6714e+01    |    6.5527e-04    |    1.3823e-04    |
  ForWD::sum                     ||      42432       |    4.5602e-01    |    1.0747e-05    |    5.4574e-06    |

TIME PER EPOCH: 1.00e+01
TIME PER TEST:  2.60e+00
```
