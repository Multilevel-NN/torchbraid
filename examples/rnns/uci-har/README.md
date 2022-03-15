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
TORCHBRAID REV: a16f1ca73724f406bc43dec45874557956c0fc68
DRIVER REV:     a16f1ca73724f406bc43dec45874557956c0fc68

MPI: procs = 4
INPUT: Namespace(seed=3649905550453979348, log_interval=10, percent_data=1.0, ensemble_size=1, sequence_length=128, tf=128.0, input_size=9, hidden_size=100, num_layers=2, num_classes=6, implicit_serial=False, batch_size=100, batch_bug=False, epochs=4, lr=0.001, use_sgd=False, force_lp=False, lp_levels=3, lp_iters=1, lp_fwd_iters=2, lp_print=0, lp_braid_print=0, lp_cfactor=4, lp_use_downcycle=False, lp_fwd_finerelax=0, lp_fwd_relax=1)

USING SEED: 3649905550453979348

Loading UCI HAR Dataset:

Using ParallelNet:
-- max_levels = 3
-- max_iters  = 1
-- fwd_iters  = 2
-- cfactor    = 4
-- fwd0 relax = 0
-- fwd relax  = 1
-- skip down  = True

Train Epoch:  1 [ 100/7352]	Loss: 1.766864	Time Per Batch 0.143418 - F 2/5.29e+01, B 1/-1.00e+00
Train Epoch:  1 [1100/7352]	Loss: 1.466586	Time Per Batch 0.139338 - F 2/5.61e+01, B 1/-1.00e+00
Train Epoch:  1 [2100/7352]	Loss: 1.171703	Time Per Batch 0.144470 - F 2/6.69e+01, B 1/-1.00e+00
Train Epoch:  1 [3100/7352]	Loss: 1.054489	Time Per Batch 0.140653 - F 2/6.94e+01, B 1/-1.00e+00
Train Epoch:  1 [4100/7352]	Loss: 1.026831	Time Per Batch 0.138289 - F 2/7.57e+01, B 1/-1.00e+00
Train Epoch:  1 [5100/7352]	Loss: 0.968135	Time Per Batch 0.138150 - F 2/7.40e+01, B 1/-1.00e+00
Train Epoch:  1 [6100/7352]	Loss: 0.887377	Time Per Batch 0.138058 - F 2/7.54e+01, B 1/-1.00e+00
Train Epoch:  1 [7100/7352]	Loss: 0.804214	Time Per Batch 0.149602 - F 2/7.72e+01, B 1/-1.00e+00
Train Epoch:  1 [7352/7352]	Loss: 0.805688	Time Per Batch 0.152026 - F 2/7.93e+01, B 1/-1.00e+00, fp=0.001038, cm=0.000001, bp=0.000492

PARALLEL: Test set epoch  1: Accuracy: 1551/2947 (53%)	Time Per Batch 0.000909
SERIAL:   Test set epoch  1: Accuracy: 1516/2947 (51%)	Time Per Batch 0.000648

Train Epoch:  2 [ 100/7352]	Loss: 0.869030	Time Per Batch 0.161138 - F 2/7.89e+01, B 1/-1.00e+00
Train Epoch:  2 [1100/7352]	Loss: 0.758809	Time Per Batch 0.147118 - F 2/7.84e+01, B 1/-1.00e+00
Train Epoch:  2 [2100/7352]	Loss: 0.612238	Time Per Batch 0.187199 - F 2/7.81e+01, B 1/-1.00e+00
Train Epoch:  2 [3100/7352]	Loss: 0.758961	Time Per Batch 0.185065 - F 2/8.15e+01, B 1/-1.00e+00
Train Epoch:  2 [4100/7352]	Loss: 0.608233	Time Per Batch 0.187313 - F 2/8.13e+01, B 1/-1.00e+00
Train Epoch:  2 [5100/7352]	Loss: 0.706127	Time Per Batch 0.188875 - F 2/7.92e+01, B 1/-1.00e+00
Train Epoch:  2 [6100/7352]	Loss: 0.597811	Time Per Batch 0.191374 - F 2/7.89e+01, B 1/-1.00e+00
Train Epoch:  2 [7100/7352]	Loss: 0.615691	Time Per Batch 0.190473 - F 2/8.47e+01, B 1/-1.00e+00
Train Epoch:  2 [7352/7352]	Loss: 0.534003	Time Per Batch 0.190036 - F 2/8.14e+01, B 1/-1.00e+00, fp=0.001269, cm=0.000001, bp=0.000643

PARALLEL: Test set epoch  2: Accuracy: 2105/2947 (71%)	Time Per Batch 0.001171
SERIAL:   Test set epoch  2: Accuracy: 1966/2947 (67%)	Time Per Batch 0.001155

Train Epoch:  3 [ 100/7352]	Loss: 0.590576	Time Per Batch 0.138844 - F 2/8.50e+01, B 1/-1.00e+00
Train Epoch:  3 [1100/7352]	Loss: 0.547530	Time Per Batch 0.149103 - F 2/8.07e+01, B 1/-1.00e+00
Train Epoch:  3 [2100/7352]	Loss: 0.582506	Time Per Batch 0.144416 - F 2/8.50e+01, B 1/-1.00e+00
Train Epoch:  3 [3100/7352]	Loss: 0.590091	Time Per Batch 0.144248 - F 2/8.41e+01, B 1/-1.00e+00
Train Epoch:  3 [4100/7352]	Loss: 0.424599	Time Per Batch 0.141874 - F 2/8.29e+01, B 1/-1.00e+00
Train Epoch:  3 [5100/7352]	Loss: 0.498275	Time Per Batch 0.141904 - F 2/8.33e+01, B 1/-1.00e+00
Train Epoch:  3 [6100/7352]	Loss: 0.477795	Time Per Batch 0.145853 - F 2/8.69e+01, B 1/-1.00e+00
Train Epoch:  3 [7100/7352]	Loss: 0.459735	Time Per Batch 0.145522 - F 2/9.00e+01, B 1/-1.00e+00
Train Epoch:  3 [7352/7352]	Loss: 0.311050	Time Per Batch 0.144962 - F 2/8.36e+01, B 1/-1.00e+00, fp=0.000973, cm=0.000001, bp=0.000485

PARALLEL: Test set epoch  3: Accuracy: 2376/2947 (81%)	Time Per Batch 0.000916
SERIAL:   Test set epoch  3: Accuracy: 2223/2947 (75%)	Time Per Batch 0.000839

Train Epoch:  4 [ 100/7352]	Loss: 0.415718	Time Per Batch 0.180367 - F 2/8.43e+01, B 1/-1.00e+00
Train Epoch:  4 [1100/7352]	Loss: 0.349135	Time Per Batch 0.186241 - F 2/8.68e+01, B 1/-1.00e+00
Train Epoch:  4 [2100/7352]	Loss: 0.263656	Time Per Batch 0.179802 - F 2/8.65e+01, B 1/-1.00e+00
Train Epoch:  4 [3100/7352]	Loss: 0.260927	Time Per Batch 0.175975 - F 2/8.72e+01, B 1/-1.00e+00
Train Epoch:  4 [4100/7352]	Loss: 0.393104	Time Per Batch 0.167702 - F 2/8.54e+01, B 1/-1.00e+00
Train Epoch:  4 [5100/7352]	Loss: 0.283164	Time Per Batch 0.162508 - F 2/9.33e+01, B 1/-1.00e+00
Train Epoch:  4 [6100/7352]	Loss: 0.305417	Time Per Batch 0.159079 - F 2/8.81e+01, B 1/-1.00e+00
Train Epoch:  4 [7100/7352]	Loss: 0.182982	Time Per Batch 0.155653 - F 2/8.86e+01, B 1/-1.00e+00
Train Epoch:  4 [7352/7352]	Loss: 0.207726	Time Per Batch 0.154667 - F 2/8.59e+01, B 1/-1.00e+00, fp=0.001042, cm=0.000001, bp=0.000514

PARALLEL: Test set epoch  4: Accuracy: 2576/2947 (87%)	Time Per Batch 0.000977
SERIAL:   Test set epoch  4: Accuracy: 2480/2947 (84%)	Time Per Batch 0.000725


   *** Proc = 0        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9768       |    1.4275e-02    |    1.4614e-06    |    4.1279e-06    |
  BckWD::bufsize                 ||       592        |    2.8668e-03    |    4.8426e-06    |    2.1239e-06    |
  BckWD::bufunpack               ||       592        |    2.6596e-02    |    4.4925e-05    |    2.3001e-05    |
  BckWD::clone                   ||      26936       |    4.1508e-01    |    1.5410e-05    |    2.2271e-05    |
  BckWD::eval                    ||      12432       |    1.1626e+01    |    9.3513e-04    |    5.9828e-04    |
  BckWD::free                    ||      27504       |    6.8099e-02    |    2.4760e-06    |    1.2598e-05    |
  BckWD::func:postrun            ||       296        |    2.3598e-01    |    7.9723e-04    |    4.2332e-04    |
  BckWD::func:precomm            ||       296        |    8.8497e-03    |    2.9898e-05    |    3.0121e-05    |
  BckWD::func:run                ||       296        |    1.5359e+01    |    5.1888e-02    |    1.1638e-02    |
  BckWD::getUVector              ||       9735       |    3.0396e-03    |    3.1223e-07    |    1.6014e-06    |
  BckWD::init                    ||        9         |    1.2879e-04    |    1.4310e-05    |    4.8404e-06    |
  BckWD::step                    ||      12432       |    1.1921e+01    |    9.5887e-04    |    6.2043e-04    |
  BckWD::sum                     ||       7104       |    1.0087e-01    |    1.4199e-05    |    2.1829e-05    |
  ForWD::access                  ||      13728       |    9.0026e-03    |    6.5578e-07    |    4.6494e-06    |
  ForWD::bufpack                 ||       4160       |    1.4328e-01    |    3.4442e-05    |    1.9398e-05    |
  ForWD::bufsize                 ||       4160       |    1.8679e-02    |    4.4902e-06    |    6.2704e-06    |
  ForWD::clone                   ||      84032       |    1.3199e+00    |    1.5708e-05    |    1.7198e-05    |
  ForWD::eval                    ||      40768       |    3.0955e+01    |    7.5930e-04    |    2.7399e-04    |
  ForWD::free                    ||      84008       |    2.6045e-01    |    3.1003e-06    |    3.1747e-05    |
  ForWD::func:precomm            ||       416        |    1.3076e-02    |    3.1433e-05    |    1.5598e-05    |
  ForWD::func:run                ||       416        |    4.3402e+01    |    1.0433e-01    |    2.2490e-02    |
  ForWD::getPrimalWithGrad-long  ||       5328       |    4.8745e+00    |    9.1487e-04    |    3.7365e-04    |
  ForWD::getPrimalWithGrad-short ||       7104       |    3.4236e-03    |    4.8192e-07    |    2.7495e-07    |
  ForWD::getUVector              ||      19023       |    2.0559e-02    |    1.0808e-06    |    1.2465e-06    |
  ForWD::init                    ||        9         |    6.7542e-04    |    7.5046e-05    |    2.3576e-05    |
  ForWD::norm                    ||       3328       |    1.4463e-01    |    4.3458e-05    |    1.8725e-05    |
  ForWD::run:postcomm            ||       416        |    1.1751e-01    |    2.8247e-04    |    2.5040e-04    |
  ForWD::run:precomm             ||       416        |    5.5530e-02    |    1.3349e-04    |    2.7161e-04    |
  ForWD::run:run                 ||       416        |    4.3174e+01    |    1.0378e-01    |    2.2332e-02    |
  ForWD::step                    ||      40768       |    3.1235e+01    |    7.6617e-04    |    2.7667e-04    |
  ForWD::sum                     ||      42432       |    4.9621e-01    |    1.1694e-05    |    8.5068e-06    |

   *** Proc = 1        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9472       |    8.6879e-03    |    9.1722e-07    |    1.9025e-06    |
  BckWD::bufpack                 ||       592        |    1.8117e-02    |    3.0603e-05    |    1.4909e-05    |
  BckWD::bufsize                 ||       1184       |    4.9388e-03    |    4.1713e-06    |    1.8261e-06    |
  BckWD::bufunpack               ||       592        |    2.3284e-02    |    3.9331e-05    |    2.1677e-05    |
  BckWD::clone                   ||      24272       |    3.7425e-01    |    1.5419e-05    |    1.9589e-05    |
  BckWD::eval                    ||      11840       |    1.0813e+01    |    9.1323e-04    |    5.5440e-04    |
  BckWD::free                    ||      24840       |    5.7175e-02    |    2.3017e-06    |    8.4238e-06    |
  BckWD::func:postrun            ||       296        |    2.3578e-01    |    7.9657e-04    |    4.1671e-04    |
  BckWD::func:precomm            ||       296        |    9.6863e-04    |    3.2724e-06    |    9.4384e-07    |
  BckWD::func:run                ||       296        |    1.5375e+01    |    5.1942e-02    |    1.1639e-02    |
  BckWD::getUVector              ||       9735       |    3.2322e-03    |    3.3202e-07    |    1.9339e-06    |
  BckWD::init                    ||        8         |    1.1413e-04    |    1.4266e-05    |    6.1739e-06    |
  BckWD::step                    ||      11840       |    1.1076e+01    |    9.3543e-04    |    5.7561e-04    |
  BckWD::sum                     ||       5920       |    7.8334e-02    |    1.3232e-05    |    8.3519e-06    |
  ForWD::access                  ||      13312       |    7.9077e-03    |    5.9403e-07    |    1.5841e-06    |
  ForWD::bufpack                 ||       4160       |    1.2313e-01    |    2.9598e-05    |    1.9883e-05    |
  ForWD::bufsize                 ||       8320       |    2.2621e-02    |    2.7189e-06    |    9.7717e-06    |
  ForWD::bufunpack               ||       4160       |    1.6226e-01    |    3.9004e-05    |    2.3870e-05    |
  ForWD::clone                   ||      76544       |    1.1724e+00    |    1.5316e-05    |    1.5871e-05    |
  ForWD::eval                    ||      40768       |    3.0593e+01    |    7.5042e-04    |    2.6247e-04    |
  ForWD::free                    ||      80680       |    1.9896e-01    |    2.4661e-06    |    1.5358e-05    |
  ForWD::func:precomm            ||       416        |    7.6099e-02    |    1.8293e-04    |    3.1758e-04    |
  ForWD::func:run                ||       416        |    4.3372e+01    |    1.0426e-01    |    2.2439e-02    |
  ForWD::getPrimalWithGrad-long  ||       5032       |    4.5352e+00    |    9.0127e-04    |    3.4291e-04    |
  ForWD::getPrimalWithGrad-short ||       6808       |    4.0494e-03    |    5.9480e-07    |    4.6321e-06    |
  ForWD::getUVector              ||      18727       |    1.9901e-02    |    1.0627e-06    |    1.9729e-06    |
  ForWD::init                    ||        8         |    6.0488e-04    |    7.5609e-05    |    3.7287e-05    |
  ForWD::norm                    ||       3328       |    1.4350e-01    |    4.3118e-05    |    3.0827e-05    |
  ForWD::run:postcomm            ||       416        |    1.1324e-01    |    2.7221e-04    |    8.6150e-05    |
  ForWD::run:precomm             ||       416        |    6.1807e-02    |    1.4857e-04    |    2.8066e-04    |
  ForWD::run:run                 ||       416        |    4.3138e+01    |    1.0370e-01    |    2.2295e-02    |
  ForWD::step                    ||      40768       |    3.0778e+01    |    7.5495e-04    |    2.6444e-04    |
  ForWD::sum                     ||      42432       |    4.8234e-01    |    1.1367e-05    |    8.2068e-06    |

   *** Proc = 2        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9472       |    9.6732e-03    |    1.0212e-06    |    3.4244e-06    |
  BckWD::bufpack                 ||       592        |    1.6462e-02    |    2.7808e-05    |    1.3616e-05    |
  BckWD::bufsize                 ||       1184       |    4.8966e-03    |    4.1357e-06    |    1.9419e-06    |
  BckWD::bufunpack               ||       592        |    2.5053e-02    |    4.2320e-05    |    2.6062e-05    |
  BckWD::clone                   ||      24272       |    3.7522e-01    |    1.5459e-05    |    1.3909e-05    |
  BckWD::eval                    ||      11840       |    1.0687e+01    |    9.0262e-04    |    5.3348e-04    |
  BckWD::free                    ||      24840       |    5.7307e-02    |    2.3070e-06    |    7.7170e-06    |
  BckWD::func:postrun            ||       296        |    2.3304e-01    |    7.8730e-04    |    4.7465e-04    |
  BckWD::func:precomm            ||       296        |    1.0477e-03    |    3.5394e-06    |    1.4881e-06    |
  BckWD::func:run                ||       296        |    1.5373e+01    |    5.1937e-02    |    1.1585e-02    |
  BckWD::getUVector              ||       9735       |    3.1401e-03    |    3.2256e-07    |    3.6594e-07    |
  BckWD::init                    ||        8         |    1.2175e-04    |    1.5219e-05    |    6.5258e-06    |
  BckWD::step                    ||      11840       |    1.0946e+01    |    9.2447e-04    |    5.5322e-04    |
  BckWD::sum                     ||       5920       |    7.3946e-02    |    1.2491e-05    |    7.3974e-06    |
  ForWD::access                  ||      13312       |    8.4044e-03    |    6.3134e-07    |    5.6572e-07    |
  ForWD::bufpack                 ||       4160       |    1.2843e-01    |    3.0873e-05    |    2.0337e-05    |
  ForWD::bufsize                 ||       8320       |    2.0784e-02    |    2.4981e-06    |    2.5433e-06    |
  ForWD::bufunpack               ||       4160       |    1.6073e-01    |    3.8637e-05    |    2.6308e-05    |
  ForWD::clone                   ||      76544       |    1.1866e+00    |    1.5503e-05    |    1.7977e-05    |
  ForWD::eval                    ||      40768       |    3.0713e+01    |    7.5336e-04    |    2.8634e-04    |
  ForWD::free                    ||      80680       |    2.0840e-01    |    2.5831e-06    |    1.2941e-05    |
  ForWD::func:precomm            ||       416        |    8.2619e-02    |    1.9860e-04    |    3.2808e-04    |
  ForWD::func:run                ||       416        |    4.3371e+01    |    1.0426e-01    |    2.2456e-02    |
  ForWD::getPrimalWithGrad-long  ||       5032       |    4.4385e+00    |    8.8205e-04    |    3.1596e-04    |
  ForWD::getPrimalWithGrad-short ||       6808       |    3.5884e-03    |    5.2709e-07    |    5.1936e-07    |
  ForWD::getUVector              ||      18727       |    1.9993e-02    |    1.0676e-06    |    1.8142e-06    |
  ForWD::init                    ||        8         |    5.9717e-04    |    7.4646e-05    |    3.5173e-05    |
  ForWD::norm                    ||       3328       |    1.4170e-01    |    4.2579e-05    |    1.5656e-05    |
  ForWD::run:postcomm            ||       416        |    1.2650e-01    |    3.0408e-04    |    3.3253e-04    |
  ForWD::run:precomm             ||       416        |    9.9643e-02    |    2.3953e-04    |    3.8125e-04    |
  ForWD::run:run                 ||       416        |    4.3088e+01    |    1.0358e-01    |    2.2193e-02    |
  ForWD::step                    ||      40768       |    3.0905e+01    |    7.5807e-04    |    2.9142e-04    |
  ForWD::sum                     ||      42432       |    4.9449e-01    |    1.1654e-05    |    1.2941e-05    |

   *** Proc = 3        ***
              timer              ||      count       |      total       |       mean       |      stdev       |
======================================================
  BckWD::access                  ||       9472       |    9.1356e-03    |    9.6448e-07    |    9.6457e-07    |
  BckWD::bufpack                 ||       592        |    1.9051e-02    |    3.2180e-05    |    1.5860e-05    |
  BckWD::bufsize                 ||       592        |    2.9586e-03    |    4.9976e-06    |    6.1981e-06    |
  BckWD::clone                   ||      23680       |    3.6831e-01    |    1.5554e-05    |    1.7305e-05    |
  BckWD::eval                    ||      11248       |    9.9475e+00    |    8.8438e-04    |    5.3579e-04    |
  BckWD::free                    ||      23656       |    5.9203e-02    |    2.5027e-06    |    1.0236e-05    |
  BckWD::func:postrun            ||       296        |    2.3059e-01    |    7.7902e-04    |    4.6810e-04    |
  BckWD::func:precomm            ||       296        |    2.2023e-01    |    7.4402e-04    |    4.4598e-04    |
  BckWD::func:run                ||       296        |    1.5151e+01    |    5.1186e-02    |    1.1510e-02    |
  BckWD::getUVector              ||       9735       |    3.4369e-03    |    3.5305e-07    |    1.9170e-06    |
  BckWD::init                    ||        8         |    9.5498e-05    |    1.1937e-05    |    4.7917e-06    |
  BckWD::step                    ||      11248       |    1.0192e+01    |    9.0609e-04    |    5.5715e-04    |
  BckWD::sum                     ||       4736       |    6.0803e-02    |    1.2839e-05    |    9.6908e-06    |
  ForWD::access                  ||      13312       |    1.3777e-02    |    1.0350e-06    |    2.9918e-06    |
  ForWD::bufsize                 ||       4160       |    1.7437e-02    |    4.1917e-06    |    3.4167e-06    |
  ForWD::bufunpack               ||       4160       |    1.7148e-01    |    4.1220e-05    |    2.5105e-05    |
  ForWD::clone                   ||      76544       |    1.1634e+00    |    1.5200e-05    |    1.2902e-05    |
  ForWD::eval                    ||      40768       |    3.0827e+01    |    7.5616e-04    |    2.7977e-04    |
  ForWD::free                    ||      80680       |    2.1451e-01    |    2.6588e-06    |    1.1126e-05    |
  ForWD::func:precomm            ||       416        |    1.0036e-01    |    2.4125e-04    |    4.2542e-04    |
  ForWD::func:run                ||       416        |    4.3329e+01    |    1.0416e-01    |    2.2406e-02    |
  ForWD::getPrimalWithGrad-long  ||       4440       |    3.9251e+00    |    8.8403e-04    |    3.3476e-04    |
  ForWD::getPrimalWithGrad-short ||       6808       |    3.3440e-03    |    4.9118e-07    |    6.4845e-07    |
  ForWD::getUVector              ||      18135       |    1.8206e-02    |    1.0039e-06    |    1.2158e-06    |
  ForWD::init                    ||        8         |    5.3292e-04    |    6.6615e-05    |    3.5902e-05    |
  ForWD::norm                    ||       3328       |    1.4458e-01    |    4.3443e-05    |    1.9453e-05    |
  ForWD::run:postcomm            ||       416        |    1.1836e-01    |    2.8453e-04    |    2.3690e-04    |
  ForWD::run:precomm             ||       416        |    3.5673e-02    |    8.5753e-05    |    2.1377e-04    |
  ForWD::run:run                 ||       416        |    4.3116e+01    |    1.0365e-01    |    2.2254e-02    |
  ForWD::step                    ||      40768       |    3.1019e+01    |    7.6086e-04    |    2.8131e-04    |
  ForWD::sum                     ||      42432       |    5.0049e-01    |    1.1795e-05    |    9.8291e-06    |

TIME PER EPOCH: 1.20e+01
TIME PER TEST:  2.94e+00
```
