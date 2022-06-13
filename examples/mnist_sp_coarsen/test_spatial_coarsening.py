from __future__ import print_function
from ast import Store
from math import ceil, sin
import sys
import argparse
import torch
import torchbraid
import torchbraid.utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import statistics as stats
from math import pi
import json

from main import ParallelNet, test, root_print, compute_levels, my_interp, my_restrict

import numpy as np
import matplotlib.pyplot as pyplot

from torchvision import datasets, transforms

from timeit import default_timer as timer

from mpi4py import MPI


def main():
    # Load arguments from file
    parser = argparse.ArgumentParser(description='TORCHBRAID CIFAR10 Example')
    # these will change which file is loaded
    parser.add_argument('--lp-activation', type=int, default=2, metavar='N',
                        help="Layer parallel activation function (0: relu, 1: sigmoid, 2: tanh, 3: leaky relu; default 0")
    parser.add_argument('--channels', type=int, default=4, metavar='N',
                        help='Number of channels in resnet layer (default: 4)')
    parser.add_argument('--sgd', action='store_true', default=False,
                        help="use stochastic gradient descent")

    # these will just change how this file is run
    parser.add_argument('--lp-levels', type=int, default=1, metavar='N',
                        help='Layer parallel levels (default: 4)')
    parser.add_argument('--lp-iters', type=int, default=2, metavar='N',
                        help='Layer parallel iterations (default: 2)')
    parser.add_argument('--lp-fwd-iters', type=int, default=-1, metavar='N',
                        help='Layer parallel (forward) iterations (default: -1, default --lp-iters)')
    parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                        help='Layer parallel internal print level (default: 0)')
    parser.add_argument('--lp-braid-print', type=int, default=0, metavar='N',
                        help='Layer parallel braid print level (default: 0)')
    parser.add_argument('--lp-cfactor', type=int, default=4, metavar='N',
                        help='Layer parallel coarsening factor (default: 4)')
    parser.add_argument('--lp-finefcf', action='store_true', default=True,
                        help='Layer parallel fine FCF on or off (default: False)')
    parser.add_argument('--lp-use-downcycle', action='store_true', default=False,
                        help='Layer parallel use downcycle on or off (default: False)')
    parser.add_argument('--lp-use-fmg', action='store_true', default=False,
                        help='Layer parallel use FMG for one cycle (default: False)')
    parser.add_argument('--lp-init-heat', action='store_true', default=False,
                        help="Layer parallel initialize convolutional kernel to the heat equation")
    parser.add_argument('--lp-sc-levels', type=int, nargs='+', default=[-2], metavar='N',
                        help="Layer parallel do spatial coarsening on provided levels (-2: no sc, -1: sc all levels, default: -2)")

    args = parser.parse_args()

    batched = args.sgd

    func_str = ("relu", "sigmoid", "tanh", "leaky_relu")
    # fname = f"examples/mnist/models/{func_str[args.lp_activation]}_net"
    fname = f"models/{func_str[args.lp_activation]}_net"
    with open(fname + f"_{args.channels}_args.json", 'r') as f:
        loaded_args = json.load(f)

    for key in args.__dict__:
        loaded_args[key] = args.__dict__[key]
    args.__dict__ = loaded_args
    print(args.__dict__)

    rank = MPI.COMM_WORLD.Get_rank()
    procs = MPI.COMM_WORLD.Get_size()

    if args.lp_init_heat:
        ker1 = torch.tensor([
            [0., 1.,  0.],
            [1.,  -4., 1.],
            [0., 1.,  0.]
        ])
        ker2 = torch.tensor([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.]
        ])
        init_conv = [ker1, ker2]
    else:
        init_conv = None

    funcs = (torch.relu, torch.sigmoid, torch.tanh,
             nn.LeakyReLU(negative_slope=0.05))
    activation = funcs[args.lp_activation]

    # logic to determine on which levels spatial coarsening is performed
    if -2 in args.lp_sc_levels:
        sc_levels = None
    elif -1 in args.lp_sc_levels:
        sc_levels = list(range(args.lp_levels))
    else:
        sc_levels = args.lp_sc_levels

    # torch.manual_seed(torchbraid.utils.seed_from_rank(args.seed,rank))
    torch.manual_seed(args.seed)

    if args.lp_levels == -1:
        min_coarse_size = 3
        args.lp_levels = compute_levels(
            args.steps, min_coarse_size, args.lp_cfactor)

    local_steps = int(args.steps/procs)
    if args.steps % procs != 0:
        root_print(rank, 'Steps must be an even multiple of the number of processors: %d %d' % (
            args.steps, procs))
        sys.exit(0)

    root_print(rank, 'MNIST ODENet:')

    def heat_init(im):
        n, m = im.shape[-2:]
        x = torch.clone(im)

        dx = pi/(n + 1)
        dy = pi/(m + 1)

        for i in range(n):
            for j in range(m):
                x[..., i, j] = sin((i+1)*dx)*sin((j+1)*dy)

        return x

    def rand_init(im):
        return torch.rand(size=im.shape)

    def to_double(im):
        return im.double()

    # read in Digits MNIST or Fashion MNIST
    if args.digits:
        root_print(rank, '-- Using Digit MNIST')
        transform = transforms.Compose([transforms.Pad((2, 2, 1, 1), padding_mode="edge"),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        # to_double,
                                        # heat_init                  # comment in to initialize all images to the sin-bump
                                        # rand_init                  # comment in to initialize all images to uniform random
                                        ])
        dataset = datasets.MNIST('./data', download=True, transform=transform)
    else:
        root_print(rank, '-- Using Fashion MNIST')
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.FashionMNIST(
            './fashion-data', download=True, transform=transform)

    root_print(rank, '-- procs    = {}\n'
               '-- channels = {}\n'
               '-- tf       = {}\n'
               '-- steps    = {}'.format(procs, args.channels, args.tf, args.steps))

    train_size = int(50000 * args.percent_data)
    test_size = int(10000 * args.percent_data)
    batch_size = args.batch_size if batched else train_size
    # test_size = 1
    train_set = torch.utils.data.Subset(dataset, range(train_size))
    test_set = torch.utils.data.Subset(
        dataset, range(train_size, train_size+test_size))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False)
    # train_loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=args.batch_size, shuffle=False)

    root_print(rank, '')
    root_print(rank, 'Using ParallelNet:')
    root_print(rank, '-- max_levels = {}\n'
               '-- max_iters  = {}\n'
               '-- fwd_iters  = {}\n'
               '-- cfactor    = {}\n'
               '-- fine fcf   = {}\n'
               '-- skip down  = {}\n'
               '-- fmg        = {}\n'
               '-- activation = {}\n'
               '-- sc levels  = {}\n'.format(args.lp_levels,
                                             args.lp_iters,
                                             args.lp_fwd_iters,
                                             args.lp_cfactor,
                                             args.lp_finefcf,
                                             not args.lp_use_downcycle,
                                             args.lp_use_fmg,
                                             func_str[args.lp_activation],
                                             args.lp_sc_levels))
    model = ParallelNet(channels=args.channels,
                        local_steps=local_steps,
                        max_levels=args.lp_levels,
                        max_iters=args.lp_iters,
                        fwd_max_iters=args.lp_fwd_iters,
                        print_level=args.lp_print,
                        braid_print_level=args.lp_braid_print,
                        cfactor=args.lp_cfactor,
                        fine_fcf=args.lp_finefcf,
                        skip_downcycle=not args.lp_use_downcycle,
                        fmg=args.lp_use_fmg,
                        Tf=args.tf,
                        sc_levels=sc_levels,
                        init_conv=init_conv,
                        activation=activation)

    compose = model.compose
    # model.double()

    if not args.lp_init_heat:
        model.load_state_dict(torch.load(fname + f"{args.channels}.pt"))

    # test(rank, model, test_loader, compose)

    def write_grad(f, grad):
        for val in torch.flatten(grad).tolist():
            f.write(f"{val}\n")

    # write the gradients to a file
    sc_label = "_None" if sc_levels is None else ''.join(
        [f"_{l}" for l in sc_levels])
    strterm = "_batched_grad.txt" if batched else "_grad.txt"
    fname = f"experiments/grads/{func_str[args.lp_activation]}_sc{sc_label}_ml_{args.lp_levels}_it_{args.lp_iters}" + strterm
    model.train()
    criterion = nn.CrossEntropyLoss()
    print("Writing grad...")
    with open(fname, 'w') as f:
        for i_batch, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = compose(criterion, output, target)
            loss.backward()
            print(f"batch: {i_batch}")
            f.write(f"batch: {i_batch}\n")
            for layer in model.parallel_nn.layer_models:
                write_grad(f, layer.conv1.weight.grad)
                write_grad(f, layer.conv1.bias.grad)
                write_grad(f, layer.conv2.weight.grad)
                write_grad(f, layer.conv2.bias.grad)
            f.write('\n')


if __name__ == '__main__':
    # torch.set_default_dtype(torch.float64)
    main()
