#!/usr/bin/env python
import argparse
import torch
from math import pi, sin
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import gradcheck
import torchbraid
from mpi4py import MPI
# import torchbraid.utils

# MPI Stuff
rank  = MPI.COMM_WORLD.Get_rank()
procs = MPI.COMM_WORLD.Get_size()

class SineDataset(torch.utils.data.Dataset):
    """ Dataset for sine approximation
        x in [-pi,pi], y = sin(x) """

    def __init__(self, filename, size):
        self.x = []
        self.y = []
        self.length = size

        f = open(filename, "r")
        cnt = 1
        for line in f.readlines():
            words = line.split()
            self.x.append(np.float32(float(words[0])))
            self.y.append(np.float32(float(words[1])))
            cnt += 1
            if cnt > size:
                break
        f.close()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Opening layer maps x to network width
class OpenLayer(torch.nn.Module):
    def __init__(self, width):
        super(OpenLayer, self).__init__()
        self.width = width

    def forward(self,x):
        x = torch.repeat_interleave(x, repeats=self.width, dim=1)
        return x

# Closing layer takes the mean over network width
class ClosingLayer(torch.nn.Module):
    def __init__(self):
        super(ClosingLayer, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True) # take the mean of each example
        return x

# Define the steplayer
cnt_local = 0 # counting the number of layers created on this processer
class StepLayer(torch.nn.Module):
    def __init__(self, width):
        super(StepLayer, self).__init__()
        global cnt_local

        # Global identifier for this layer
        layerID = rank * nlayers / procs + cnt_local
        cnt_local = cnt_local + 1
        self.ID = layerID

        # Create linear layer
        self.linearlayer = torch.nn.Linear(width, width)
        torch.nn.init.constant_(self.linearlayer.weight, layerID) # make constant for debugging
        self.linearlayer.bias.data.fill_(0)

        print(rank ,": Creating ", layerID, "-th Layer, weights_const=", self.linearlayer.weight[0].data)

    def getID(self):
        return self.ID

    def forward(self, x):
        x = torch.tanh(self.linearlayer(x))
        return x

class ParallelNet(torch.nn.Module):
    def __init__(self, Tstop=10.0, width=4, local_steps=10, max_levels=1, max_iters=1, fwd_max_iters=0, print_level=0, braid_print_level=0, cfactor=4, fine_fcf=False, skip_downcycle=True, fmg=False):
        super(ParallelNet, self).__init__()

        step_layer = lambda: StepLayer(width)

        # Create and store parallel net
        self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD, step_layer, local_steps, Tstop, max_levels=max_levels, max_iters=max_iters)

        # Set options
        if fwd_max_iters > 0:
            # print("FWD_max_iter = ", fwd_max_iters)
            self.parallel_nn.setFwdMaxIters(fwd_max_iters)
        self.parallel_nn.setPrintLevel(print_level,True)
        self.parallel_nn.setPrintLevel(braid_print_level,False)
        self.parallel_nn.setCFactor(cfactor)
        self.parallel_nn.setSkipDowncycle(skip_downcycle)

        if fmg:
            self.parallel_nn.setFMG()
        self.parallel_nn.setNumRelax(1)         # FCF elsewehre
        if not fine_fcf:
            self.parallel_nn.setNumRelax(0,level=0) # F-Relaxation on the fine grid
        else:
            self.parallel_nn.setNumRelax(1,level=0) # F-Relaxation on the fine grid

        # this object ensures that only the LayerParallel code runs on ranks!=0
        compose = self.compose = self.parallel_nn.comp_op()

        # by passing this through 'compose' (mean composition: e.g. OpenFlatLayer o channels)
        # on processors not equal to 0, these will be None (there are no parameters to train there)
        self.openlayer = compose(OpenLayer,width)
        self.closinglayer = compose(ClosingLayer)
        # self.openlayer = OpenLayer(width)
        # self.closinglayer = ClosingLayer()


    def forward(self, x):
        x = self.compose(self.openlayer,x)
        x = self.parallel_nn(x)
        x = self.compose(self.closinglayer,x)

        return x
# end ParallelNet

# Compute L-2 norm gradient of model parameters
def gradnorm(parameters):
    norm = 0.0
    for p in parameters:
            param_norm = p.grad.data.norm(2)
            norm += param_norm.item() ** 2
    # norm = norm ** (1. / 2)
    return norm


# Parse command line
parser = argparse.ArgumentParser(description='TORCHBRAID Sine Example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N', help='batch size for training (default: 50)')
parser.add_argument('--max-levels', type=int, default=10, metavar='N', help='maximum number of braid levels (default: 10)')
args = parser.parse_args()

# Set a seed for reproducability
torch.manual_seed(0)

# Specify network
width = 2
nlayers = 10
Tstop = 10.0

# Specify training params
batch_size = args.batch_size
max_levels = args.max_levels
learning_rate = 1e-3


# Get sine data
ntraindata = 20
nvaldata = 20
training_set = SineDataset("./xy_train.dat", ntraindata)
training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)
validation_set = SineDataset("./xy_val.dat", nvaldata)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

torch.manual_seed(0)

# Layer-parallel parameters
lp_max_levels = max_levels
lp_max_iter = 1
lp_printlevel = 1
lp_braid_printlevel = 1
lp_cfactor = 2
# Number of local steps
local_steps  = int(nlayers / procs)
if nlayers % procs != 0:
    print(rank,'NLayers must be an even multiple of the number of processors: %d %d' % (nlayers, procs) )
    stop

# print(rank, "Building parallel net")
# Create layer parallel network
model = ParallelNet(Tstop=Tstop,
            width=width,
            local_steps=local_steps,
            max_levels=lp_max_levels,
            max_iters=lp_max_iter,
            fwd_max_iters=lp_max_iter,
            print_level=lp_printlevel,
            braid_print_level=lp_braid_printlevel,
            cfactor=lp_cfactor,
            fine_fcf=False,
            skip_downcycle=False,
            fmg=False)

compose = model.compose   # NOT SO SURE WHAT THAT DOES

# Enable diagnostics (?)
model.parallel_nn.diagnostics(True)

# Construct loss function
myloss = torch.nn.MSELoss(reduction='sum')

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training set: Eval one epoch
for local_batch, local_labels in training_generator:
    local_batch = local_batch.reshape(len(local_batch),1)
    local_labels= local_labels.reshape(len(local_labels),1)

    # Forward pass
    ypred = model(local_batch)
    loss = compose(myloss, ypred, local_labels)
    # loss = myloss(ypred, local_labels)

    # Comput gradient through backpropagation
    optimizer.zero_grad()
    loss.backward()


    # Compute gradient norm
    with torch.no_grad():
        gnorm = gradnorm(model.parameters())


        grads = []
        # params = []
        for param in model.parameters():
              # print("Main: model gradients: ", param.grad)
        #     # print("Model parameter: ", param)
            grads.append(param.grad.view(-1))
        #     # params.append(param.view(-1))
        print(rank, ": These are the model gradients: ", grads)
        # print("Theses are all the params:", params)

        # Print gradients
        # for p in model.parameters():
            # print("Serial grad: ", p.grad.data)


# Output
print(rank, ": Loss=", loss.item(), ", GradNorm^2=", gnorm)
# For gradient norm: Need to MPIreduce sum from all processors first, then take sqrt:
# MPIAllreduce(gnorm)
# gnorm = gnorm**(1./2.)

print("WE ARE DONE. NOTHING SHOULD HAPPEN ANYMORE.")
