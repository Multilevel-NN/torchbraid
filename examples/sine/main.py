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

from bspline import evalBsplines


def root_print(rank,s):
  if rank==0:
    print(s)


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

# ResNet layer
class StepLayer(torch.nn.Module):
    def __init__(self, width):
        super(StepLayer, self).__init__()
        self.linearlayer = torch.nn.Linear(width, width)

    def forward(self, x):
        x = torch.tanh(self.linearlayer(x))
        return x

class StepWithSpline(torch.nn.Module):
    def __init__(self, linlayers, nSplines, splinedegree, deltaKnots):
        super(StepWithSpline, self).__init__()

        self.linlayers = linlayers
        self.splinedegree = splinedegree
        self.deltaKnots = deltaKnots
        self.splinecoeffs = np.zeros(splinedegree + 1)
        self.time = 0.0
        self.k = 0

    def setTime(self, time):
        self.time = time
        self.k = int(time / self.deltaKnots)


    def forward(self, x):
            print("Eval spline at t=", self.time, ", k=", self.k)
            evalBsplines(self.splinedegree, self.deltaKnots, self.time, self.splinecoeffs)

            y = torch.zeros(x.size())
            for l in range(self.splinedegree + 1):
                y = y + self.splinecoeffs[l] * self.linlayers[self.k+l](x)

            y = torch.tanh(y)
            return y

class SerialSpliNet(torch.nn.Module):
    """ Network definition """
    def __init__(self, width, nlayers, nSplines, splinedegree, Tstop):
        #Constructor
        super(SerialSpliNet, self).__init__()

        self.stepsize = Tstop / float(nlayers)

        self.nSplines = nSplines
        self.splinedegree = splinedegree
        self.nKnots = nSplines - splinedegree + 1
        self.deltaKnots = self.stepsize
        self.splinecoeffs = np.zeros(splinedegree + 1)   # no gradients needed. Not sure if a numpy array is fine here...

        # Layers
        self.openlayer = OpenLayer(width)
        self.closinglayer = ClosingLayer()
        self.layers= torch.nn.ModuleList([torch.nn.Linear(width, width) for i in range(nSplines)])

    def forward(self, x):
        # print("---- FWD SpliNet ---- ")
        # Opening Layer
        x = self.openlayer(x)

        # Hidden layers aka Timestepping
        for i in range(nlayers):
            time = i * self.stepsize
            k = int(time / self.deltaKnots)

            # Eval splines
            evalBsplines(self.splinedegree, self.deltaKnots, time, self.splinecoeffs)

            # sum over splines and update x
            for d in range(self.splinedegree + 1):
                x = x + self.stepsize * torch.tanh(self.splinecoeffs[d] * self.layers[k+d](x))

        # Closing layer
        x = self.closinglayer(x)
        return x


class SerialResnet(torch.nn.Module):
    """ Network definition """
    def __init__(self, width, nlayers, Tstop):
        #Constructor
        super(SerialResnet, self).__init__()

        self.stepsize = Tstop / float(nlayers)
        # Layers
        self.layers= torch.nn.ModuleList([StepLayer(width) for i in range(nlayers)])
        self.openlayer = OpenLayer(width)
        self.closinglayer = ClosingLayer()

    def forward(self, x):
        # print("---- FWD ResNet ---- ")
        # Opening Layer
        x = self.openlayer(x)

        # Hidden layers
        for i, layer in enumerate(self.layers):
            x = x + self.stepsize * layer(x)

        # Closing layer
        x = self.closinglayer(x)
        return x


class ParallelNet(torch.nn.Module):
    def __init__(self, Tstop=10.0, width=4, local_steps=10, max_levels=1, max_iters=1, fwd_max_iters=0, print_level=0, braid_print_level=0, cfactor=4, fine_fcf=False, skip_downcycle=True, fmg=False, nSplines=10, splinedegree=0):
        super(ParallelNet, self).__init__()

        # Create lambda function for normal step layer
        if splinedegree >= 1:
            # Create d+1 linear layers
            linlayers = torch.nn.ModuleList([torch.nn.Linear(width, width) for i in range(nSplines)])

            # Set up step_layer
            nKnots = nSplines - splinedegree + 1
            deltaKnots = Tstop / (nKnots - 1)
            step_layer = lambda: StepWithSpline(linlayers, nSplines, splinedegree, deltaKnots)
        else:
            step_layer = lambda: StepLayer(width)

        # Create and store parallel net
        self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD, step_layer, local_steps, Tstop, max_levels=max_levels, max_iters=max_iters)

        # Set options
        if fwd_max_iters > 0:
            print("FWD_max_iter = ", fwd_max_iters)
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


    def forward(self, x):
        x = self.compose(self.openlayer,x)
        x = self.parallel_nn(x)
        # print("Parallel NN(x) = ", x)
        x = self.compose(self.closinglayer,x)

        return x
# end ParallelNet


# Compute L-2 norm gradient of model parameters
def gradnorm(parameters):
    norm = 0.0

    # print("Trainable parameters:")
    for p in parameters:
            #print(name, p.data, p.requires_grad)
            param_norm = p.grad.data.norm(2)
            norm += param_norm.item() ** 2
    norm = norm ** (1. / 2)
    return norm


# Parse command line
parser = argparse.ArgumentParser(description='TORCHBRAID Sine Example')
parser.add_argument('--force-lp', action='store_true', default=False, help='Use layer parallel even if there is only 1 MPI rank')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 2)')
parser.add_argument('--batch-size', type=int, default=20, metavar='N', help='batch size for training (default: 50)')
parser.add_argument('--plot', default=True, help='Plot the results (default: true)')
parser.add_argument('--splinet', action='store_true', default=False, help='Use SpliNet instead of Resnet')
args = parser.parse_args()

if args.splinet:
    splinet = True
else:
    splinet = False


# MPI Stuff
rank  = MPI.COMM_WORLD.Get_rank()
procs = MPI.COMM_WORLD.Get_size()


# some logic to default to Serial if on one processor,
# can be overriden by the user to run layer-parallel
if args.force_lp:
    force_lp = True
elif procs>1:
    force_lp = True
else:
    force_lp = False

# Set a seed for reproducability
torch.manual_seed(0)

# Specify network
width = 4
nlayers = 5
Tstop = 5.0

# Specify training params
batch_size = args.batch_size
max_epochs = args.epochs
learning_rate = 1e-3


# Get sine data
ntraindata = 5
nvaldata = 20
training_set = SineDataset("./xy_train.dat", ntraindata)
training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False)
validation_set = SineDataset("./xy_val.dat", nvaldata)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

# Serial network model
torch.manual_seed(0)
if not force_lp:
    if splinet:
        # Create SpliNet. These setting ensure that SpliNet gives same results as ResNet.
        splinedegree = 1
        nSplines = nlayers + splinedegree

        print("Building serial SpliNet, nsplines=", nSplines)
        model = SerialSpliNet(width, nlayers, nSplines, splinedegree, Tstop)
        compose = lambda op,*p: op(*p)    # NO IDEA WHAT THAT IS
    else:
        # Create ResNet
        root_print(rank, "Building serial Resnet")
        model = SerialResnet(width, nlayers, Tstop)
        compose = lambda op,*p: op(*p)    # NO IDEA WHAT THAT IS
else:
    root_print(rank, "Building parallel net")
    # Layer-parallel parameters
    lp_max_levels = 1
    lp_max_iter = 10
    lp_printlevel = 0
    lp_braid_printlevel = 0
    lp_cfactor = 2
    # Number of local steps
    local_steps  = int(nlayers / procs)
    if nlayers % procs != 0:
        print(rank,'NLayers must be an even multiple of the number of processors: %d %d' % (nlayers, procs) )
        stop

    if splinet:
        # Create SpliNet. These setting ensure that SpliNet gives same results as ResNet.
        splinedegree = 1
        nSplines = nlayers + splinedegree
    else:
        splinedegree = 0
        nSplines = 0

    # Create layer parallel network
    model = ParallelNet(Tstop=Tstop,
                        width=width,
                        local_steps=local_steps,
                        max_levels=lp_max_levels,
                        max_iters=lp_max_iter,
                        fwd_max_iters=10,
                        print_level=lp_printlevel,
                        braid_print_level=lp_braid_printlevel,
                        cfactor=lp_cfactor,
                        fine_fcf=False,
                        skip_downcycle=False,
                        fmg=False,
                        nSplines=nSplines,
                        splinedegree=splinedegree)
    compose = model.compose   # NOT SO SURE WHAT THAT DOES

# Construct loss function
myloss = torch.nn.MSELoss(reduction='sum')

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Prepare output
print("rank, epoch, loss, loss_val, gnorm")

# # Debug gradient
torch.autograd.set_detect_anomaly(True)

# Training loop
for epoch in range(max_epochs):
    # print("## Epoch ", epoch)

    # TRAINING SET: Train one epoch
    for local_batch, local_labels in training_generator:
        local_batch = local_batch.reshape(len(local_batch),1)
        local_labels= local_labels.reshape(len(local_labels),1)

        # Forward pass
        ypred = model(local_batch)
        loss = compose(myloss, ypred, local_labels)

        # Comput gradient through backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Print gradients
        # for p in model.parameters():
            # print("Serial grad: ", p.grad.data)

        # Update network parameters
        optimizer.step()

    # # VALIDATION
    # for local_batch, local_labels in validation_generator:
    #     with torch.no_grad():
    #
    #         local_batch = local_batch.reshape(len(local_batch),1)
    #         local_labels= local_labels.reshape(len(local_labels),1)
    #         ypred = model(local_batch)
    #         loss_val = compose(myloss, ypred, local_labels).item()
    loss_val = 0.0


    # Output and stopping
    with torch.no_grad():
        gnorm = gradnorm(model.parameters())
        if rank == 0:
            print(rank, epoch, loss.item(), loss_val, gnorm)

    # Stopping criterion
    if gnorm < 1e-4:
        break



# plot validation and training
if args.plot is True:
    xtrain = torch.tensor(training_set[0:len(training_set)])[0].reshape(len(training_set),1)
    ytrain = model(xtrain).detach().numpy()
    xval = torch.tensor(validation_set[0:len(validation_set)])[0].reshape(len(validation_set),1)
    yval = model(xval).detach().numpy()
    # ytrain = MPI.COMM_WORLD.bcast(ytrain,root=0)
    # yval = MPI.COMM_WORLD.bcast(yval,root=0)

    if rank == 0:
        plt.plot(xtrain, ytrain, 'ro')
        plt.plot(xval, yval, 'bo')
        # Groundtruth
        xtruth = np.arange(-pi, pi, 0.1)
        plt.plot(xtruth, np.sin(xtruth))

        # Shot the plot
        plt.legend(['training', 'validation', 'groundtruth'])
        plt.show()
