#!/usr/bin/env python
import argparse
import torch
from math import pi, sin
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from torch.autograd import gradcheck
import torchbraid
import torchbraid.utils as utils
from mpi4py import MPI

def root_print(rank,s):
  if rank==0:
    print(s)

# flatten model parameters or gradients into one vector of doubles:
def flatten(modelparas, gradient=False):
    with torch.no_grad():
        vec = []
        for p in modelparas:
            if gradient:
                p = p.grad
            if p != None:
                for elem in utils.pack_buffer(p):
                    vec.append(elem)
        return vec


# Central Finite Difference testing:
def runFinDiff(model, eps=1e-2):
    print("### FINITE DIFFERENCES ###")

    # get original loss and gradient:
    loss_orig = evalNetwork(gradient=True)
    grad_orig = flatten(model.parameters(), gradient=True)

    gradID=0
    maxerr = 0.0

    # Loop over model parameters
    for tens in model.parameters():

        # loop over elements in this tensor 
        for elem in tens.data.view(-1):

            # Evaluate the loss function at perturbed points loss(p+eps), loss(p-eps)
            elem += eps
            loss_p1 = evalNetwork(gradient=False)
            elem -= 2.*eps
            loss_p2 = evalNetwork(gradient=False)
            elem += eps
            # print(" Orig params: ", flatten(model.parameters()))

            # Central finite differences
            fd = (loss_p1 - loss_p2) / (2*eps)
            g = grad_orig[gradID]
            err = abs(fd - g)
            if fd == 0:
                relerr=0.0
            else:
                relerr = err/abs(fd)
            maxerr = max(maxerr, err)

            # Output
            print("FD= ", fd, " grad_orig=", g, ": abs. error=", err, ", rel. err=", relerr)
            print("Rel. err=", relerr)

            gradID = gradID+1

    return maxerr


def train(rank, model, training_generator, optimizer, epoch):

    model.train()
    for batch_idx, (local_batch, local_labels) in enumerate(training_generator):
        local_batch = local_batch.reshape(len(local_batch),1)
        local_labels= local_labels.reshape(len(local_labels),1)

        # Forward pass
        ypred = model(local_batch)
        loss = compose(myloss, ypred, local_labels)

        # Comput gradient through backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Parameter update
        optimizer.step()

        # output
        root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(local_batch), len(training_generator.dataset),
          100. * batch_idx / len(training_generator), loss.item()))
 

def test(rank, model, validation_generator):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for local_batch, local_labels in validation_generator:
            local_batch = local_batch.reshape(len(local_batch),1)
            local_labels= local_labels.reshape(len(local_labels),1)

            output = model(local_batch)
            test_loss += compose(myloss, output, local_labels)

        test_loss /= len(validation_generator.dataset)

        root_print(rank,'\nTest set: Average loss: {:.4f}\n'.format(test_loss))


fig,axs = plt.subplots()
def plot_validation(rank, model, validation_generator):

    if rank != 0:
        return

    data_in = []
    data_out = []

    with torch.no_grad():
        #prediction
        for local_batch, local_labels in validation_generator:
            local_batch = local_batch.reshape(len(local_batch),1)
            local_labels= local_labels.reshape(len(local_labels),1)

            output = model(local_batch)

            data_in.append(local_batch.tolist())
            data_out.append(output.tolist())

        data_array_in  = [item for sublist in data_in for item in sublist]
        data_array_out = [item for sublist in data_out for item in sublist]

        axs.plot(data_array_in, data_array_out, 'x')

        # exact solution
        t = np.arange(-np.pi, np.pi, 0.1)
        s = np.sin(t)
        axs.plot(t,s)

        # plt.show(block=False)



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
class StepLayer(torch.nn.Module):
    def __init__(self, width):
        super(StepLayer, self).__init__()

        # Create a linear layer.
        # print(rank ,": Creating a stepLayer")
        self.linearlayer = torch.nn.Linear(width, width, bias=True)

    def forward(self, x):
        x = torch.tanh(self.linearlayer(x))
        return x

class ParallelNet(torch.nn.Module):
    def __init__(self, Tstop=10.0, width=4, local_steps=10, max_levels=1, max_iters=1, fwd_max_iters=0, print_level=0, braid_print_level=0, cfactor=4, fine_fcf=False, skip_downcycle=True, fmg=False, nsplines=0, splinedegree=1):
        super(ParallelNet, self).__init__()

        step_layer = lambda: StepLayer(width)
        numprocs = MPI.COMM_WORLD.Get_size()

        # Create and store parallel net, use splinet flag to tell torchbraid to use spline parameterization
        self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD, step_layer, local_steps*numprocs, Tstop, max_fwd_levels=max_levels, max_bwd_levels=max_levels, max_iters=max_iters, nsplines=nsplines, splinedegree=splinedegree)

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

# MPI Stuff
rank  = MPI.COMM_WORLD.Get_rank()
procs = MPI.COMM_WORLD.Get_size()


# Parse command line
parser = argparse.ArgumentParser(description='TORCHBRAID Sine Example')
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 1, i.e. only one gradient evaluation)')
parser.add_argument('--batch-size', type=int, default=5, metavar='N', help='batch size for training (default: 5)')
parser.add_argument('--max-levels', type=int, default=10, metavar='N', help='maximum number of braid levels (default: 10)')
parser.add_argument('--max-iters', type=int, default=2, metavar='N', help='maximum number of braid iteration (default: 2)')
parser.add_argument('--nlayers', type=int, default=10, metavar='N', help='Number of Layers (i.e. time-steps) (default: 10)')
parser.add_argument('--width', type=int, default=2, metavar='N', help='Network width (default: 2)')
parser.add_argument('--nsplines', type=int, default=0, metavar='N', help='Number of splines for SpliNet (default: 0, i.e. do not use a SpliNet)')
parser.add_argument('--splinedegree', type=int, default=1, metavar='N', help='Degree of splines (default: 1, hat-functions)')
parser.add_argument('--recoverResNet', action='store_true', default=False, help='For debugging: Using a SpliNet to recover a ResNet structure.')
args = parser.parse_args()

# Set a seed for reproducability
torch.manual_seed(0)

# Specify network
width = args.width
nlayers = args.nlayers
Tstop = 10.0

# spline parameters
nsplines=args.nsplines
splinedegree=args.splinedegree

# In order to recover a ResNet using the spline basis functions, choose nspline=nlayer+1 and spline degree=1
if args.recoverResNet:
    nsplines = nlayers+1
    d=1

# Specify training params
batch_size = args.batch_size
max_epochs = args.epochs
max_levels = args.max_levels
max_iters = args.max_iters
learning_rate = 1e-1



# Get sine data
ntraindata = 50
nvaldata = 50
training_set = SineDataset("./xy_train.dat", ntraindata)
training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
validation_set = SineDataset("./xy_val.dat", nvaldata)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

# torch.manual_seed(0)

# Layer-parallel parameters
lp_max_levels = max_levels
lp_max_iter = max_iters
lp_printlevel = 2
lp_braid_printlevel = 1
lp_cfactor = 2
# Number of local steps
local_steps  = int(nlayers / procs)
if nlayers % procs != 0:
    print(rank,'NLayers must be an even multiple of the number of processors: %d %d' % (nlayers, procs) )
    stop

# Create layer parallel network
root_print(rank, "Building parallel net")
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
            fmg=False, 
            nsplines=nsplines,
            splinedegree=splinedegree)

compose = model.compose   # NOT SO SURE WHAT THAT DOES

# Construct loss function
myloss = torch.nn.MSELoss(reduction='sum')

# Set up optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


out_log = 1
out_plot = args.epochs

# Training
for epoch in range(1, args.epochs+1):

    train(rank, model, training_generator, optimizer, epoch)
  
    if epoch % out_log == 0:
        test(rank, model, validation_generator)

    if epoch % out_plot == 0:
        print(epoch, "now")
        plot_validation(rank, model, validation_generator)

plt.show()

# with torch.no_grad():
#    param_vec = flatten(model.parameters())
#    grad_vec = flatten(model.parameters(), gradient=True)

# Output
# print(rank, ": Final Loss=", loss.item())
# print(rank, ": parameters=", param_vec)
# print(rank, ": gradient=", grad_vec)
# print(rank, ": ||Grad||=", LA.norm(grad_vec))
# print("\n")


##### FINITE DIFFERENCE TESTING. Run on one core!! #######
# eps = 1e-3
# maxerr = runFinDiff(model, eps)
# print("Central Finite Difference: eps=", eps, " max. abs. error=", maxerr)
