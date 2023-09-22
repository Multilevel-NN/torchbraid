# @HEADER
# ************************************************************************
# 
#                        Torchbraid v. 0.1
# 
# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC 
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. 
# Government retains certain rights in this software.
# 
# Torchbraid is licensed under 3-clause BSD terms of use:
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name National Technology & Engineering Solutions of Sandia, 
# LLC nor the names of the contributors may be used to endorse or promote 
# products derived from this software without specific prior written permission.
# 
# Questions? Contact Eric C. Cyr (eccyr@sandia.gov)
# 
# ************************************************************************
# @HEADER


# Train a graph neural network for CORA, Pubmed and Citeseer
# Network architecture is Encoder + PDE-GCN Layers + Decoder
#
#
# Running the following should reproduce parallel output of node_classification_notebook.ipynb
# $ mpirun -np 4 python3 node_classification_script.py
#
# To reproduce serial node_classification_notebook.ipynb output, do
# $ mpirun -np 4 python3 node_classification_script.py --lp-max-levels 1
#
# Many options are available, see
# $ python3 node_classification_script.py --help
#
# For example, 
# $ mpirun -np 4 python3 node_classification_script.py --dataset CORA --steps 24 --channels 8 --percent-data 0.25 --batch-size 100 --epochs 5 
# trains on 25% of fashion MNIST with 24 steps in the PDE-GCN, 8 channels, 100 images per batch over 5 epochs 


from __future__ import print_function

import statistics as stats
from timeit import default_timer as timer
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T

import torchbraid
import torchbraid.utils
from torchvision import datasets, transforms
from torch_geometric.datasets import Planetoid
import graphOps as GO
from network_architecture import parse_args, ParallelGraphNet
from mpi4py import MPI

def index_op(lhs,index):
    return lhs[index]

# Train model for one epoch
# Return values: per batch losses and training times, model parameters updated in-place
def train(rank, params, model, data, optimizer, epoch, compose, device):
    # set the model in training mode.
    model.train()
    
    start_time = timer()
    data= data.to(device)
    # clear (zero out) the gradients of all the parameters 
    # optimized by the optimizer. Usually used before .backward() called
    optimizer.zero_grad()

    # extracting the information in graph
    I = data.edge_index[0, :]   # source node 
    J = data.edge_index[1, :]   # target node
    N = data.y.shape[0]         # number of node

    # data.x: the features associated with each node in the graph.
    # squeeze(): remove dimensions with size 1 from the tensor. 
    # t(): transpose operation, swaps the dimensions of the tensor.
    features = data.x.squeeze().t() # features: [num_features, num_nodes]

    # D1 =  torch.sum(features ** 2, dim=0, keepdim=True) [1, num_nodes]
    # calculates the sum of squared features along the rows, 
    # resulting in a row tensor of shape [1, num_nodes].

    # D1_t = D1.t() [num_nodes,1]
    # D2 = D1 + D1_t() [num_nodes,num_nodes] 
    # features.t() @ features： [num_nodes, num_nodes]
            # D: [num_nodes, num_nodes] (for citeer is [3327, 3327])
    D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                    torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                    2 * features.t() @ features)
    # Normalization by dividing D by its standard deviation:
    D = D / D.std()
    # Exponential transformation using the negative values of D:
    D = torch.exp(-2 * D)
    # results in smaller values for closer distances 
    # and larger values for farther distances.

    w = D[I, J] #tmp. replaced inside the net for gcn norm
    # create a graph G using the indices I and J, 
    # the total number of nodes N, and the edge weights w.
    G = GO.graph(I, J, N, W=w, pos=None, faces=None)
    G = G.to(device)

    # transposes the feature tensor data.x 
    # and adds an extra dimension at the
    # beginning using .unsqueeze(0).
    # reshapes to [1, num_features, num_nodes]
    xn = data.x.t().unsqueeze(0)
    # model returns two outputs: 
    # out represents the output predictions, 
    # G represents the updated graph.
    out = model(xn,[],Graph=G,rank=rank)
    # nll_loss: calculates the negative log-likelihood loss 
    # loss between the predicted out values and the target labels data.y 
    # only for the nodes specified by data.train_mask.
    criterion = F.nll_loss
    
    out_masked = compose(index_op,out,data.train_mask)
    loss = compose(criterion, out_masked, data.y[data.train_mask])
    # gradients of the loss with respect to the model's parameters 
    # using backpropagation.
    loss.backward()
    stop_time = timer()
    # updates the model's parameters by performing an optimization step 
    # using torch.optim.Adam
    optimizer.step()  # Update parameters based on gradients.
    train_time = stop_time - start_time
    return float(loss) , train_time

# Evaluate model on validation data
# Return: number of correctly classified test items, total number of test items, loss on test data set
@torch.no_grad()
def test(model, data, device):
    model.eval()
    I = data.edge_index[0, :]
    J = data.edge_index[1, :]
    N = data.y.shape[0]
    features = data.x.squeeze().t()
    D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                    torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                    2 * features.t() @ features)

    D = D / D.std()
    D = torch.exp(-2 * D)
    w = D[I, J]
    G = GO.graph(I, J, N, W=w, pos=None, faces=None)
    G = G.to(device)
    xn = data.x.t().unsqueeze(0)
    out = model(xn,[],Graph=G)  
    ###################
    # .argmax(dim=-1): computes the index of the maximum value along the 
    # last dimension of out, resulting in the predicted class labels pred. 
    # accs: initialized as an empty list to store the accuracies.
    pred, accs = out.argmax(dim=-1), []

    # This loop iterates over the masks from data corresponding 
    # to the subsets: 'train_mask', 'val_mask', and 'test_mask'. 
    # For each mask, it computes the accuracy by comparing the 
    # predicted labels pred with the ground truth labels data.y 
    # using pred[mask] == data.y[mask]. 
    # The .sum() function counts the number of correct predictions. 
    # It then divides the number of correct predictions by the total 
    # number of samples in the mask using int(mask.sum()) to obtain 
    # the accuracy. The accuracy is appended to the accs list.
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred_masked = model.compose(index_op,pred,mask)
        accs.append(int((pred_masked == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def remove_outliers(array):
    median = stats.median(array)
    for ind in range(0,len(array)):
        if(array[ind]>median*10):
            if ind==0:
                array[ind] = array[ind+1]
            array[ind] = array[ind-1]
    return array

def plot_save_metric_data(metrics={},title="",):
    colors = ['-r','-g','-b','-c','-m','-k']
    i = 0
    plt.title(title)
    for key in metrics:
        value = remove_outliers(metrics[key])
        length = len(value)
        iters = range(1,length+1)
        plt.plot(iters,value,colors[i],label=key)
        i+=1
    plt.legend(loc='upper left')
    if title[0]=="A":
        path = "accuracy"
    elif title[0]=="L":
        path = "loss"
    else:
        path = "time"
    my_path = os.path.abspath(__file__) 
    my_path = os.path.dirname(my_path)
    plt.savefig(my_path+"/"+path+"/"+title+".png")  # should before show method
    plt.clf()


##
# Parallel printing helper function  
def root_print(rank, s):
  if rank == 0:
    print(s)


def main():
    # Begin setting up run-time environment 
    # MPI information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procs = comm.Get_size()
    args = parse_args()

    # Use device or CPU?
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device, host = torchbraid.utils.getDevice(comm=comm)
    if not use_cuda:
        device = torch.device("cuda" if use_cuda else "cpu")
    print(rank,f'Run info rank: {rank}: Torch version: {torch.__version__} | Device: {device} | Host: {host}')

    # Set seed for reproducibility
    # torch.manual_seed(args.seed)

    # Compute number of steps in PDE-GCN per processor
    local_steps = int(args.steps / procs)

    # Read in CORA ,Citeseer or PubMed datasets
    root_print(rank, 'Node Classification PDE GCN:')
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)


    # set the transform to be normalize features 
    # (Row-normalizes the attributes given in attrs to sum-up to one)
    transform = T.Compose([T.NormalizeFeatures()])
    
    if args.dataset == 'CORA':
        root_print(rank, '-- Using CORA')
        dataset = Planetoid(root=data_dir, name='Cora', transform=transform)
        nNin = 1433     # number of unique words (features)
    elif args.dataset == 'CiteSeer':
        root_print(rank, '-- Using CiteSeer')
        dataset = Planetoid(root=data_dir, name='CiteSeer', transform=transform)
        nNin = 3703
    elif args.dataset == 'PubMed':
        root_print(rank, '-- Using PubMed')
        dataset = Planetoid(root=data_dir, name='PubMed', transform=transform)
        nNin = 500
    else: 
        root_print(rank,f'Unrecognized dataset {args.dataset}')
        return 0

    nEin = 1
    # number of channels (different from fully_supervised, which has 64)
    n_channels = 256    # trial.suggest_categorical('n_channels', [64, 128, 256])
    nopen = n_channels
    nhid = n_channels
    nNclose = n_channels
    batchSize = 32 # number of samples processed before the model is updated.

    # Finish assembling training and test datasets
    root_print(rank,f'Dataset: {dataset}:')
    root_print(rank,'======================')
    root_print(rank,f'Number of graphs: {len(dataset)}')
    root_print(rank,f'Number of features: {dataset.num_features}')
    root_print(rank,f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    root_print(rank," ")
    root_print(rank,data)
    root_print(rank,'===========================================================================================================')

    # Gather some statistics about the graph.
    root_print(rank,f'Number of nodes: {data.num_nodes}')
    root_print(rank,f'Number of edges: {data.num_edges}')
    root_print(rank,f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    root_print(rank,f'Number of training nodes: {data.train_mask.sum()}')
    root_print(rank,f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    root_print(rank,f'Has isolated nodes: {data.has_isolated_nodes()}')
    root_print(rank,f'Has self-loops: {data.has_self_loops()}')
    root_print(rank,f'Is undirected: {data.is_undirected()}')

    # Diagnostic information
    root_print(rank, '-- procs    = {}\n'
                    '-- channels = {}\n'
                    '-- Tf       = {}\n'
                    '-- steps    = {}\n'
                    '-- max_levels     = {}\n'
                    '-- max_bwd_iters  = {}\n'
                    '-- max_fwd_iters  = {}\n'
                    '-- cfactor        = {}\n'
                    '-- fine fcf       = {}\n'
                    '-- skip down      = {}\n'.format(procs, args.channels, 
                                                        args.Tf, args.steps,
                                                        args.lp_max_levels,
                                                        args.lp_bwd_max_iters,
                                                        args.lp_fwd_max_iters,
                                                        args.lp_cfactor,
                                                        args.lp_fine_fcf,
                                                        not args.lp_use_downcycle) )    

    data = data.to(device)
    # extracting the information in graph
    I = data.edge_index[0, :]   # source node 
    J = data.edge_index[1, :]   # target node
    N = data.y.shape[0]         # number of node

    # data.x: the features associated with each node in the graph.
    # squeeze(): remove dimensions with size 1 from the tensor. 
    # t(): transpose operation, swaps the dimensions of the tensor.
    features = data.x.squeeze().t() # features: [num_features, num_nodes]

    # D1 =  torch.sum(features ** 2, dim=0, keepdim=True) [1, num_nodes]
    # calculates the sum of squared features along the rows, 
    # resulting in a row tensor of shape [1, num_nodes].

    # D1_t = D1.t() [num_nodes,1]
    # D2 = D1 + D1_t() [num_nodes,num_nodes] 
    # features.t() @ features： [num_nodes, num_nodes]
            # D: [num_nodes, num_nodes] (for citeer is [3327, 3327])
    D = torch.relu(torch.sum(features ** 2, dim=0, keepdim=True) + \
                    torch.sum(features ** 2, dim=0, keepdim=True).t() - \
                    2 * features.t() @ features)
    # Normalization by dividing D by its standard deviation:
    D = D / D.std()
    # Exponential transformation using the negative values of D:
    D = torch.exp(-2 * D)
    # results in smaller values for closer distances 
    # and larger values for farther distances.

    w = D[I, J] #tmp. replaced inside the net for gcn norm
    # create a graph G using the indices I and J, 
    # the total number of nodes N, and the edge weights w.
    Graph = GO.graph(I, J, N, W=w, pos=None, faces=None)
    Graph = Graph.to(device)

    hyper_parameters_datasets = {
        "CORA" : {
            "8":  {'h': 2.75, 'dropout': 0.8, 'lr': 0.0017343966715010824, 'lrGCN': 0.00024138503983822774, 'wd': 0.00014416129160101603, 'lr_alpha': 2.1914060049579967e-05},
            "16": {'h': 1.0, 'dropout': 0.7, 'lr': 0.002408417397652311, 'lrGCN': 2.9386512408090523e-05, 'wd': 0.00033035514207532037, 'lr_alpha': 0.0039007013583180652},
            "32": {'h': 0.875, 'dropout': 0.7, 'lr': 0.006102254566580768, 'lrGCN': 1.2660249012543178e-05, 'wd': 0.0006373023961365857, 'lr_alpha': 0.0007038973758344859},
            "64": {'h': 0.796875, 'dropout': 0.8, 'lr': 0.008819856508590504, 'lrGCN': 1.8702187212357512e-05, 'wd': 0.00024200526856409135, 'lr_alpha': 0.00016337842151693108}
        },
        "CiteSeer" : {
            "8": {},
            "16": {},
            "32": {},
            "64":{'h': 0.90625, 'dropout': 0.6, 'lr': 0.0026835403361853212, 'lrGCN': 5.067346690238226e-05, 'wd': 2.7416952046582594e-05, 'lr_alpha': 0.0010118119877900122}
        },
        "PubMed" : {
            "8":{'h': 2.4375, 'dropout': 0.7, 'lr': 0.002672024141685073, 'lrGCN': 4.5256636114969606e-05, 'wd': 0.00022110173620958117, 'lr_alpha': 0.00019229692933031442},
            "16":{'h': 2.4375, 'dropout': 0.7, 'lr': 0.002672024141685073, 'lrGCN': 4.5256636114969606e-05, 'wd': 0.00022110173620958117, 'lr_alpha': 0.00019229692933031442},
            "32":{},
            "64":{}
        },        
    }
    # Carry out parallel training
    epoch_losses = {"8":[],"16":[],"32":[],"64":[]}
    epoch_accuracies = {"8":[],"16":[],"32":[],"64":[]}
    epoch_times = {"8":[],"16":[],"32":[],"64":[]}
    num_layers = [8,16,32,64]
    for nl in num_layers:
        # Create layer-parallel Graph Neural Network
        # Note this can be done on only one processor, but will be slow
        # setup the model
        h = hyper_parameters_datasets[args.dataset][str(nl)]['h']
        dropout = hyper_parameters_datasets[args.dataset][str(nl)]['dropout']
        wd = hyper_parameters_datasets[args.dataset][str(nl)]['wd']
        lr = hyper_parameters_datasets[args.dataset][str(nl)]['lr']
        lrGCN = hyper_parameters_datasets[args.dataset][str(nl)]['lrGCN']
        lr_alpha = hyper_parameters_datasets[args.dataset][str(nl)]['lr_alpha']

        model = ParallelGraphNet(nNin, nopen, nhid, nNclose, nl, h=h, dense=False, varlet=True,wave=False, 
                                diffOrder=1, num_output=dataset.num_classes, dropOut=dropout, gated=False, 
                                realVarlet=False, mixDyamics=False, doubleConv=False, tripleConv=False, 
                                max_levels=args.lp_max_levels, 
                                bwd_max_iters=args.lp_bwd_max_iters, 
                                fwd_max_iters=args.lp_fwd_max_iters,
                                print_level=args.lp_print_level,
                                braid_print_level=args.lp_braid_print_level,
                                cfactor=args.lp_cfactor,
                                fine_fcf=args.lp_fine_fcf,
                                skip_downcycle=not args.lp_use_downcycle,
                                fmg=False, 
                                Tf=args.Tf,
                                relax_only_cg=False,
                                user_mpi_buf=args.lp_user_mpi_buf).to(device)
        # modelnet = false， faust = false


        # Detailed XBraid timings are output to these files for the forward and backward phases
        model.parallel_nn.fwd_app.setTimerFile(
            f'b_fwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')
        model.parallel_nn.bwd_app.setTimerFile(
            f'b_bwd_s_{args.steps}_c_{args.channels}_bs_{args.batch_size}_p_{procs}')
        
        # Declare optimizer  
        # KN1 = nn.Parameter(torch.rand(nlayer, Nfeatures(n_channel), nhid(n_channel)) * stdvp)
        # KN2 =  nn.Parameter(torch.rand(nlayer, nhid(n_channel), 1 * nhid(n_channel)) * stdvp)
        # K1Nopen = nn.Parameter(torch.randn(nopen(n_channel), nNin) * stdv)
        # KNclose = nn.Parameter(torch.randn(num_output, nopen) * stdv)  #num_output=dataset.num_classes

        # control the optimization behavior for different groups 
        # of parameters within your model, allowing you to apply 
        # different learning rates or weight decays as needed.
        if (rank==0):
            optimizer = torch.optim.Adam([
            dict(params=model.parallel_nn.parameters(), lr=lrGCN, weight_decay=0),
            # model.KN1 parameters with learning rate lrGCN and no weight decay.
            dict(params=model.open_nn.parameters(), weight_decay=wd),
            # model.K1Nopen parameters with weight decay wd and default learning rate from the optimizer.
            dict(params=model.close_nn.parameters(), weight_decay=wd),
            # model.KNclose parameters with weight decay wd and default learning rate from the optimizer.     
            #   dict(params=model.alpha, lr=lr_alpha, weight_decay=0) appears on fully-supervised
            ], lr=lr)
        else:
            optimizer = torch.optim.Adam([
            dict(params=model.parallel_nn.parameters(), lr=lrGCN, weight_decay=0),
            # model.KN1 parameters with learning rate lrGCN and no weight decay.
            ], lr=lr)            
        # lr=lr is the base learning rate for the optimizer, 
        # which applies to any parameter groups that do not 
        # explicitly specify their own learning rate.
        for epoch in range(1, args.epochs + 1):
            start_time = timer()
            [losses, train_times] = train(rank=rank, params=args, model=model, data=data, optimizer=optimizer, epoch=epoch,
                compose=model.compose, device=device)
            epoch_times[str(nl)] += [timer() - start_time]
            root_print(rank,f'num_layers: {nl} epoch: {epoch} -  {"{:.3f}".format(losses)}')
            epoch_losses[str(nl)] += [losses]
            accs=test(model=model, data=data, device=device)
            epoch_accuracies[str(nl)] += [accs[2]] 
            root_print(rank, accs)
        # Print out Braid internal timings, if desired
        #timer_str = model.parallel_nn.getTimersString()
        #root_print(rank, timer_str)

        if args.serial_file is not None:
            # Model can be reloaded in serial format with: model = torch.load(filename)
            model.saveSerialNet(args.serial_file)

        # Plot the loss, validation 
        root_print(rank, f'\nMin epoch time:   {"{:.3f}".format(np.min(epoch_times[str(nl)]))} ')
        root_print(rank, f'Mean epoch time:  {"{:.3f}".format(stats.mean(epoch_times[str(nl)]))} ')
    if rank == 0:
        plot_save_metric_data(epoch_accuracies,"Accuracies-wrt Layers_ num_proc"+str(procs)+" num_levels "+str(args.lp_max_levels))
        plot_save_metric_data(epoch_times,"Times-wrt Layers_ num_proc"+str(procs)+" num_levels "+str(args.lp_max_levels))
        plot_save_metric_data(epoch_losses,"Loss-wrt Layers_ num_proc"+str(procs)+" num_levels "+str(args.lp_max_levels))
    root_print(rank,epoch_accuracies)
    root_print(rank,epoch_times)


if __name__ == '__main__':
    main()
