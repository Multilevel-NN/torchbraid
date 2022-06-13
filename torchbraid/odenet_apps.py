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

import torch
import torch.nn as nn

from math import pi
from braid_vector import BraidVector
from torchbraid_app import BraidApp
import utils

import sys
import traceback
import resource
import copy

from bisect import bisect_right
from bsplines import BsplineBasis
from mpi4py import MPI


class ForwardODENetApp(BraidApp):

    def __init__(self, comm, layer_models, local_num_steps, Tf, max_levels, max_iters, timer_manager, spatial_ref_pair=None, layer_block=None, sc_levels=None):

        """
        # note that a simple equals would result in a shallow copy...bad!
    def __init__(self, comm, layer_models, local_num_steps, Tf, max_levels, max_iters, timer_manager, spatial_ref_pair=None, layer_block=None, sc_levels=None):
        # build up the core
        self.py_core = self.initCore()
        BraidApp.__init__(self, 'FWDApp', comm, local_num_steps, Tf, max_levels,
                          max_iters, spatial_ref_pair=spatial_ref_pair, require_storage=True)
        self.timer_manager = timer_manager
        # note that a simple equals would result in a shallow copy...bad!
        self.layer_models = [l for l in layer_models]

        comm = self.getMPIComm()
        my_rank = self.getMPIComm().Get_rank()
        num_ranks = self.getMPIComm().Get_size()
        self.clearTempLayerWeights()
        self.layer_block = layer_block
    # end __init__
        # need access to this in order to coarsen state vectors up from the fine grid
        # for getPrimalWithGrad
        if spatial_ref_pair is not None:
            self.spatial_coarsen = spatial_ref_pair[0]
            self.sc_levels = sc_levels
        else:
            self.spatial_coarsen = None
            self.sc_levels = None

        # add a sentinal at the end
        self.layer_models.append(None)

        # build up the core
        self.py_core = self.initCore()
                dest_p.data = src_w
    # end setLayerWeights

    def initializeVector(self, t, x):
        self.setVectorWeights(t, 0.0, 0, x)
        for p in layer_models[0].parameters():
    def updateParallelWeights(self):
        # send everything to the left (this helps with the adjoint method)
        self.temp_layer = layer_block()
        self.clearTempLayerWeights()
            # reset derivative papth
            self.use_deriv = False

        if y is not None:
            return y[0]
    def getTensorShapes(self):
        return list(self.shape0)+self.parameter_shapes
        if index < 0:
    def setVectorWeights(self, t, tf, level, x):
        layer = self.getLayer(t, tf, level)
        if layer != None:
            weights = [p.data for p in layer.parameters()]
            # q = dt * layer(t_x)                 # default
            weights = []
        x.addWeightTensors(weights)
            del q
    def clearTempLayerWeights(self):
        layer = self.temp_layer

        for dest_p in list(layer.parameters()):
            dest_p.data = torch.empty(())
    # end setLayerWeights
        #  2. x is a torch tensor: called internally (probably for the adjoint)
    def setLayerWeights(self, t, tf, level, weights):
        layer = self.getLayer(t, tf, level)
        if isinstance(y, BraidVector):
            t_y = y.tensor().detach()
            for dest_p, src_w in zip(list(layer.parameters()), weights):
                dest_p.data = src_w
    # end setLayerWeights
            with torch.enable_grad():
    def initializeVector(self, t, x):
        self.setVectorWeights(t, 0.0, 0, x)
        time step and also get its derivative. This is
    def updateParallelWeights(self):
        # send everything to the left (this helps with the adjoint method)
        comm = self.getMPIComm()
        my_rank = self.getMPIComm().Get_rank()
        num_ranks = self.getMPIComm().Get_size()
        being recomputed.
        if my_rank > 0:
            comm.send(
                list(self.layer_models[0].parameters()), dest=my_rank-1, tag=22)
        if my_rank < num_ranks-1:
            neighbor_model = comm.recv(source=my_rank+1, tag=22)
            new_model = self.layer_block()
            b_x = self.getUVector(0, tstart)
                for dest_p, src_w in zip(list(new_model.parameters()), neighbor_model):

            self.layer_models[-1] = new_model
            self.setLayerWeights(tstart, tstop, level, b_x.weightTensors())
    def run(self, x):
            x = t_x.detach()
            y = t_x.detach().clone()
            self.eval(y, tstart, tstop, 0, done=0, x=x)
        except:
            sys.stdout.flush()
            traceback.print_exc()
            # do boundary exchange for parallel weights
            if self.use_deriv:
                self.updateParallelWeights()

            sys.stdout.flush()

        return (y, x), layer
    # end getPrimalWithGrad

# end ForwardODENetApp

##############################################################


class BackwardODENetApp(BraidApp):
    def timer(self, name):
    def __init__(self, fwd_app, timer_manager):
        # call parent constructor
    def getLayer(self, t, tf, level):
        index = self.getLocalTimeStepIndex(t, tf, level)
        if index < 0:
            #pre_str = "\n{}: WARNING: getLayer index negative at {}: {}\n".format(self.my_rank,t,index)
            #stack_str = utils.stack_string('{}: |- '.format(self.my_rank))
            # print(pre_str+stack_str)
            return self.temp_layer

        return self.layer_models[index]

        BraidApp.__init__(self, 'BWDApp',
                          fwd_app.getMPIComm(),
                          fwd_app.local_num_steps,
            if l != None:
                          fwd_app.max_levels,
                          fwd_app.max_iters,
                          spatial_ref_pair=fwd_app.spatial_ref_pair)

    def eval(self, y, tstart, tstop, level, done, x=None):

        # build up the core
        self.py_core = self.initCore()

        # reverse ordering for adjoint/backprop
        self.setRevertedRanks(1)
        # this function is used twice below to define an in place evaluation
        def in_place_eval(t_y, tstart, tstop, level, t_x=None):
            # get some information about what to do
            dt = tstop-tstart
            layer = self.getLayer(tstart, tstop, level)  # resnet "basic block"
            dx = pi/(t_y.size()[-2] + 1)
            dy = pi/(t_y.size()[-1] + 1)

            # print(self.my_rank, ": FWDeval level ", level, " ", tstart, "->", tstop, " using layer ", layer.getID(), ": ", layer.linearlayer.weight[0].data)
            if t_x == None:
                t_x = t_y
            else:
                t_y.copy_(t_x)

            q = dt/(dx*dy) * layer(t_x)           # scale by cfl number
            t_y.add_(q)

            del q
        # end in_place_eval

        # there are two paths by which eval is called:
        #  1. x is a BraidVector: my step has called this method
        #  2. x is a torch tensor: called internally (probably for the adjoint)

        if isinstance(y, BraidVector):
            self.setLayerWeights(tstart, tstop, level, y.weightTensors())
        self.finalRelax()

        self.timer_manager = timer_manager
    # end __init__
    def __del__(self):
                in_place_eval(t_y, tstart, tstop, level)

            if y.getSendFlag():
                self.clearTempLayerWeights()

                y.releaseWeightTensors()
                y.setSendFlag(False)
            # wipe out any sent information

            self.setVectorWeights(tstop, 0.0, level, y)
    def getTensorShapes(self):
        else:
            x.requires_grad = True
            with torch.enable_grad():
                in_place_eval(y, tstart, tstop, level, t_x=x)
    def timer(self, name):
        return self.timer_manager.timer("BckWD::"+name)
    def getPrimalWithGrad(self, tstart, tstop, level):
    def run(self, x):

        try:
            f = self.runBraid(x)
            if f is not None:
        Its intent is to abstract the forward solution
        so it can be stored internally instead of
        being recomputed.
                f = f[0]
        try:
            layer = self.getLayer(tstart, tstop, level)

            # get state from fine-grid, then coarsen appropriately
            b_x = self.getUVector(0, tstart)
            t_x = b_x.tensor()

            if self.spatial_coarsen:
                for l in range(level):
                    t_x = self.spatial_coarsen(t_x, l)
            # The ownership of the time steps is shifted to the left (and no longer balanced)
            self.setLayerWeights(tstart, tstop, level, b_x.weightTensors())
            my_params = self.fwd_app.parameters()
                sub_gradlist = []
                for item in sublist:
                    if item.grad is not None:
            x.requires_grad = t_x.requires_grad

            self.eval(y, tstart, tstop, 0, done=0, x=x)
        except:
            print(f'\n*** {tstart}, {tstop}, {level} ***\n')
            sys.stdout.flush()
            traceback.print_exc()
            sys.stdout.flush()

                self.grads += [sub_gradlist]
            # end for sublist

            for l in self.fwd_app.layer_models:
                if l == None:
                    continue
                l.zero_grad()

        except:
            print('\n**** Torchbraid Internal Exception ****\n')
    def __init__(self, fwd_app, timer_manager):

        BraidApp.__init__(self, 'BWDApp',
    def eval(self, w, tstart, tstop, level, done):
                          fwd_app.local_num_steps,
        Evaluate the adjoint problem for a single time step. Here 'w' is the
                          fwd_app.max_levels,
        problem solutions at the beginning (x) and end (y) of the type step.
        """
        try:
            # we need to adjust the time step values to reverse with the adjoint
            # this is so that the renumbering used by the backward problem is properly adjusted
            (t_y, t_x), layer = self.fwd_app.getPrimalWithGrad(
        self.py_core = self.initCore()
            # t_x should have no gradient (for memory reasons)
            assert(t_x.grad is None)

            # we are going to change the required gradient, make sure they return
            # to where they started!
            required_grad_state = []

            # play with the layers gradient to make sure they are on appropriately
            for p in layer.parameters():
                required_grad_state += [p.requires_grad]
                if done == 1:
                    if not p.grad is None:
                        p.grad.data.zero_()
                else:
                    # if you are not on the fine level, compute no parameter gradients
                    p.requires_grad = False
    def timer(self, name):
            # perform adjoint computation
            t_w = w.tensor()
    def run(self, x):
            # stored too long in this calculation (in particulcar setting
            # the grad to None after saving it and returning it to braid)
            for p, s in zip(layer.parameters(), required_grad_state):
        except:
            print('\n**** Torchbraid Internal Exception ****\n')
            traceback.print_exc()
            # this code is due to how braid decomposes the backwards problem
            # The ownership of the time steps is shifted to the left (and no longer balanced)
            first = 1
            if self.getMPIComm().Get_rank() == 0:
                first = 0
            for sublist in my_params[first:]:
                        sub_gradlist += [item.grad.clone()]
                        sub_gradlist += [None]
                self.grads += [sub_gradlist]
                if l == None:
                    continue
    def eval(self, w, tstart, tstop, level, done):
            (t_y, t_x), layer = self.fwd_app.getPrimalWithGrad(
                self.Tf-tstop, self.Tf-tstart, level)
            # play with the layers gradient to make sure they are on appropriately
                if done == 1:
                    # if you are not on the fine level, compute no parameter gradients
            # perform adjoint computation
            t_w.copy_(t_x.grad.detach())
            for p, s in zip(layer.parameters(), required_grad_state):
            print('\n**** Torchbraid Internal Exception ****\n')
