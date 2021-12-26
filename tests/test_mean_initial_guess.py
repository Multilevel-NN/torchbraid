#@HEADER
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
#@HEADER

import torch
import math
import unittest
import numpy as np

import torchbraid
from torchbraid.utils import MeanInitialGuessStorage

import test_cbs as cbs

class TestMeanInitialGuess(unittest.TestCase):
  def test_init(self):
    class_count = 4
    average_weight = 0.1

    ig_storage = MeanInitialGuessStorage(class_count,average_weight) 

    self.assertTrue(ig_storage!=None)
    self.assertTrue(ig_storage.class_count!=None)
    self.assertTrue(ig_storage.average_weight!=None)
    self.assertTrue(ig_storage.state_map!=None)

    ig_storage = MeanInitialGuessStorage(class_count,0.0) 
    ig_storage = MeanInitialGuessStorage(class_count,1.0) 

    with self.assertRaises(AssertionError) as ctx:
      ig_storage = MeanInitialGuessStorage(-class_count,average_weight) 
    with self.assertRaises(AssertionError) as ctx:
      ig_storage = MeanInitialGuessStorage(0,average_weight) 
    with self.assertRaises(AssertionError) as ctx:
      ig_storage = MeanInitialGuessStorage(class_count,-0.1) 
    with self.assertRaises(AssertionError) as ctx:
      ig_storage = MeanInitialGuessStorage(class_count,1.1) 

  def test__initialize(self):
    class_count = 4
    average_weight = 0.1
    batch_size = 10

    sizes = lambda batch_size: [(batch_size,4,9),
                                (batch_size,3,7),
                                (batch_size,9,2,2)]

    # FIRST BLOCK OF TESTS: Look at shape of output tensor

    # test the single tensor case
    ########################################################
    size = sizes(batch_size)[0]

    classes = torch.randint(0,class_count, (batch_size,))
    state   = torch.zeros(size)

    ig_storage = MeanInitialGuessStorage(class_count,average_weight) 
    result = ig_storage._initialize(state,classes,average_weight)

    self.assertTrue(len(result)==1)
    
    self.assertTrue(sizes(class_count)[0]==result[0].shape)
    self.assertTrue(torch.norm(result[0]).item()==0.0)

    # test the multi-tensor case
    ########################################################
    classes = torch.randint(0,class_count, (batch_size,))
    state   = tuple([torch.zeros(s) for s in sizes(batch_size)])

    ig_storage = MeanInitialGuessStorage(class_count,average_weight) 
    result = ig_storage._initialize(state,classes,average_weight)

    self.assertTrue(len(result)==len(state))
    
    for s,nstate,ostate in zip(sizes(class_count),result,state):
      self.assertTrue(s==nstate.shape)
      self.assertTrue(torch.norm(nstate).item()==0.0)

  def test__average(self):
    class_count = 2
    average_weight = 0.9
    batch_size = 5

    sizes = lambda batch_size: [(batch_size,2,2),
                                (batch_size,7,7)]

    batch_sizes = sizes(batch_size)
    class_sizes = sizes(class_count)

    classes = [0,1,0,1,1]
    state   = tuple([torch.zeros(s) for s in batch_sizes])

    ig_storage = MeanInitialGuessStorage(class_count,average_weight) 
    initial = ig_storage._initialize(state,classes,average_weight)

    # put in a nontrivial initial guess
    for i in initial:
      i[:] = 7.0

    # put in a non-trivial starting state
    for s in state:
      s[0],s[1],s[2],s[3],s[4] = 1.,2.,3.,4.,5.

    ig_storage._average(initial,state,classes,average_weight)

    # test the average
    self.assertTrue(initial[0].shape[0]==class_count)
    for avg in initial:
      for v in avg[0].flatten(): 
        self.assertAlmostEqual(v.item(),(1.0-average_weight)*7.+average_weight*((1.+3.)/2.))
      for v in avg[1].flatten(): 
        self.assertAlmostEqual(v.item(),(1.0-average_weight)*7.+average_weight*((2.+4.+5.)/3.))

    # Check for modifying only one class
    #########################################################

    classes = [1,1,1,1,1]
    state   = tuple([torch.zeros(s) for s in batch_sizes])

    ig_storage = MeanInitialGuessStorage(class_count,average_weight) 
    initial = ig_storage._initialize(state,classes,average_weight)

    # put in a nontrivial initial guess
    for i in initial:
      i[:] = 7.0

    # put in a non-trivial starting state
    for s in state:
      s[0],s[1],s[2],s[3],s[4] = 1.,2.,3.,4.,5.

    ig_storage._average(initial,state,classes,average_weight)

    # test the average
    self.assertTrue(initial[0].shape[0]==class_count)
    for avg in initial:
      for v in avg[0].flatten(): 
        self.assertAlmostEqual(v.item(),7.0)
      for v in avg[1].flatten(): 
        self.assertAlmostEqual(v.item(),(1.0-average_weight)*7.+average_weight*((1.+2.+3.+4.+5.)/5.),places=6)

  def test_add_get_state(self):
    class_count = 2
    average_weight = 0.9
    batch_size = 5

    def check_state(state,val,ind):
      for s in state:
        for v in s[ind].flatten():
          self.assertAlmostEqual(v.item(),val,places=6)

    mult = lambda m,state: tuple([m*s for s in state])
    sizes = lambda batch_size: [(batch_size,2,2),
                                (batch_size,7,7)]

    batch_sizes = sizes(batch_size)
    class_sizes = sizes(class_count)

    classes = [0,1,0,1,1]
    state   = tuple([torch.zeros(s) for s in batch_sizes])

    # put in a non-trivial starting state
    for s in state:
      s[0],s[1],s[2],s[3],s[4] = 1.,2.,3.,4.,5.

    class_0_ind = [0,2]
    class_1_ind = [1,3,4]

    class_0_val = ((1.+3.)/2.)
    class_1_val = ((2.+4.+5.)/3.)

    ig_storage = MeanInitialGuessStorage(class_count,average_weight) 

    # check add state on the initial pass
    ##########################################

    ig_storage.addState(0.1,mult(0.1,state),classes)
    ig_storage.addState(0.9,mult(0.9,state),classes)

    class_0 = 0.1*class_0_val*average_weight 
    class_1 = 0.1*class_1_val*average_weight 

    check_state(ig_storage.getState(0.1,classes),class_0,class_0_ind)
    check_state(ig_storage.getState(0.1,classes),class_1,class_1_ind)

    class_0 = 0.9*class_0_val*average_weight 
    class_1 = 0.9*class_1_val*average_weight 

    check_state(ig_storage.getState(0.9,classes),class_0,class_0_ind)
    check_state(ig_storage.getState(0.9,classes),class_1,class_1_ind)

    # check an add state after the time stamp has been initialized
    ##########################################

    ig_storage.addState(0.1,mult(0.2,state),classes)
    ig_storage.addState(0.9,mult(-0.3,state),classes)

    class_0 = 0.1*class_0_val*average_weight*(1.0-average_weight)+0.2*class_0_val*average_weight
    class_1 = 0.1*class_1_val*average_weight*(1.0-average_weight)+0.2*class_1_val*average_weight

    check_state(ig_storage.getState(0.1,classes),class_0,class_0_ind)
    check_state(ig_storage.getState(0.1,classes),class_1,class_1_ind)

    class_0 = 0.9*class_0_val*average_weight*(1.0-average_weight)-0.3*class_0_val*average_weight
    class_1 = 0.9*class_1_val*average_weight*(1.0-average_weight)-0.3*class_1_val*average_weight

    check_state(ig_storage.getState(0.9,classes),class_0,class_0_ind)
    check_state(ig_storage.getState(0.9,classes),class_1,class_1_ind)

    # check the time stamps
    ##########################################

    stamps = ig_storage.getTimeStamps()
    self.assertEqual(len(stamps),2)
    self.assertEqual(sorted(stamps),[0.1,0.9])
    
if __name__ == '__main__':
  unittest.main()
