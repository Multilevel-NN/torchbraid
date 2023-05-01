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

class MeanInitialGuessStorage:
  """
  A class that stores the average state value at each time step 
  for different classes.
  """
  class InitialGuess:
    """
    Utility class to go into the neural network to set the intial
    guess for the state.
    """

    def __init__(self,parent,classes):
      self.parent = parent
      self.classes = classes

    def getState(self,t):
      return self.parent.getState(t,self.classes)

  @torch.no_grad()
  def __init__(self,class_count,average_weight):
    """
    Constructor for the mean value storage class.
    
      Parameters: 
        class_count (int: >0): How many classes will be used
        average_weight (float: [0,1]): Weighting for the average operation. The result is
                                       (1-average_weight)*old_state + average_weight*new_state
    """

    assert(0 < class_count)
    assert(0.0 <= average_weight <= 1.0)

    self.class_count = class_count
    self.average_weight = average_weight
    self.state_map = dict()

  def initialGuess(self,classes):
    """
    Create a convenience class that changes the initial guess.
    """
    return MeanInitialGuessStorage.InitialGuess(self,classes)

  def getTimeStamps(self):
    return self.state_map.keys()

  @torch.no_grad()
  def getState(self,t,classes):
    """
    Get the state at time t, associated with a list of classes.
    """
    state = []
    for s in self.state_map[t]:
      state += [s[classes].clone()]
    return tuple(state)

  @torch.no_grad()
  def addState(self,t,state,classes):
    """
    Incoporate, through averaging, the state at time t into the 
    average of states. 
    """

    if t in self.state_map:
      self._average(self.state_map[t],state,classes,self.average_weight) 
    else:
      self.state_map[t] = self._initialize(state,classes,self.average_weight) 

  @torch.no_grad()
  def _ensure_tuple(self,state):
    """ 
    Ensure argument is a tuple of Tensors or a single Tensor.
    
    Return:
      A tuple of tensors
    """

    if not isinstance(state,tuple):
      state = (state,)

    for s in state: 
      assert(isinstance(s,torch.Tensor))

    return state

  @torch.no_grad()
  def _initialize(self,new_state,classes,average_weight):
    """
    Compute an initial set of mean states, based on
    a passed in state
    """

    state = self._ensure_tuple(new_state)

    # create a bunch of zeros, with the batch size replacing the classes
    zeros = len(state)*[None]
    for i,s in enumerate(state):
      dtype = s.dtype
      shape = tuple([self.class_count]+list(s.shape[1:]))
      device = s.device
      zeros[i] = torch.zeros(shape, dtype=dtype, device=s.device)

    zeros = tuple(zeros)

    # average (in place) the current state into the newly created zeros
    self._average(zeros,state,classes,average_weight)

    return zeros
 
  @torch.no_grad()
  def _average(self,old_mean,new_state,classes,average_weight):
    """
    Average in a new set of states
    """

    if isinstance(classes,torch.Tensor):
      classes = classes.tolist()

    state = self._ensure_tuple(new_state)

    u_classes = list(set(classes))
    for cls in u_classes:
      # find all indices that match
      indices = [ind for ind,c in enumerate(classes) if c==cls]
 
      # perform the weighted average
      for o,s in zip(old_mean,state):
        o[cls] *= (1.0-average_weight)
        o[cls] += average_weight*torch.mean(s[indices],dim=0)
