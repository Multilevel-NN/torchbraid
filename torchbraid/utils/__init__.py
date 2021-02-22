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

from .context_timer import ContextTimer
from .context_timer_manager import ContextTimerManager

# import some useful helper functions
from .functional import l2_reg
from .gittools import git_rev 

# import bufpackunpack tools
from .bufpackunpack import buffer_size, pack_buffer, unpack_buffer

import gc
import torch
import traceback
import pickle

def seed_from_rank(seed,rank):
  """
  Helper function to compute a new seed from the parallel rank using an LCG

  Note that this is not a good parallel number generator, just a starting point.
  """
  # set the seed (using a LCG: from Wikipedia article, apparently Numerical recipes)
  return (1664525*(seed+rank) + 1013904113)% 2**32
# end seed_from_rank

def tensor_memory(prefix,min_size=0,total_only=False):
  """
  Helper function to print the memory footprint of all the torch tensors.

  This will print the memory usage of all tensors above a particular size.
  Setting the total_only=True will only print a summary
  """

  objects = gc.get_objects()
  tqueue = [o for o in objects if isinstance(o,torch.Tensor)]
  s = ''
  total_size_printed = 0 
  total_size = 0 
  total_count = 0
  for t in tqueue:
    numel = t.numel()
    esz = t.element_size()
    total_size += numel*esz
    if numel*esz>min_size:
      total_size_printed += numel*esz
      total_count += 1
      if not total_only:
        if hasattr(t,'label'):
          s += '  {}) TENSOR {:.2f} MiB: {} label: {}\n'.format(prefix,esz*numel/1024/1024,t.shape,t.label)
        else:
          s += '  {}) TENSOR {:.2f} MiB: {} none\n'.format(prefix,esz*numel/1024/1024,t.shape)

  s += '  {}) TENSOR Total/Above Bound ({}) = {:.2f} MiB/{:.2f} MiB'.format(prefix,total_count,total_size/2**20,total_size_printed/2**20)
  if not total_only:
    s += '\n'
  print(s)
# end print_tensors

def stack_string(prefix=None):
  stack = traceback.format_stack()
  lines = []
  for l in stack:
    lines += l.splitlines()

  if prefix==None:
    prefix = ''

  stack_str = ('\n{}').format(prefix).join(lines)
  stack_str = prefix+stack_str

  return stack_str

def pickle_size(obj):
  """
  What is the size in bytes of the pickled stream from
  this object.
  """
  return len(pickle.dumps(obj))

# def getMaxMemory(comm,message):
#   usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# 
#   total_usage = comm.reduce(usage,op=MPI.SUM)
#   min_usage = comm.reduce(usage,op=MPI.MIN)
#   max_usage = comm.reduce(usage,op=MPI.MAX)
#   if comm.Get_rank()==0:
#     result = '%.2f MiB, (min,avg,max)=(%.2f,%.2f,%.2f)' % (total_usage/2**20, min_usage/2**20,total_usage/comm.Get_size()/2**20,max_usage/2**20)
#     print(message.format(result))
# 
# def getLocalMemory(comm,message):
#   usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# 
#   result = '%.2f MiB' % (usage/2**20)
#   print(('{}) ' + message).format(comm.Get_rank(),result))
