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

from timeit import default_timer as timer

class ContextTimer:
  def __init__(self,name):
    self.name   = name
    self.times  = []
    self.timing = False 

  def __enter__(self):
    self.timing = True
    self.start_time = timer()
    return self

  def __exit__(self,except_type,except_value,except_traceback):
    self.end_time = timer()
    self.timing = False

    self.times += [ self.end_time-self.start_time ] 
    return except_type==None

  def getName(self):
    return self.name

  def isTiming(self):
    return self.timing

  def getTimes(self):
    return self.times
