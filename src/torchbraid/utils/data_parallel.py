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

import numpy as np

from mpi4py import MPI


def split_communicator(comm: MPI.Comm, splitting: int):
  """
  Creates new communicators for data parallelism & layer parallelism by
  "splitting" the input communicator into two sub-communicators.
  :param comm: Communicator to be used as the basis for new communicators
  :param splitting: Splitting factor (number of processes for spatial parallelism)
  :return: Space and time communicator
  """

  # Determine color based on splitting factor
  # All processes with the same color will be assigned to the same communicator.
  rank = comm.Get_rank()
  x_color = rank // splitting
  t_color = rank % splitting

  # Split the communicator based on the color and key
  comm_dp = comm.Split(color=x_color, key=rank)
  comm_lp = comm.Split(color=t_color, key=rank)
  return comm_dp, comm_lp


def average_gradients(model, comm_dp):
  """
  Averages gradients for comm_dp
  """
  for param in model.parameters():
    comm_dp.Allreduce(MPI.IN_PLACE, param.grad.data, op=MPI.SUM)
    param.grad.data /= float(comm_dp.Get_size())


class Partition(object):
  def __init__(self, data, index):
    self.data = data
    self.index = index

  def __len__(self):
    return len(self.index)

  def __getitem__(self, index):
    data_idx = self.index[index]
    return self.data[data_idx]


class Partioner(object):
  def __init__(self, data, procs, seed, batch_size):
    def partion(iterable, start, select, skip):
      return [x for i, x in enumerate(iterable[start:]) if i % (select + skip) < select]

    self.data = data
    indices = np.arange(0, len(self.data))
    np.random.seed(seed)
    np.random.shuffle(indices)
    self.partitions = [partion(indices,
                               start=batch_size * i,
                               select=batch_size,
                               skip=batch_size * (procs - 1)) for i in range(procs)]

  def get_partion(self, rank):
    return Partition(self.data, self.partitions[rank])
