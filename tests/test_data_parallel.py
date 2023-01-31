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

import torchbraid.utils.data_parallel
import unittest

res = {}
res[(1, 2)] = [[3, 16, 6, 10, 2, 14, 4, 17, 7, 1, 13, 0, 19, 18, 9, 15, 8, 12, 11, 5]]
res[(2, 2)] = [[3, 16, 2, 14, 7, 1, 19, 18, 8, 12], [6, 10, 4, 17, 13, 0, 9, 15, 11, 5]]
res[(3, 2)] = [[3, 16, 4, 17, 19, 18, 11, 5], [6, 10, 7, 1, 9, 15], [2, 14, 13, 0, 8, 12]]

res[(1, 5)] = [[3, 16, 6, 10, 2, 14, 4, 17, 7, 1, 13, 0, 19, 18, 9, 15, 8, 12, 11, 5]]
res[(2, 5)] = [[3, 16, 6, 10, 2, 13, 0, 19, 18, 9], [14, 4, 17, 7, 1, 15, 8, 12, 11, 5]]
res[(3, 5)] = [[3, 16, 6, 10, 2, 15, 8, 12, 11, 5], [14, 4, 17, 7, 1], [13, 0, 19, 18, 9]]

class TestDataParallel(unittest.TestCase):

  def test_splitting(self):
    data = [[i] for i in range(20)]
    for procs in range(1, 4):
      for batch_size in [2, 5]:
        train_partition = torchbraid.utils.data_parallel.Partioner(data=data, procs=procs, seed=1, batch_size=batch_size)
        for rank in range(procs):
          self.assertListEqual(train_partition.partitions[rank], res[(procs,batch_size)][rank])

if __name__ == '__main__':
  unittest.main()



