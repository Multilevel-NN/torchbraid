import math

class EmptyLPIterScheduler():
  def __init__(self):
    pass

  def reset(self):
    pass

  def step(self):
    pass

class IterLPIterScheduler():
  def __init__(self,model,data):
    data = data.split()
    assert len(data)>0
    assert len(data)<4

    self.count = 1
    self.parallel_nn = model.parallel_nn
    self.fwd_iters = self.parallel_nn.getFwdMaxIters()
    self.bwd_iters = self.parallel_nn.getBwdMaxIters()

    self.acc_fwd_iters = self.fwd_iters
    self.acc_bwd_iters = self.bwd_iters

    if len(data)>2:
      self.acc_bwd_iters = int(data[2])
    elif len(data)>1:
      self.acc_fwd_iters = int(data[1])

    self.itrs = math.ceil(1./float(data[0]))

  def reset(self):
    self.count = 1
    self.parallel_nn.setFwdMaxIters(self.fwd_iters)
    self.parallel_nn.setBwdMaxIters(self.bwd_iters)

  def _set_accurate(self):
    self.parallel_nn.setFwdMaxIters(self.acc_fwd_iters)
    self.parallel_nn.setBwdMaxIters(self.acc_bwd_iters)

  def step(self):
    if self.count==0:
      self.reset()
    if self.count % self.itrs == 0:
      self._set_accurate()
      self.count = 0
    else: 
      self.count += 1

# this doesn't work
######################
#
# class SerialLPIterScheduler():
#   def __init__(self,model,frequency):
#     self.count = 1
#     self.itrs = math.ceil(1./frequency)
#     print('self.itrs ',frequency,self.itrs)
#     self.parallel_nn = model.parallel_nn
#     self.fwd_iters = self.parallel_nn.getFwdMaxIters()
#     self.bwd_iters = self.parallel_nn.getBwdMaxIters()
#     self.fwd_levels = self.parallel_nn.getFwdMaxLevels()
#     self.bwd_levels = self.parallel_nn.getBwdMaxLevels()
# 
#   def reset(self):
#     print('    -- SET PARALLEL --')
#     self.count = 1
#     self.parallel_nn.setFwdMaxIters(self.fwd_iters)
#     self.parallel_nn.setBwdMaxIters(self.bwd_iters)
#     self.parallel_nn.setFwdMaxLevels(self.fwd_levels)
#     self.parallel_nn.setBwdMaxLevels(self.bwd_levels)
# 
#   def _set_serial(self):
#     print('    -- SET SERIAL --')
#     self.parallel_nn.setFwdMaxIters(1)
#     self.parallel_nn.setBwdMaxIters(1)
#     self.parallel_nn.setFwdMaxLevels(1)
#     self.parallel_nn.setBwdMaxLevels(1)
# 
#   def step(self):
#     print('step = ',self.count)
#     if self.count==0:
#       self.reset()
#     if self.count % self.itrs == 0:
#       self._set_serial()
#       self.count = 0
#     else: 
#       self.count += 1
