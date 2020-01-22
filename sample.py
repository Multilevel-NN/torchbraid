import torch
import torch.nn as nn
import torch.nn.functional as F

import torchbraid

class BasicBlock(nn.Module):
  def __init__(self,dim):
    super(BasicBlock, self).__init__()
    self.lin = nn.Linear(dim, dim)

  def __del__(self):
    print('DEL: BasicBlock')

  def forward(self, x):
    return F.relu(self.lin(x))
# end layer

class ODEBlock(nn.Module):
  def __init__(self,layer,dt):
    super(ODEBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def __del__(self):
    print('DEL: ODEBlock')

  def forward(self, x):
    return x + self.dt*self.layer(x)
  

dim = 10
basic_block = lambda: BasicBlock(dim)

Tf = 2.0
num_steps = 10
m = torchbraid.Model(basic_block,num_steps,Tf,max_levels=2,max_iters=2)

x = torch.randn(5,10) 

dt = Tf/num_steps
ode_layers = [ODEBlock(l,dt) for l in m.local_layers.children()]
f = torch.nn.Sequential(*ode_layers)

ym = m(x)
yf = f(x)

val = torch.norm(ym-yf).item()
print('error = %.6e' % val)

print(' ------- end ------- ')
