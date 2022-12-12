import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

steps     = 128
dim       = 10
decay     = 0.5

A_weight = np.random.randn(dim,dim)
B_weight = np.random.randn(dim,dim)
b_bias   = np.random.randn(dim)

def rnn_step(h_nminus1,d_n,dt):
  temp = A_weight.dot(h_nminus1)+B_weight.dot(d_n)+b_bias
  return (h_nminus1 +dt*1./(1.+np.exp(-temp)))/(1.0+decay*dt)

def forward_prop(data,dt,correct=None):
  if correct is None:
    correct = np.zeros(data.shape)
  hall = np.zeros(data.shape)
  h = hall[0,:]
  for step in range(data.shape[0]-1):
    dstep = data[step+1,:]
    h = rnn_step(h,dstep,dt) + correct[step+1,:]
    hall[step+1,:] = h
  return hall

def residual(hinput,data,dt):
  steps = data.shape[0]-1
  res = np.zeros((steps+1,dim))
  for step in range(steps):
    dstep = data[step+1,:]
    res[step+1,:] = hinput[step+1]-rnn_step(hinput[step],dstep,dt)
  return res

def relax(hinput,data,dt,cf,relax_type):
  relax_type = 1 if relax_type=='f' else 0
  houtput = hinput.copy()
  for c in range(int((data.shape[0]-1)/cf)):
    h = hinput[c*cf,:]
    for substep in range(cf-relax_type):
      step = c*cf+substep
      h = rnn_step(h,data[step+1,:],dt)
      houtput[step+1,:] = h
  return houtput

def restrict(h,cf):
  return h[::cf]

def prolong(h_c,cf):
  steps_c = h_c.shape[0]-1
  steps_f = cf*steps_c
  h = np.zeros((steps_f+1,dim))
  for c in range(steps_c):
    for substep in range(cf):
      step = c*cf+substep
      h[step,:] = h_c[c]
    
  return h
  
def two_level_mg(h,data,dt,cf,relax_steps):
  h_prime = h.copy()
  for _ in range(relax_steps):
    h_prime = relax(h_prime,data,1.0,cf,relax_type='fc')
    h_prime = relax(h_prime,data,1.0,cf,relax_type='f')
  
  res_c     = restrict(residual(h_prime,data,dt),cf)
  h_c_prime = restrict(h_prime,cf)

  data_c    = restrict(data,cf)
  correct_c = residual(h_c_prime,data_c,dt*cf)-res_c

  # coarse correction
  h_c_star = forward_prop(data_c,cf*dt,correct_c)

  h_pprime = h_prime+prolong(h_c_star-h_c_prime,cf)
    
  return relax(h_pprime,data,1.0,cf,relax_type='f')

cf = 4
mg_iters = 8
iters = 2
dt = 1.0

timenodes = np.linspace(0.0,steps,steps+1)
data = np.sin(3.0*np.pi*np.outer(timenodes,np.random.randn(dim))/steps)
data[0,:] = 0.0

exact = forward_prop(data,dt)

print('FCF Relaxation')
hrelax_fcf = np.zeros(exact.shape)
for _ in range(iters):
  hrelax_fcf = relax(hrelax_fcf,data,dt,cf,relax_type='fc')
  hrelax_fcf = relax(hrelax_fcf,data,dt,cf,relax_type='f')
  print(f'  error = {la.norm(hrelax_fcf-exact):.4e}, residual = {la.norm(residual(hrelax_fcf,data,dt)):.4e}')

print('\nTwo Level MG')
two_level = np.zeros(exact.shape)
for _ in range(mg_iters):
  two_level = two_level_mg(two_level,data,dt,cf,2)
  print(f'  error = {la.norm(two_level-exact):.4e}, residual = {la.norm(residual(two_level,data,dt)):.4e}')

fig, (ax1,ax2,ax3) = plt.subplots(3,1)
ax1.plot(timenodes,exact)
ax1.set_title('fine')

ax2.plot(timenodes,hrelax_fcf)
ax2.set_title('hrelax_fcf')

ax3.plot(timenodes,two_level)
ax3.set_title('two_level')

plt.show()
