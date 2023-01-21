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

# some helpful examples
# 
# BATCH_SIZE=50
# STEPS=12
# CHANNELS=8

# IN SERIAL
# python  main.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out
# mpirun -n 4 python  main.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out

from __future__ import print_function
import sys
import argparse
import torch
import torchbraid
import torchbraid.utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import statistics as stats

import numpy as np
import matplotlib.pyplot as pyplot

from torchvision import datasets, transforms

from timeit import default_timer as timer

from mpi4py import MPI

import preproc    
from torch.utils.data import DataLoader  
from models import PositionalEncoding, PE_Alternative

import time

L_mem, L_tgt = None, None

fn = '../log.txt'
f = open(fn, 'w'); f.close()
# f = open('../llog3.txt', 'w'); f.close()
def write_log(*x):
  with open(fn, 'a') as f:
    f.write(' '.join(str(i) + '\n' for i in x))


def encode_encoder(x, mask):
  y = x.reshape(x.shape[0]*x.shape[2], x.shape[1])
  y = torch.cat((y, mask), axis=0)
  return y

def encode_decoder(mem, tgt, msk_pad_mem, msk_pad_tgt):
  global L_mem, L_tgt
  n, L_mem, d, L_tgt = *mem.shape, tgt.shape[1]
  y = mem.reshape(n*d, L_mem)
  y = torch.cat((y, msk_pad_mem), axis=0)
  z = tgt.reshape(n*d, L_tgt)
  z = torch.cat((z, msk_pad_tgt), axis=0)
  t = torch.zeros(y.shape[0] + z.shape[0], max(L_mem, L_tgt),
                  dtype=mem.dtype).to(mem.device)
  t[:y.shape[0], :L_mem] = y
  t[y.shape[0]:, :L_tgt] = z
  return t

def decode_encoder(x):
  n = x.shape[0]//(128 + 1)
  x, mask = x[:n*128], x[n*128:]
  x = x.reshape(n, x.shape[1], 128)
  return x, mask

def decode_decoder(x):
  n = x.shape[0]//(128+1+128+1)
  mem, msk_pad_mem, tgt, msk_pad_tgt = x[:n*128], x[n*128:n*(128+1)], \
                                x[n*(128+1):n*(128+1+128)], x[n*(128+1+128):]
  mem, msk_pad_mem = mem[:, :L_mem], msk_pad_mem[:, :L_mem]
  tgt, msk_pad_tgt = tgt[:, :L_tgt], msk_pad_tgt[:, :L_tgt]
  mem = mem.reshape(n, mem.shape[1], 128)
  tgt = tgt.reshape(n, tgt.shape[1], 128)
  return mem, tgt, msk_pad_mem, msk_pad_tgt


def root_print(rank,s):
  if rank==0:
    print(s)

class OpenLayer(nn.Module):
  def __init__(self, encoding):
    super(OpenLayer, self).__init__()
    self.encoding = encoding

    self.emb_src = nn.Embedding(50001, 128)
    self.emb_tgt = nn.Embedding(50001, 128)
    self.dout_src = nn.Dropout(p=.1)
    self.dout_tgt = nn.Dropout(p=.1)
    self.posenc = PositionalEncoding(128) if encoding == 'Torch'\
      else PE_Alternative(128) if encoding == 'Alternative'\
      else Exception('encoding unknown')

    self.tgt = None

  def forward(self, x):
    self.msk_pad_src = (x == 0)

    x = self.emb_src(x)
    x = self.dout_src(x)
    x = self.posenc(x)

    tgt = self.tgt
    assert tgt is not None
    self.msk_pad_tgt = (tgt == 0)
    self.msk_pad_mem = self.msk_pad_src
    tgt = self.emb_tgt(tgt)
    tgt = self.dout_tgt(tgt)
    tgt = self.posenc(tgt)
    self.tgt = tgt

    return x
# end layer

class CloseLayer(nn.Module):
  def __init__(self):
    super(CloseLayer, self).__init__()
    self.fc = nn.Linear(128, 50001)

  def forward(self, x):
    x = self.fc(x.transpose(0, 1))

    return x
# end layer

class StepLayer_enc(nn.Module):
  def __init__(self):
    super(StepLayer_enc, self).__init__()

    self.msk_pad_src = None

    ## Encoder    
    self.enc_att = nn.MultiheadAttention(
      embed_dim=128, 
      num_heads=8, 
      dropout=.3, 
      batch_first=True
    )

    self.enc_fc1 = nn.Linear(128, 1024)
    self.enc_dout = nn.Dropout(.1)
    self.enc_fc2 = nn.Linear(1024, 128)

    self.enc_ln1 = nn.LayerNorm(128, eps=1e-5)
    self.enc_ln2 = nn.LayerNorm(128, eps=1e-5)
    self.enc_dout1 = nn.Dropout(.1)
    self.enc_dout2 = nn.Dropout(.1)

  def forward(self, y):
    # ContinuousBlock - dxdtEncoder1DBlock
    ## Encoder
    # msk_pad_src = self.msk_pad_src#(src == 0)
    src, msk_pad_src = decode_encoder(y)

    x = src

    x_sa, _ = self.enc_att(x, x, x, key_padding_mask=msk_pad_src)
    x_sa = self.enc_dout1(x_sa)
    x = self.enc_ln1(x + x_sa)

    x_ff = self.enc_fc1(x).relu()
    x_ff = self.enc_dout(x_ff)
    x_ff = self.enc_fc2(x_ff)
    x_ff = self.enc_dout2(x_ff)
    x = self.enc_ln2(x + x_ff)

    y = encode_encoder(x, msk_pad_src)

    return y
# end layer

class StepLayer_dec(nn.Module):
  def __init__(self):
    super(StepLayer_dec, self).__init__()

    self.mem = None
    self.tgt = None
    self.msk_pad_mem = None
    self.msk_pad_tgt = None

    ## Decoder
    self.dec_att_tgt = nn.MultiheadAttention(
      embed_dim=128, 
      num_heads=8, 
      dropout=.3, 
      batch_first=True
    )
    self.dec_att_mha = nn.MultiheadAttention(
      embed_dim=128, 
      num_heads=8, 
      dropout=.3, 
      batch_first=True
    )
    self.dec_fc1 = nn.Linear(128, 1024)
    self.dec_dout = nn.Dropout(.1)
    self.dec_fc2 = nn.Linear(1024, 128)

    self.dec_ln1 = nn.LayerNorm(128, eps=1e-5)
    self.dec_ln2 = nn.LayerNorm(128, eps=1e-5)
    self.dec_ln3 = nn.LayerNorm(128, eps=1e-5)
    self.dec_dout1 = nn.Dropout(.1)
    self.dec_dout2 = nn.Dropout(.1)
    self.dec_dout3 = nn.Dropout(.1)

  def forward(self, y):
    # ContinuousBlock - dxdtEncoder1DBlock

    mem, tgt, msk_pad_mem, msk_pad_tgt = decode_decoder(y)
    msk_pad_mem = msk_pad_mem.bool()
    msk_pad_tgt = msk_pad_tgt.bool()

    ## Decoder
    msk_tgt = nn.Transformer.generate_square_subsequent_mask(
                             sz=tgt.shape[1]).to(mem.device)
    # msk_pad_mem = self.msk_pad_mem
    # msk_pad_tgt = self.msk_pad_tgt#(tgt == 0)

    tgt_sa, _ = self.dec_att_tgt(tgt, tgt, tgt, attn_mask=msk_tgt, 
                              key_padding_mask=msk_pad_tgt)
    tgt_sa = self.dec_dout1(tgt_sa)
    tgt = self.dec_ln1(tgt + tgt_sa)

    x_mha, _ = self.dec_att_mha(tgt, mem, mem, key_padding_mask=msk_pad_mem)
    x_mha = self.dec_dout2(x_mha)
    x = self.dec_ln2(tgt + x_mha)

    x_ff = self.dec_fc1(x).relu()
    x_ff = self.dec_dout(x_ff)
    x_ff = self.dec_fc2(x_ff)
    x_ff = self.dec_dout3(x_ff)
    x = self.dec_ln3(x + x_ff)

    y = encode_decoder(mem, x, msk_pad_mem, msk_pad_tgt)

    return y
# end layer


class SerialNet(nn.Module):
  def __init__(self,encoding,local_steps=8,Tf=1.0,serial_nn=None,open_nn=None,close_nn=None):
    super(SerialNet, self).__init__()
    
    ## Step layer
    if serial_nn is None:
      step_layer_enc = lambda: StepLayer_enc()
      step_layer_dec = lambda: StepLayer_dec()
      numprocs = 1
      parallel_nn_enc = torchbraid.LayerParallel(MPI.COMM_SELF,step_layer_enc,numprocs*local_steps,Tf,
        #max_levels=1,
        max_iters=1)
      parallel_nn_dec = torchbraid.LayerParallel(MPI.COMM_SELF,step_layer_dec,numprocs*local_steps,Tf,
        max_iters=1)
      parallel_nn_enc.setPrintLevel(0)
      parallel_nn_dec.setPrintLevel(0)
      
      self.serial_nn_enc = parallel_nn_enc.buildSequentialOnRoot()
      self.serial_nn_dec = parallel_nn_dec.buildSequentialOnRoot()
    else:
      self.serial_nn = serial_nn
      raise Exception("shouldn't be here")

    ## Open layer
    if open_nn is None:
      self.open_nn = OpenLayer(encoding)#, step_layer_enc, step_layer_dec)
    else:
      self.open_nn = open_nn

    ## Close layer
    if close_nn is None:
      self.close_nn = CloseLayer()
    else:
      self.close_nn = close_nn

  def forward(self, x):
    x = self.open_nn(x)
    tgt = self.open_nn.tgt

    ## Shapes:
    ##    src           (btch, 102, 128)
    ##    tgt           (btch, 101, 128)
    ##    msk_pad_src   (btch, 102)
    ##    msk_pad_mem   (btch, 102)
    ##    msk_pad_tgt   (btch, 101)

    # write_log(x.shape, tgt.shape, self.open_nn.msk_pad_src.shape, self.open_nn.msk_pad_mem.shape, self.open_nn.msk_pad_tgt.shape)
    msk_pad_src = self.open_nn.msk_pad_src
    msk_pad_mem = self.open_nn.msk_pad_mem
    msk_pad_tgt = self.open_nn.msk_pad_tgt

    y = encode_encoder(x, msk_pad_src)
    y = self.serial_nn_enc(y)
    mem, _ = decode_encoder(y)

    z = encode_decoder(mem, tgt, msk_pad_mem, msk_pad_tgt)
    z = self.serial_nn_dec(z)
    _, x, _, _ = decode_decoder(z)

    x = self.close_nn(x)

    return x
# end SerialNet 


class ParallelNet(nn.Module):
  def __init__(self,encoding,local_steps=8,Tf=1.0,max_levels=1,max_iters=1,fwd_max_iters=0,print_level=0,
    braid_print_level=0,cfactor=4,fine_fcf=False,skip_downcycle=True,fmg=False,relax_only_cg=0,):
    super(ParallelNet, self).__init__()

    step_layer_enc = lambda: StepLayer_enc()
    step_layer_dec = lambda: StepLayer_dec()

    numprocs = MPI.COMM_WORLD.Get_size()

    self.parallel_nn_enc = torchbraid.LayerParallel(
      MPI.COMM_WORLD,
      step_layer_enc,           #[step_layer  for i in range(numprocs)] ?
      local_steps*numprocs, #[local_steps for i in range(numprocs)] ?
      Tf,
      max_fwd_levels=max_levels,
      max_bwd_levels=max_levels,
      max_iters=max_iters,
    )
    self.parallel_nn_dec = torchbraid.LayerParallel(
      MPI.COMM_WORLD,
      step_layer_dec,           #[step_layer  for i in range(numprocs)] ?
      local_steps*numprocs, #[local_steps for i in range(numprocs)] ?
      Tf,
      max_fwd_levels=max_levels,
      max_bwd_levels=max_levels,
      max_iters=max_iters,
    )

    if fwd_max_iters>0:
      print('fwd_max_iters',fwd_max_iters)
      self.parallel_nn_enc.setFwdMaxIters(fwd_max_iters)
    self.parallel_nn_enc.setPrintLevel(print_level,True)
    self.parallel_nn_enc.setPrintLevel(braid_print_level,False)
    self.parallel_nn_enc.setCFactor(cfactor)
    self.parallel_nn_enc.setSkipDowncycle(skip_downcycle)
    # self.parallel_nn.setRelaxOnlyCG(relax_only_cg)
    if fwd_max_iters>0:
      print('fwd_max_iters',fwd_max_iters)
      self.parallel_nn_dec.setFwdMaxIters(fwd_max_iters)
    self.parallel_nn_dec.setPrintLevel(print_level,True)
    self.parallel_nn_dec.setPrintLevel(braid_print_level,False)
    self.parallel_nn_dec.setCFactor(cfactor)
    self.parallel_nn_dec.setSkipDowncycle(skip_downcycle)
    # self.parallel_nn.setRelaxOnlyCG(relax_only_cg)

    if fmg:
      self.parallel_nn_enc.setFMG()
      self.parallel_nn_dec.setFMG()
    self.parallel_nn_enc.setNumRelax(1)         # FCF elsewehre
    self.parallel_nn_dec.setNumRelax(1)         # FCF elsewehre
    if not fine_fcf:
      self.parallel_nn_enc.setNumRelax(0,level=0) # F-Relaxation on the fine grid
      self.parallel_nn_dec.setNumRelax(0,level=0) # F-Relaxation on the fine grid
    else:
      self.parallel_nn_enc.setNumRelax(1,level=0) # F-Relaxation on the fine grid
      self.parallel_nn_dec.setNumRelax(1,level=0) # F-Relaxation on the fine grid

    # this object ensures that only the LayerParallel code runs on ranks!=0
    compose_enc = self.compose_enc = self.parallel_nn_enc.comp_op()
    compose_dec = self.compose_dec = self.parallel_nn_dec.comp_op()
    
    # by passing this through 'compose' (mean composition: e.g. OpenLayer o channels) 
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.open_nn = compose(OpenLayer, encoding, step_layer_enc, 
                                                step_layer_dec)
    self.close_nn = compose(CloseLayer)
    assert self.open_nn is not None
    assert self.close_nn is not None

  def saveSerialNet(self,name):
    serial_nn = self.parallel_nn.buildSequentialOnRoot()
    if MPI.COMM_WORLD.Get_rank()==0:
      s_net = SerialNet(-1,-1,-1,serial_nn=serial_nn,open_nn=self.open_nn,close_nn=self.close_nn)
      s_net.eval()
      torch.save(s_net,name)

  def getDiagnostics(self):
    return self.parallel_nn.getDiagnostics()
 
  def forward(self, x):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x) 
    # this makes sure this is run on only processor 0
    x = self.compose(self.open_nn,x)

    mem = self.parallel_nn_enc(x)
    self.parallel_nn_dec.mem = mem
    tgt = self.parallel_nn_dec.tgt
    x = self.parallel_nn_dec(tgt)
    
    x = self.compose(self.close_nn,x)

    return x
# end ParallelNet 

def train(rank, args, model, train_loader, optimizer, epoch, compose, device):
  model.train()
  criterion = nn.CrossEntropyLoss(ignore_index=0)
  total_time = 0.0
  # batch_epochs = np.inf#args.epochs
  batch_epochs = 10000
  batch_ctr = 0
  forward_times, backward_times = [], []
  for batch_idx, (data, target) in enumerate(train_loader):
    # root_print(rank, 'checkpoint 4')
    # root_print(rank, f'data.shape {data.shape}')
    if data.shape[0] != args.batch_size:
      break
    write_log('tr1', data.shape, target.shape)
    #root_print(rank, f'train data.shape {data.shape}')
    start_time = timer()
    optimizer.zero_grad()
    # data = data.to(device)  # task 1
    target, tgt_inp = target[:, 1:], target[:, :-1] # task 2
    data, tgt_inp = data.to(device), tgt_inp.to(device) # task 2
    target = target.long()

    write_log('tr2', data.shape, tgt_inp.shape, target.shape)

    ## Forward pass
    t0_forward = time.time()
    model.open_nn.tgt = tgt_inp
    output = model(data).cpu()
    loss = compose(
      criterion,
      output.reshape(-1, output.shape[-1]),   # task 1
      target.reshape(-1)    # task 1
      # output.transpose(1,2),   # task 2     --> it's the same
      # target,   # task 2
    )
    t1_forward = time.time()
    forward_times.append(t1_forward - t0_forward)

    ## Backward pass
    t0_backward = time.time()
    loss.backward()
    t1_backward = time.time()
    backward_times.append(t1_backward - t0_backward)

    stop_time = timer()
    optimizer.step()

    total_time += stop_time-start_time
    if batch_idx % args.log_interval == 0:
      # root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
      root_print(rank,'Batch-epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
          batch_ctr, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item(),total_time/(batch_idx+1.0)))

    batch_ctr += 1
    print('batch_ctr', batch_ctr)
    print('batch_epochs', batch_epochs)
    if batch_ctr >= batch_epochs:
      print('yes?')
      break
    # elif batch_ctr%(batch_epochs//4) == 0:
    #   for g in optimizer.param_groups:
    #     g['lr'] *= .5
    #   root_print(rank, 'change lr *= .5')

  # root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
  #   epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
  #   100. * (batch_idx+1) / len(train_loader), loss.item(),total_time/(batch_idx+1.0)))

  return forward_times, backward_times

def diagnose(rank, model, test_loader,epoch):
  model.parallel_nn.diagnostics(True)
  model.eval()
  test_loss = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()

  itr = iter(test_loader)
  data,target = next(itr)

  # compute the model and print out the diagnostic information 
  with torch.no_grad():
    output = model(data)

  diagnostic = model.getDiagnostics()

  if rank!=0:
    return

  features = np.array([diagnostic['step_in'][0]]+diagnostic['step_out'])
  params = np.array(diagnostic['params'])

  fig,axs = pyplot.subplots(2,1)
  axs[0].plot(range(len(features)),features)
  axs[0].set_ylabel('Feature Norm')

  coords = [0.5+i for i in range(len(features)-1)]
  axs[1].set_xlim([0,len(features)-1])
  axs[1].plot(coords,params,'*')
  axs[1].set_ylabel('Parameter Norms: {}/tstep'.format(params.shape[1]))
  axs[1].set_xlabel('Time Step')

  fig.suptitle('Values in Epoch {}'.format(epoch))

  #pyplot.show()
  pyplot.savefig('diagnose{:03d}.png'.format(epoch))


def test(rank, args, model, test_loader, compose, device):
  model.eval()
  test_loss = 0
  correct, total = 0, 0
  criterion = nn.CrossEntropyLoss(ignore_index=0)
  batch_ctr = 0
  with torch.no_grad():
    for data, target in test_loader:#test_loader
      if data.shape[0] != args.batch_size:#64:#187
        break
      write_log('te1', data.shape, target.shape)
      #root_print(rank, f'test data.shape {data.shape}')
      # data = data.to(device)  # task 1
      target, tgt_inp = target[:, 1:], target[:, :-1] # task 2
      data, tgt_inp = data.to(device), tgt_inp.to(device) # task 2
      target = target.long()

      write_log('te2', data.shape, tgt_inp.shape, target.shape)

      model.open_nn.tgt = tgt_inp
      output = model(data).cpu()
      test_loss += compose(
        criterion,
        output.reshape(-1, output.shape[-1]),
        target.reshape(-1)
      ).item()

      output = MPI.COMM_WORLD.bcast(output,root=0)
      pred = output.reshape(-1, output.shape[-1]).argmax(dim=-1, keepdim=False)  # get the index of the max log-probability
      correct += ((pred == target.reshape(-1))*(target.reshape(-1) != 0)).sum().item()
      total += ((target.reshape(-1) != 0).sum()).item()

      batch_ctr += 1
      if batch_ctr == 1000: break

  accuracy = correct/total

  test_loss /= len(test_loader.dataset)

  # root_print(rank,'\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
  #     test_loss, correct, len(test_loader.dataset),
  #     100. * correct / len(test_loader.dataset)))
  root_print(rank,'\nTest set: Average loss: {:.4f}, Accuracy: {}\n'.format(
      test_loss, accuracy))

  return accuracy

def compute_levels(num_steps,min_coarse_size,cfactor): 
  from math import log, floor 
  
  # we want to find $L$ such that ( max_L min_coarse_size*cfactor**L <= num_steps)
  levels =  floor(log(float(num_steps)/min_coarse_size,cfactor))+1 

  if levels<1:
    levels = 1
  return levels
# end compute levels

def main():
  # Training settings
  parser = argparse.ArgumentParser(description='TORCHBRAID CIFAR10 Example')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 783253419)')
  parser.add_argument('--log-interval', type=int, default=1,#10, metavar='N',
                      metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--percent-data', type=float, default=1.0, metavar='N',
                      help='how much of the data to read in and use for training/testing')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=4, metavar='N',     # 128, 32, 
                      help='Number of times steps in the resnet layer (default: 4)')
  parser.add_argument('--encoding', type=str, default='Torch',
                      help='which positional encoding will be used for the attention')
  parser.add_argument('--digits',action='store_true', default=False, 
                      help='Train with the MNIST digit recognition problem (default: False)')
  parser.add_argument('--serial-file',type=str,default=None,
                      help='Load the serial problem from file')
  parser.add_argument('--tf',type=float,default=1.0,
                      help='Final time')

  # algorithmic settings (gradient descent and batching
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',    # 32; 128, 256, 512, ... until it crashes
                      help='input batch size for training (default: 50)')
  parser.add_argument('--epochs', type=int, default=2, metavar='N',
                      help='number of epochs to train (default: 2)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')

  # algorithmic settings (parallel or serial)
  parser.add_argument('--force-lp', action='store_true', default=False,
                      help='Use layer parallel even if there is only 1 MPI rank')
  parser.add_argument('--lp-levels', type=int, default=3, metavar='N',  # 1
                      help='Layer parallel levels (default: 3)')
  parser.add_argument('--lp-iters', type=int, default=2, metavar='N',   # 1
                      help='Layer parallel iterations (default: 2)')
  parser.add_argument('--lp-fwd-iters', type=int, default=-1, metavar='N',    # 1
                      help='Layer parallel (forward) iterations (default: -1, default --lp-iters)')
  parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                      help='Layer parallel internal print level (default: 0)')
  parser.add_argument('--lp-braid-print', type=int, default=0, metavar='N',
                      help='Layer parallel braid print level (default: 0)')
  parser.add_argument('--lp-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor (default: 4)')
  parser.add_argument('--lp-finefcf',action='store_true', default=False, 
                      help='Layer parallel fine FCF on or off (default: False)')
  parser.add_argument('--lp-use-downcycle',action='store_true', default=False, 
                      help='Layer parallel use downcycle on or off (default: False)')
  parser.add_argument('--lp-use-fmg',action='store_true', default=False, 
                      help='Layer parallel use FMG for one cycle (default: False)')
  parser.add_argument('--lp-use-relaxonlycg',action='store_true', default=0, 
                      help='Layer parallel use relaxation only on coarse grid (default: False)')

  ## save model  
  parser.add_argument('--output_fn',type=str, required=True, 
                      help='Output filename (for model saving)')
  parser.add_argument('--models_dir',type=str, required=True, 
                      help='Models directory (for model saving)')


  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  root_print(rank, f'device {device}')
  # root_print(rank, 'checkpoint 1')

  # some logic to default to Serial if on one processor,
  # can be overriden by the user to run layer-parallel
  if args.force_lp:
    force_lp = True
  elif procs>1:
    force_lp = True
  else:
    force_lp = False

  # force_lp = True 
  # REMAINING TO DEBUG SERIAL w/ gpu: CUDA OUT OF MEMORY

  root_print(rank, f'force_lp {force_lp}')

  #torch.manual_seed(torchbraid.utils.seed_from_rank(args.seed,rank))
  torch.manual_seed(args.seed)

  if args.lp_levels==-1:
    min_coarse_size = 3
    args.lp_levels = compute_levels(args.steps,min_coarse_size,args.lp_cfactor)

  local_steps = int(args.steps/procs)
  if args.steps % procs!=0:
    root_print(rank,'Steps must be an even multiple of the number of processors: %d %d' % (args.steps,procs) )
    sys.exit(0)

  root_print(rank,'Transformer ODENet:')

  root_print(rank,'-- procs    = {}\n'
                  '-- encoding = {}\n'
                  '-- tf       = {}\n'
                  '-- steps    = {}'.format(procs,args.encoding,args.tf,args.steps))

  vocs, sents = preproc.main(small=False)#True)
  voc_de, voc_en = vocs
  sents_de_tr, sents_en_tr, sents_de_te, sents_en_te = sents
  # ds_tr, ds_te = (tuple(zip(sents)) for sents in [(sents_de_tr, sents_en_tr),
  #                                                 (sents_de_te, sents_en_te)])
  ds_tr, ds_te = [(i, j) for (i, j) in zip(sents_de_tr, sents_en_tr)], [(i, j) for (i, j) in zip(sents_de_te, sents_en_te)]
  dl_tr, dl_te = (DataLoader(ds, batch_size=args.batch_size, shuffle=True, 
                             drop_last=True) for ds in (ds_tr, ds_te))

  train_set, test_set, train_loader, test_loader = ds_tr, ds_te, dl_tr, dl_te

  root_print(rank,'')

  if force_lp :
    root_print(rank,'Using ParallelNet:')
    root_print(rank,'-- max_levels     = {}\n'
                    '-- max_iters      = {}\n'
                    '-- fwd_iters      = {}\n'
                    '-- cfactor        = {}\n'
                    '-- fine fcf       = {}\n'
                    '-- skip down      = {}\n'
                    '-- relax only cg  = {}\n'
                    '-- fmg            = {}\n'.format(args.lp_levels,
                                                  args.lp_iters,
                                                  args.lp_fwd_iters,
                                                  args.lp_cfactor,
                                                  args.lp_finefcf,
                                                  not args.lp_use_downcycle,
                                                  args.lp_use_relaxonlycg,
                                                  args.lp_use_fmg))

    model = ParallelNet(encoding=args.encoding,
                        local_steps=local_steps,
                        max_levels=args.lp_levels,
                        max_iters=args.lp_iters,
                        fwd_max_iters=args.lp_fwd_iters,
                        print_level=args.lp_print,
                        braid_print_level=args.lp_braid_print,
                        cfactor=args.lp_cfactor,
                        fine_fcf=args.lp_finefcf,
                        skip_downcycle=not args.lp_use_downcycle,
                        fmg=args.lp_use_fmg,Tf=args.tf,
                        relax_only_cg=args.lp_use_relaxonlycg)


    if args.serial_file is not None:
      model.saveSerialNet(args.serial_file)
    compose = model.compose
  else:
    root_print(rank,'Using SerialNet:')
    root_print(rank,'-- serial file = {}\n'.format(args.serial_file))
    if args.serial_file is not None:
      print('loading model')
      model = torch.load(args.serial_file)
    else:
      model = SerialNet(encoding=args.encoding,local_steps=local_steps,Tf=args.tf)
    compose = lambda op,*p: op(*p)

  optimizer = optim.Adam(model.parameters(), lr=args.lr)#optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)

  epoch_times = []
  test_times = []

  # check out the initial conditions
  #if force_lp:
    #diagnose(rank, model, test_loader,0)

  # root_print(rank, 'checkpoint 3')

  model = model.to(device)

  # earlystop_ctr = 0
  # earlystop_patience = 5
  try:
    filenm = f'ContTrans_lr{args.lr}_epochs{args.epochs}_tf{fill(args.tf, 2)}_usedowncycle{args.lp_use_downcycle}_steps{args.steps}_cfactor{args.lp_cfactor}_levels{args.lp_levels}_nnodestimestasksnode{procs}_proc{device}_batchsize{args.batch_size}_maxiters{args.lp_iters}'
  except:
    filenm = 'problem.txt'
  for epoch in range(1, args.epochs + 1):
    t0_epoch = time.time()

    start_time = timer()
    root_print(rank, f'epoch {epoch}')
    times_fwd, times_bwd = train(rank, args, model, train_loader, 
                                 optimizer, epoch, compose, device)

    print(f'Average  forward time:\t{np.mean(times_fwd)} (train)')
    print(f'Average backward time:\t{np.mean(times_bwd)} (train)')

    end_time = timer()
    epoch_times += [end_time-start_time]
    
    start_time = timer()
    acc = test(rank, args, model, test_loader, compose, device)
    end_time = timer()
    test_times += [end_time-start_time]

    t1_epoch = time.time()
    print(f'Epoch time:\t{t1_epoch - t0_epoch}')

    # ## Save model after each epoch    <-- too much space!!
    # model_state = {
    #   'model_state': model.state_dict(),
    #   'optimizer': optimizer.state_dict(),
    # }
    # torch.save(
    #   model_state, 
    #   f'{args.models_dir}/{args.output_fn}.pt'
    # )

    # print out some diagnostics
    #if force_lp:
    #  diagnose(rank, model, test_loader,epoch)

  #if force_lp:
  #  timer_str = model.parallel_nn.getTimersString()
  #  root_print(rank,timer_str)

  # root_print(rank,'TIME PER EPOCH: %.2e (1 std dev %.2e)' % (stats.mean(epoch_times),stats.stdev(epoch_times)))
  # root_print(rank,'TIME PER TEST:  %.2e (1 std dev %.2e)' % (stats.mean(test_times), stats.stdev(test_times)))


if __name__ == '__main__':
  main()
