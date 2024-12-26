from __future__ import print_function

import argparse
import numpy             as np
import statistics        as stats
import sys
import time
from   timeit            import default_timer as timer
import torch
import torch.nn          as nn
import torchbraid
import torchbraid.utils
from   torchbraid.mgopt  import root_print, compute_levels
import myTransformerLayers
from torchbraid.utils import LPDropout as Dropout
#from torch.nn import  Dropout

from mpi4py              import MPI

from generation          import generate
from load_save           import load, save
from positional_encoding import PositionalEncoding

__all__ = ['OpenLayer', 'CloseLayer', 'StepLayer', 
           'parse_args', 'ParallelNet']

####################################################################################
####################################################################################
# Network architecture is Open + ResNet + Close
# The StepLayer defines the ResNet (ODENet)

class OpenLayer(nn.Module):
  def __init__(self, d_model, dropout, src_vocab, tgt_vocab, device):
    super().__init__()

    torch.manual_seed(0)
    dropout = 0 # No dropout in openlayer just in case
    self.d_model   = d_model
    self.dropout   = dropout
    self.src_vocab = src_vocab
    self.tgt_vocab = tgt_vocab
    self.device    = device

    self.src_embedding = nn.Embedding(len(src_vocab), d_model, device=device)
    self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, device=device)
    self.positional_encoding = PositionalEncoding(d_model, dropout, False)

  def forward(self, src, tgt):  #/ src: [b, Ls]
                                #/ tgt: [b, Lt]
    global tgt_mask, src_key_padding_mask, tgt_key_padding_mask, \
                                        memory_key_padding_mask
    ## Encoder
    encoder_masks = self.get_encoder_masks(src)
    x = self.src_embedding(src.T)                #/ [Ls, b, d]
    x = self.positional_encoding(x)              #/ [Ls, b, d]

    ## Decoder
    decoder_masks = self.get_decoder_masks(tgt, encoder_masks)
    y = self.tgt_embedding(tgt.T)    #/ [Lt, b, d]
    y = self.positional_encoding(y)  #/ [Lt, b, d]

    z = torch.stack((x, y))

    tgt_mask                = decoder_masks[               'tgt_mask']
    src_key_padding_mask    = encoder_masks[   'src_key_padding_mask']
    tgt_key_padding_mask    = decoder_masks[   'tgt_key_padding_mask']
    memory_key_padding_mask = decoder_masks['memory_key_padding_mask']

    return z

  def get_attention_mask(self, Lt):
    return nn.Transformer.generate_square_subsequent_mask(Lt).to(self.device)

  def get_decoder_masks(self, tgt, encoder_masks):
    return {'tgt_mask': self.get_attention_mask(Lt=tgt.shape[1]),
            'tgt_key_padding_mask': 
                            self.get_padding_mask(tgt, self.tgt_vocab.pad_id),
            'memory_key_padding_mask': encoder_masks['src_key_padding_mask']}

  def get_encoder_masks(self, src):
    return {'src_key_padding_mask': 
                            self.get_padding_mask(src, self.src_vocab.pad_id)}

  def get_padding_mask(self, x, pad_id):
    return torch.zeros(x.shape, device=self.device) \
            .masked_fill((x == pad_id), float('-inf'))
# end layer

class CloseLayer(nn.Module):
  def __init__(self, d_model, tgt_vocab, device):
    super().__init__()
    torch.manual_seed(0)

    self.lm_head = nn.Linear(d_model, len(tgt_vocab), device)
    self.log_softmax = nn.LogSoftmax(dim=-1)

  def forward(self, y):  # y: [L', b, d]
    logits = self.lm_head(y)
    log_probs = self.log_softmax(logits)

    return log_probs.transpose(0, 1)
# end layer

class StepLayer_enc(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_size, 
                                              max_sequence_length, device):
    super().__init__()
    torch.manual_seed(0)
    self.encoder_layer = myTransformerLayers.TransformerEncoderLayer(d_model, nhead, 
                                                    dim_feedforward, dropout, device=device)
    self.zeros_tensor = torch.zeros(
               size=(max_sequence_length, batch_size, d_model), device=device)

  def forward(self, z):
    # t0 = time.time()
    layer = self.encoder_layer
    x, y = z
    x = (sa_x := layer._sa_block(layer.norm1(x), None, src_key_padding_mask)) \
      + (ff_x := layer._ff_block(layer.norm2(x + sa_x)))
    # print('enc pre-sa')
    # sa_x = layer._sa_block(layer.norm1(x), None, src_key_padding_mask)
    # print('enc pre-ff')
    # ff_x = layer._ff_block(layer.norm2(x + sa_x))
    # print('enc post-ff')
    # x = sa_x + ff_x

    z = torch.stack((x, self.zeros_tensor[:y.shape[0], :y.shape[1]]))
    # t1 = time.time()
    # print(f'Fwd time encoder layer: {t1 - t0} seconds')
    # print(f'x={x}')
    return z

class usual_StepLayer_enc(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_size,
                                              max_sequence_length, device):
    super().__init__()
    torch.manual_seed(0)
    self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                    dim_feedforward, dropout, device=device)
    self.zeros_tensor = torch.zeros(
               size=(max_sequence_length, batch_size, d_model), device=device)

  def forward(self, z):
    # t0 = time.time()
    layer = self.encoder_layer
    x, y = z
    x = (sa_x := layer._sa_block(layer.norm1(x), None, src_key_padding_mask)) \
      + (ff_x := layer._ff_block(layer.norm2(x + sa_x)))

    z = torch.stack((x, self.zeros_tensor[:y.shape[0], :y.shape[1]]))
    # t1 = time.time()
    # print(f'Fwd time encoder layer: {t1 - t0} seconds')
    # print(f'x={x}')
    return z

class StepLayer_dec(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_size, 
                                              max_sequence_length, device):
    super().__init__()
    torch.manual_seed(0)
    self.decoder_layer = myTransformerLayers.TransformerDecoderLayer(d_model, nhead, 
                                                    dim_feedforward, dropout, device=device)
    self.zeros_tensor = torch.zeros(
               size=(max_sequence_length, batch_size, d_model), device=device)

  def forward(self, z):
    # t0 = time.time()
    layer = self.decoder_layer
    x, y = z
    y = ( sa_y := layer._sa_block (layer.norm1(y), 
                                   tgt_mask, tgt_key_padding_mask)) \
      + (mha_y := layer._mha_block(layer.norm2(y + sa_y), x,
                                   None, memory_key_padding_mask)) \
      + ( ff_y := layer._ff_block (layer.norm3(y + sa_y + mha_y)))
    # print('dec pre-sa')
    # sa_y = layer._sa_block (layer.norm1(y), 
    #                                tgt_mask, tgt_key_padding_mask)
    # print('dec pre-mha')
    # mha_y = layer._mha_block(layer.norm2(y + sa_y), x,
    #                                None, memory_key_padding_mask)
    # print('dec pre-ff')
    # ff_y = layer._ff_block (layer.norm3(y + sa_y + mha_y))
    # print('dec post-ff')
    # y = sa_y + mha_y + ff_y

    z = torch.stack((self.zeros_tensor[:x.shape[0], :x.shape[1]], y))
    # t1 = time.time()
    # print(f'Fwd time decoder layer: {t1 - t0} seconds')

    return z

class usual_StepLayer_dec(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_size,
                                              max_sequence_length, device):
    super().__init__()
    torch.manual_seed(0)
    self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead,
                                                    dim_feedforward, dropout, device=device)
    self.zeros_tensor = torch.zeros(
               size=(max_sequence_length, batch_size, d_model), device=device)

  def forward(self, z):
    # t0 = time.time()
    layer = self.decoder_layer
    x, y = z

    y = ( sa_y := layer._sa_block (layer.norm1(y),
                                   tgt_mask, tgt_key_padding_mask)) \
      + (mha_y := layer._mha_block(layer.norm2(y + sa_y), x,
                                   None, memory_key_padding_mask)) \
      + ( ff_y := layer._ff_block (layer.norm3(y + sa_y + mha_y)))

    z = torch.stack((self.zeros_tensor[:x.shape[0], :x.shape[1]], y))
    # t1 = time.time()
    # print(f'Fwd time decoder layer: {t1 - t0} seconds')

    return z

class StepLayer_dec_SA(nn.Module):
  def __init__(self, d_model, nhead, dropout, batch_size, max_sequence_length, 
                                                    device):
    raise Exception("Uh, dont' be here; need to impelment")
    super().__init__()
    torch.manual_seed(0)
    self.self_attn = nn.MultiheadAttention(
      d_model, nhead, dropout=dropout, batch_first=False, bias=True, 
      device=device,
    )
    self.norm1 = nn.LayerNorm(d_model, eps=1e-5, #bias=True, 
                                                 device=device)
    self.dropout1 = Dropout(dropout, device=device)

    self.zeros_tensor = torch.zeros(
               size=(max_sequence_length, batch_size, d_model), device=device)

  def _sa_block(self, x, attn_mask, key_padding_mask):
    x = self.self_attn(
      x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, 
      is_causal=False, need_weights=False,
    )[0]
    return self.dropout1(x)

  def forward(self, z):
    # t0 = time.time()
    x, y = z

    y = (sa_y := self._sa_block(self.norm1(y), 
                                tgt_mask, tgt_key_padding_mask)) \

    z = torch.stack((self.zeros_tensor[:x.shape[0], :x.shape[1]], y))
    # t1 = time.time()
    # print(f'Fwd time decoder layer: {t1 - t0} seconds')

    return z

class StepLayer_dec_MHA_FF(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_size, 
                                              max_sequence_length, device):
    super().__init__()
    torch.manual_seed(0)
    print('HASKDJFKL;ASJDFKLJAL;SKDFJ')
    self.multihead_attn = nn.MultiheadAttention(
                           d_model, nhead, dropout=dropout, batch_first=False, 
                                                     bias=True, device=device)    
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward, 
                             bias=True, device=device)
    self.dropout = Dropout(dropout, device=device)
    self.linear2 = nn.Linear(dim_feedforward, d_model, 
                             bias=True, device=device)

    self.norm2 = nn.LayerNorm(d_model, eps=1e-5, #bias=True, 
                                                 device=device)
    self.norm3 = nn.LayerNorm(d_model, eps=1e-5, #bias=True, 
                                                 device=device)

    print("---------------device ", device, " --------------- ")

    self.dropout2 = Dropout(dropout, device=device)
    self.dropout3 = Dropout(dropout, device=device)

    self.zeros_tensor = torch.zeros(
               size=(max_sequence_length, batch_size, d_model), device=device)

  def _ff_block(self, x):
      x = self.linear2(self.dropout(self.linear1(x).relu()))
      return self.dropout3(x)

  def _mha_block(self, x, mem, attn_mask, key_padding_mask):
      x = self.multihead_attn(
        x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, 
        is_causal=False, need_weights=False
      )[0]
      return self.dropout2(x)

  def forward(self, z):
    # t0 = time.time()
    x, y = z

    y = (mha_y := self._mha_block(self.norm2(y), x, None, 
                                                     memory_key_padding_mask)) \
      + ( ff_y := self._ff_block (self.norm3(y + mha_y)))

    z = torch.stack((self.zeros_tensor[:x.shape[0], :x.shape[1]], y))
    # t1 = time.time()
    # print(f'Fwd time decoder layer: {t1 - t0} seconds')

    return z
####################################################################################
####################################################################################

# Parallel network class
# local_steps: number of ResNet layers per processor
# all other parameter definitions are in argument parser comments below
class ParallelNet(nn.Module):
  def __init__(
    self, d_model, nhead, dim_feedforward, dropout, source_vocabulary, 
    target_vocabulary, batch_size, max_sequence_length, device, split_decoder,
    local_steps=8, Tf=1.0, max_levels=1, serial_fwd=False, bwd_max_iters=1, 
    fwd_max_iters=2, print_level=0, braid_print_level=0, cfactor=4, 
    fine_fcf=False, skip_downcycle=True, fmg=False, relax_only_cg=0, 
    user_mpi_buf=False, comm_lp=MPI.COMM_WORLD, comm_dp=None,
  ):
    super(ParallelNet, self).__init__()

    self.d_model             = d_model
    self.nhead               = nhead            
    self.dim_feedforward     = dim_feedforward  
    self.dropout             = dropout          
    self.source_vocabulary   = source_vocabulary
    self.target_vocabulary   = target_vocabulary
    self.batch_size          = batch_size       
    self.max_sequence_length = max_sequence_length
    self.device              = device           
    self.split_decoder       = split_decoder           
    self.comm_lp             = comm_lp
    self.comm_dp             = comm_dp
    self.serial_fwd          = serial_fwd

    numprocs = self.comm_lp.Get_size()

    step_layer_enc = lambda: StepLayer_enc(
                                   d_model, nhead, dim_feedforward, dropout, 
                                   batch_size, max_sequence_length, device)
    if not split_decoder:
      print('IN NOT SPLIT')
      step_layer_dec = lambda: StepLayer_dec(
                                     d_model, nhead, dim_feedforward, dropout, 
                                     batch_size, max_sequence_length, device)
      layers    = [   step_layer_enc   ,    step_layer_dec   ]
      num_steps = [local_steps*numprocs, local_steps*numprocs]

    else:
      print('IN SPLIT ')
      step_layer_sa     = lambda: StepLayer_dec_SA(d_model, nhead, dropout, 
                                      batch_size, max_sequence_length, device)
      step_layer_mha_ff = lambda: StepLayer_dec_MHA_FF(d_model, nhead, 
            dim_feedforward, dropout, batch_size, max_sequence_length, device)

      layers, num_steps = [], []
      for i in range(local_steps * numprocs): 
        layers.append(step_layer_enc   ); num_steps.append(1)
      for i in range(local_steps * numprocs):
        layers.append(step_layer_sa    ); num_steps.append(1)
        layers.append(step_layer_mha_ff); num_steps.append(1)

    print(f'#fwd-levels used: {max_levels if not self.serial_fwd else 1}')

    self.parallel_nn = torchbraid.LayerParallel(
      comm_lp, 
      layers, num_steps, #step_layer, local_steps*numprocs, 
      Tf,
      max_fwd_levels=max_levels if not self.serial_fwd else 1,#1,#max_levels, 
      max_bwd_levels=max_levels,
      max_iters=2, user_mpi_buf=user_mpi_buf,
    )
    self.parallel_nn.setBwdMaxIters(bwd_max_iters)
    self.parallel_nn.setFwdMaxIters(fwd_max_iters)
    self.parallel_nn.setPrintLevel(print_level, True)
    self.parallel_nn.setPrintLevel(braid_print_level, False)
    self.parallel_nn.setCFactor(cfactor)
    self.parallel_nn.setSkipDowncycle(skip_downcycle)
    self.parallel_nn.setBwdRelaxOnlyCG(relax_only_cg)
    self.parallel_nn.setFwdRelaxOnlyCG(relax_only_cg)
    if fmg:
      self.parallel_nn.setFMG()

    self.parallel_nn.setNumRelax(1)  # FCF relaxation default on coarse levels
    if not fine_fcf:
      self.parallel_nn.setNumRelax(0, level=0)  # Set F-Relaxation only on the fine grid
    else:
      self.parallel_nn.setNumRelax(1, level=0)  # Set FCF-Relaxation on the fine grid

    # this object ensures that only the LayerParallel code runs on ranks!=0
    compose = self.compose = self.parallel_nn.comp_op()

    # by passing this through 'compose' (mean composition: e.g. OpenLayer o channels)
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.open_nn = compose(OpenLayer, 
               d_model, 0, source_vocabulary, target_vocabulary, device)
    self.close_nn = compose(CloseLayer, 
               d_model, target_vocabulary, device)

  def new_mask(self, x=None): 
    """
    Generates a new mask given a sample input
    """
    for layer in self.parallel_nn.local_layers.modules():
      # print(layer)
      if isinstance(layer, Dropout):
        # print('Generating')
        if layer.mask is not None:
          layer.generate_new_mask(x, self.device)

  def saveSerialNet(self, name):
    # Model can be reloaded in serial format with: model = torch.load(filename)
    serial_nn = self.parallel_nn.buildSequentialOnRoot()
    if self.comm_lp.Get_rank() == 0:
      s_net = SerialNet(
        -1, -1, -1, serial_nn=serial_nn, open_nn=self.open_nn, 
        close_nn=self.close_nn,
      )
      s_net.eval()
      torch.save(s_net, name)

  def forward(self, src, tgt):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x)
    # this makes sure this is run on only processor 0
    global tgt_mask, src_key_padding_mask, tgt_key_padding_mask, \
           memory_key_padding_mask

    tgt_mask                = 'asfd'
    src_key_padding_mask    = 'asdf'
    tgt_key_padding_mask    = 'asdf'
    memory_key_padding_mask = 'asdf'

    t0_open_layer_time = time.time()
    z = self.compose(self.open_nn, src, tgt)
    t1_open_layer_time = time.time()

    t0_masks_comm_time = time.time()
    tgt_mask, src_key_padding_mask, tgt_key_padding_mask, \
     memory_key_padding_mask = self.comm_lp.bcast(
      [tgt_mask, src_key_padding_mask, tgt_key_padding_mask, 
                                    memory_key_padding_mask], root=0
    )
    t1_masks_comm_time = time.time()
    
    lp_rank = self.comm_lp.Get_rank()
    # device = z.device
    gpu_id = torch.cuda.current_device()
    device = torch.device(f"cuda:{gpu_id}")
    #print("MPI Rank:", lp_rank, "GPU Name:", gpu_id, " device   ", device)

    tgt_mask =tgt_mask.to(device)
    src_key_padding_mask = src_key_padding_mask.to(device) 
    tgt_key_padding_mask = tgt_key_padding_mask.to(device)
    memory_key_padding_mask = memory_key_padding_mask.to(device)

    t0_continuous_block_time = time.time()
    z = self.parallel_nn(z)
    t1_continuous_block_time = time.time()

    t0_close_layer_time = time.time()
    mem, y = z
    y = self.compose(self.close_nn, y)
    t1_close_layer_time = time.time()

    lp_rank = self.comm_lp.Get_rank()
    dp_rank = self.comm_dp.Get_rank() if self.comm_dp is not None else None
    if 0:
      # print(f'''lp_rank={lp_rank}, dp_rank={dp_rank}: {t1_continuous_block_time - t0_continuous_block_time :.4f}''')
      print(f'''lp_rank={lp_rank}, dp_rank={dp_rank}, open={t1_open_layer_time - t0_open_layer_time:.4f}, masks-comm={t1_masks_comm_time - t0_masks_comm_time}, CB={t1_continuous_block_time - t0_continuous_block_time :.4f}, close={t1_close_layer_time - t0_close_layer_time}''')

    return y

class SerialNet(nn.Module):
  def __init__(
    self, d_model, nhead, dim_feedforward, dropout, source_vocabulary, 
    target_vocabulary, batch_size, max_sequence_length, device, split_decoder,
    local_steps=8, Tf=1.0, serial_nn=None, open_nn=None, close_nn=None,
  ):
    super(SerialNet, self).__init__()

    self.d_model             = d_model
    self.nhead               = nhead            
    self.dim_feedforward     = dim_feedforward  
    self.dropout             = dropout          
    self.source_vocabulary   = source_vocabulary
    self.target_vocabulary   = target_vocabulary
    self.batch_size          = batch_size       
    self.max_sequence_length = max_sequence_length
    self.device              = device           
    self.split_decoder       = split_decoder
    self.comm_lp             = None

    numprocs = 1

    step_layer_enc = lambda: usual_StepLayer_enc(
                                   d_model, nhead, dim_feedforward, dropout, 
                                   batch_size, max_sequence_length, device)

    if not split_decoder:
      step_layer_dec = lambda: usual_StepLayer_dec(
                                     d_model, nhead, dim_feedforward, dropout, 
                                     batch_size, max_sequence_length, device)
      layers    = [   step_layer_enc   ,    step_layer_dec   ]
      num_steps = [local_steps*numprocs, local_steps*numprocs]

    else:
      step_layer_sa     = lambda: StepLayer_dec_SA(d_model, nhead, dropout, 
                                      batch_size, max_sequence_length, device)
      step_layer_mha_ff = lambda: StepLayer_dec_MHA_FF(d_model, nhead, 
            dim_feedforward, dropout, batch_size, max_sequence_length, device)

      layers, num_steps = [], []
      for i in range(local_steps * numprocs): 
        layers.append(step_layer_enc   ); num_steps.append(1)
      for i in range(local_steps * numprocs):
        layers.append(step_layer_sa    ); num_steps.append(1)
        layers.append(step_layer_mha_ff); num_steps.append(1)

    if serial_nn is None:
      parallel_nn = torchbraid.LayerParallel(
        MPI.COMM_SELF, layers, num_steps, Tf,
        max_fwd_levels=1, max_bwd_levels=1, max_iters=1
      )
      parallel_nn.setPrintLevel(0, True)
      self.serial_nn = parallel_nn.buildSequentialOnRoot()
    else:
      self.serial_nn = serial_nn

    if open_nn is None:
      self.open_nn = OpenLayer(d_model, dropout, source_vocabulary, 
                                                 target_vocabulary, device=device)
    else: self.open_nn = open_nn

    if close_nn is None: self.close_nn = CloseLayer(d_model, 
                                                    target_vocabulary, device=device)
    else: self.close_nn = close_nn

  def forward(self, src, tgt):
    # global tgt_attention_mask, src_padding_mask, tgt_padding_mask, \
    #        mem_padding_mask
    # (x, y, tgt_attention_mask, src_padding_mask, tgt_padding_mask, 
    # mem_padding_mask,) = self.open_nn(src, tgt)
    # x = torch.stack((x, y))
    x = self.open_nn(src, tgt)
    x = self.serial_nn(x)
    mem, y = x
    y = self.close_nn(y)

    return y

####################################################################################
####################################################################################

# Parse command line 
def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """

  # Command line settings
  parser = argparse.ArgumentParser(description='MNIST example argument parser')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')

  # artichtectural settings
  parser.add_argument('--steps', type=int, default=32, metavar='N',
                      help='Number of times steps in the resnet layer (default: 32)')
  parser.add_argument('--Tf',type=float,default=1.0,
                      help='Final time for ResNet layer-parallel part')
  parser.add_argument('--serial-file', type=str, default=None,
                      help='Save network to file in serial (not parallel) format')

  # algorithmic settings (batching)
  parser.add_argument('--percent-data', type=float, default=0.05, metavar='N',
                      help='how much of the data to read in and use for training/testing')
  parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                      help='input batch size for training (default: 50)')
  parser.add_argument('--epochs', type=int, default=3, metavar='N',
                      help='number of epochs to train (default: 3)')
  parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                      help='learning rate (default: 0.01)')

  # algorithmic settings (layer-parallel)
  parser.add_argument('--lp-max-levels', type=int, default=3, metavar='N',
                      help='Layer parallel max number of levels (default: 3)')
  parser.add_argument('--lp-bwd-max-iters', type=int, default=1, metavar='N',
                      help='Layer parallel max backward iterations (default: 1)')
  parser.add_argument('--lp-fwd-max-iters', type=int, default=2, metavar='N',
                      help='Layer parallel max forward iterations (default: 2)')
  parser.add_argument('--lp-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel internal print level (default: 0)')
  parser.add_argument('--lp-braid-print-level', type=int, default=0, metavar='N',
                      help='Layer parallel braid print level (default: 0)')
  parser.add_argument('--lp-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor (default: 4)')
  parser.add_argument('--lp-fine-fcf',action='store_true', default=False,
                      help='Layer parallel fine FCF for forward solve, on or off (default: False)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('--warm-up', action='store_true', default=False,
                      help='Warm up for GPU timings (default: False)')
  parser.add_argument('--lp-user-mpi-buf',action='store_true', default=False,
                      help='Layer parallel use user-defined mpi buffers (default: False)')
  parser.add_argument('--lp-use-downcycle', action='store_true', default=False,
                      help='Layer parallel use downcycle on or off (default: False)')

  # data parallelism
  parser.add_argument('--dp-size', type=int, default=1, metavar='N',
                      help='Data parallelism (used if value != 1)')

  ## additional arguments
  parser.add_argument('--d_model'              , type=int  , default=  256      )
  parser.add_argument('--nhead'                , type=int  , default=    8      )
  parser.add_argument('--dim_feedforward'      , type=int  , default= 1024      )
  parser.add_argument('--drop_last'            , action='store_true'            )
  parser.add_argument('--dropout'              , type=float, default=    0.     )
  parser.add_argument('--gradient_accumulation', type=int  , default=    1      )
  parser.add_argument('--num_warmup_steps'     , type=int  , default= 4000      )
  parser.add_argument('--debug'                , action='store_true'            )
  parser.add_argument('--enforce_serial'       , action='store_true'            )
  parser.add_argument('--scale'                , action='store_true'            )
  parser.add_argument('--split_decoder'        , action='store_true'            )
  parser.add_argument('--initialize_parameters', action='store_true'            )
  # parser.add_argument('--seed'                 , type=int  , default=    0      )
  parser.add_argument('--tokenization'         , type=str  , default='news-web' )
  parser.add_argument('--vocab_size'           , type=int  , default=32000      )
  parser.add_argument('--load'                 , action='store_true'            )
  parser.add_argument('--num_training_batches' , type=int  , default= 2000      )
  parser.add_argument('--serial_fwd'           , action='store_true'            )

  ##
  # Do some parameter checking
  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()

  ## Temp
  args.nhead = (args.d_model * 8) // 512
  args.dim_feedforward = 4 * args.d_model

  if procs % args.dp_size != 0:
    root_print(
      rank, 1, 1, (
        'Data parallel size must be an even multiple of the number of ' 
      + 'processors: %d %d'
      ) % (procs, args.dp_size) 
    )
    sys.exit(0)
  else:
    procs_lp = int(procs / args.dp_size)

  ##
  # Compute number of parallel-in-time multigrid levels 
  def compute_levels(num_steps, min_coarse_size, cfactor):
    from math import log, floor
    # Find L such that ( max_L min_coarse_size*cfactor**L <= num_steps)
    levels = floor(log(float(num_steps) / min_coarse_size, cfactor)) + 1

    if levels < 1:
      levels = 1
    return levels

  if args.lp_max_levels < 1:
    min_coarse_size = 3
    args.lp_max_levels = compute_levels(
      args.steps, min_coarse_size, args.lp_cfactor,
    )

  if args.steps % procs_lp != 0:
    root_print(
      rank, 1, 1, (
        'Steps must be an even multiple of the number of layer parallel '
        'processors: %d %d'
      ) % (args.steps, procs_lp)
    )
    sys.exit(0)

  return args


####################################################################################
####################################################################################



