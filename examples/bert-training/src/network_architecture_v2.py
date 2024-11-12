from __future__ import print_function

import numpy as np

import sys
import statistics as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid
import torchbraid.utils

from torchbraid.mgopt import root_print, compute_levels

from timeit import default_timer as timer

from mpi4py import MPI
import math

from my_hf_bert import BertEmbeddings, BertLayer, BertPooler, BertPreTrainingHeads, BertLMPredictionHead
from transformers import BertConfig
from torchbraid.utils import LPDropout as Dropout



__all__ = [ 'OpenLayer', 'CloseLayer', 'StepLayer', 'parse_args', 'ParallelNet' ]

# Ported Huggingface layers

"""
BertForPreTraining(
 # OpenLayer
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    # Step layers
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
# Closer layer (a BertPooler and then BertPreTrainingHeads)
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (cls): BertPreTrainingHeads(
    (predictions): BertLMPredictionHead(
      (transform): BertPredictionHeadTransform(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (transform_act_fn): GELUActivation()
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      )
      (decoder): Linear(in_features=768, out_features=30522, bias=True)
    )
    (seq_relationship): Linear(in_features=768, out_features=2, bias=True)
  )
)
"""

####################################################################################
####################################################################################
# Network architecture is Open + BERT Encoders + multiple closing layers

class OpenLayer(nn.Module):
    """
    Wrapper for BertEmbeddings

    Checked; correct. 
    """
    def __init__(self, config):
        super(OpenLayer, self).__init__()
        self.embedding = BertEmbeddings(config)
  
    def forward(self, input_ids, segment_info):
        # embedding the indexed sequence to sequence of vectors
        # We ignore a bunch of the inputs which are not needed for pretraining and fine-tuning. 
        x = self.embedding(
          input_ids=input_ids,
          token_type_ids=segment_info
        )
        return x

class CloseLayerMLM(nn.Module):
    """
    Only the part of BertPreTrainingHeads for MLM. Note, no pooling here. 

    Checked; correct. 
    """
    def __init__(self, config):
        super(CloseLayerMLM, self).__init__()

        self.predictions = BertLMPredictionHead(config)

    def forward(self, x):
        return self.predictions (x)

class CloseLayerNSP(nn.Module):
    """
    Combines the BERTPooler and the BertPreTrainingHeads

    Note that in BertModel, we have
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    and in BertForPreTraining

    sequence_output, pooled_output = outputs[:2]
    prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

    where outputs is from BertModel.
    """
    def __init__(self, config):
        super(CloseLayerNSP, self).__init__()

        self.pooler = BertPooler(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        pooled_output = self.pooler(x)
        return self.seq_relationship(pooled_output)

class StepLayer(nn.Module):
    """
    Just the encoder layer

    Checked; correct
    """
    def __init__(self, config):
        super(StepLayer, self).__init__()

        self.layer = BertLayer(config)
    def forward(self, x, dt: float=1.0):    # Default must be there for shape
        # TODO: look at 
        # self.comm_lp.bcast in MT2 code 
        global mask

        x = self.layer(x, attention_mask=mask, dt=dt)

        return x

####################################################################################
####################################################################################

# Parallel network class
# local_steps: number of ResNet layers per processor
# all other parameter definitions are in argument parser comments below
class ParallelNet(nn.Module):
    def __init__(self, vocab_size, model_dimension=768, num_heads=12, seq_len=64,
        local_steps=8, Tf=1.0,max_levels=1, bwd_max_iters=1,
               fwd_max_iters=2, print_level=0, braid_print_level=0, cfactor=4,
               fine_fcf=False, skip_downcycle=True, fmg=False, relax_only_cg=0,
               user_mpi_buf=False, comm_lp=MPI.COMM_WORLD):
        super(ParallelNet, self).__init__()

        # Use the same config (except for hidden layers which is contorlled elsewhere)
        # as bert-large-uncased
        config = BertConfig(
            vocab_size=30522,          # Vocabulary size of BERT
            hidden_size=1024,           # Hidden size of the transformer layers
            num_hidden_layers=4,       # (Not used) Number of transformer layers (encoder blocks)
            num_attention_heads=16,    # Number of attention heads in each attention layer
            intermediate_size=4096,     # Size of the "intermediate" (feed-forward) layer
            hidden_act="gelu",         # Activation function ("gelu" or "relu")
            hidden_dropout_prob=0.1,   # Dropout probability for fully connected layers
            attention_probs_dropout_prob=0.1,  # Dropout probability for attention probabilities
            max_position_embeddings=512,       # Maximum number of position embeddings
            type_vocab_size=2,         # Vocabulary size of token type IDs (for sentence A/B)
            initializer_range=0.02,    # Standard deviation of the truncated_normal_initializer
            layer_norm_eps=1e-12       # Epsilon for layer normalization
        )

        step_layer = lambda: StepLayer(config)
        self.comm_lp = comm_lp
        numprocs = self.comm_lp.Get_size()

        # Seperate max_levels to forward and backawrd
        if not isinstance(max_levels, int):
            max_fwd_levels = max_levels[0]
            max_bwd_levels = max_levels[1]
        else:
            max_fwd_levels = max_levels
            max_bwd_levels = max_levels


        self.parallel_nn = torchbraid.LayerParallel(comm_lp, step_layer, local_steps*numprocs, Tf,
                                                    max_fwd_levels=max_fwd_levels, max_bwd_levels=max_bwd_levels,
                                                    max_iters=2, user_mpi_buf=user_mpi_buf)
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
        self.open_nn = compose(OpenLayer, config)
        self.close_nsp = compose(CloseLayerNSP, config)
        self.close_mlm = compose(CloseLayerMLM, config)
        # self.close_nn_mlm = compose(CloseLayerMLM, model_dimension, vocab_size)

    def saveSerialNet(self, name):
        # Model can be reloaded in serial format with: model = torch.load(filename)
        serial_nn = self.parallel_nn.buildSequentialOnRoot()
        if self.comm_lp.Get_rank() == 0:
            s_net = SerialNet(serial_nn=serial_nn, open_nn=self.open_nn, close_nn_nsp=self.close_nsp,
                              close_nn_mlm=self.close_mlm)
            s_net.eval()
            torch.save(s_net, name)

    def forward(self, x, segment_info):
        # We need the mask to be passed through, so using global allows us 
        # to pass through without interfering with the parallel_nn
        
        # Start with (bs, seq_len) -> unsqueeze -> (bs, 1, seq_len) -> repeat -> (bs, seq_len, seq_len) -> unsqueeze -> (bs, 1, seq_len, seq_len)
        global mask
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1).int()

        # by passing this through 'o' (mean composition: e.g. self.open_nn o x)
        # this makes sure this is run on only processor 0
        x = self.compose(self.open_nn, x, segment_info)
        # print("Finished open layer!")
        # print(x)

        # TODO: output mask to make sure this is fine
        x = self.parallel_nn(x)

        # Go through the classifier layers
        # Note that because of the way compose work, in order to return multiple objects, we need multiple layers
        out_nsp = self.compose(self.close_nsp, x)
        out_mlm = self.compose(self.close_mlm, x)

        return out_mlm, out_nsp
    
    def new_mask(self, x=None): 
      """
      Generates a new mask given a sample input
      """
      for layer in self.parallel_nn.local_layers.modules():
        # print(layer)
        if isinstance(layer, Dropout):
          # print('Generating')
          layer.generate_new_mask(x)

# Serial Network Class (used by the saveSerialNet functionality in ParallelNet)
class SerialNet(nn.Module):
  def __init__(self, serial_nn=None, open_nn=None, close_nn_nsp=None, close_nn_mlm=None):
    super(SerialNet, self).__init__()

    self.open_nn = open_nn

    self.serial_nn = serial_nn
    
    self.close_nn_nsp = close_nn_nsp
    self.close_nn_mlm = close_nn_mlm
  def new_mask(self, x=None): 
    """
    Generates a new mask given a sample input
    """
    for layer in self.serial_nn.modules():
      if isinstance(layer, Dropout):
        layer.generate_new_mask(x)

  def forward(self, x, segment_info):
    global mask
    mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1).int()

    x = self.open_nn(x, segment_info)
    x = self.serial_nn(x)
    out_nsp = self.close_nn_nsp(x)
    out_mlm = self.close_nn_mlm(x)

    return out_mlm, out_nsp

from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class MyBertForSequenceClassification(nn.Module):
    """
    Given a trained SerialNet, extract the relevant layers and build the classifier
    """
    def __init__(self, trained_model, num_labels: int=2, classifier_dropout: float=0.0):
        super().__init__()
        self.num_labels = num_labels
        self.problem_type = None
        self.dropout = nn.Dropout(classifier_dropout)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier = nn.Linear(1024, num_labels) # TODO: make more flexible
        self.use_return_dict = True

        # Obtain the model stuff
        self.open = trained_model.open_nn
        self.serial_nn = trained_model.serial_nn
        try:
            self.close = trained_model.close.pooler
        except:
            self.close = trained_model.close_nn_nsp.pooler

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        
        # First evaluate model
        global mask
        mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1).int()
        
        x = self.open(input_ids, token_type_ids)
        x = self.serial_nn(x)
        pooled_output = self.close(x)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # print(logits)
        loss = None
        if labels is not None:
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) 
            return ((loss,) + output) if loss is not None else output
        # print(loss)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
