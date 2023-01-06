import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

##
# Trivial 2-layer linear classifier
class Linear(nn.Module):		
	def __init__(self, **kwargs):
		super(Linear, self).__init__()
		self.emb = nn.Embedding(15514, 256)
		self.fc1 = nn.Linear(256, 64)
		self.fc2 = nn.Linear(64, 49)

	def forward(self, x):
		x = self.emb(x)
		x = self.fc1(x).relu()
		x = self.fc2(x)
		return x

##
# Taken from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# or also here:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.0, max_len=5000):
		import math
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.max_len = max_len

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float()
							 * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)  # shape (max_len, 1, dim)
		self.register_buffer('pe', pe)  # Will not be trained.

	def forward(self, x):
		"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embed dim]
			output: [sequence length, batch size, embed dim]
		"""
		assert x.size(0) < self.max_len, (
			f"Too long sequence length: increase `max_len` of pos encoding")
		# shape of x (len, B, dim)
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

##
# Taken from: 
# https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
class PE_Alternative(nn.Module):
	def __init__(self, d_model, dropout=0.0, max_len=5000):
		super(PE_Alternative, self).__init__()
		self.max_len = max_len

		pe = torch.zeros(max_len, d_model)
		for k in range(max_len):
			for i in np.arange(d_model//2):
				denominator = np.power(10000, 2*i/d_model)
				pe[k, 2*i] = np.sin(k/denominator)
				pe[k, 2*i + 1] = np.cos(k/denominator)

		# self.register_buffer('pe', pe)  # Will not be trained.
		self.pe = pe

	def forward(self, x):
		assert x.shape[1] < self.max_len, (
			f"Too long sequence length: increase `max_len` of pos encoding")
		# shape of x (len, B, dim)
		x = x + self.pe[:x.shape[1], :].unsqueeze(dim=0)
		return x

##
# Transformer encoder layer using their code's scheme & <i>MultiheadAttention</i>
class Transformer(nn.Module):
	def __init__(self, init_method, encoding, **kwargs):
		super(Transformer, self).__init__()
		self.encoding, self.init_method = encoding, init_method

		self.emb = nn.Embedding(15514, 128)
		self.dropout = nn.Dropout(p=.1)
		print(f'encoding {encoding}')
		self.posenc = PositionalEncoding(128) if encoding == 'Torch'\
		  else PE_Alternative(128) if encoding == 'Alternative'\
		  else Exception('encoding unknown')
		self.fc1 = nn.Linear(128, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, 49)
		self.att = nn.MultiheadAttention(
			embed_dim=128, 
			num_heads=1, 
			dropout=.3, 
			batch_first=True
		)

		self.init_params()

	def forward(self, x):
		mask = (x == 0)
		x = self.emb(x)
		x = self.dropout(x)
		x = self.posenc(x)

		for _ in range(4):
			# Encoder1DBlock
			x0 = x
			x = nn.LayerNorm(x.shape)(x)
			x, _ = self.att(x, x, x, mask)
			x = self.dropout(x)
			x = x + x0

			y = nn.LayerNorm(x.shape)(x)
			# MlpBlock
			y = self.fc1(y)
			y = nn.ELU()(y)
			y = self.dropout(y)
			y = self.fc2(y)
			y = self.dropout(y)

			x = x + y

		x = nn.LayerNorm(x.shape)(x)
		x = self.fc3(x)
		return x

	def init_params(self):
		self.apply(self._init_layers)

	def _init_layers(self, m):
		classname = m.__class__.__name__
		if isinstance(m, nn.Conv2d):
			if m.weight is not None:
				torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))

			if m.bias is not None:
				torch.nn.init.constant_(m.bias, 0)
		
		if isinstance(m, nn.BatchNorm2d):
			if m.weight is not None:
				torch.nn.init.constant_(m.weight, 1)

			if m.bias is not None:
				torch.nn.init.constant_(m.bias, 0)
		
		if isinstance(m, nn.Linear):
			if m.weight is not None:
				torch.nn.init.normal_(m.weight)

			if m.bias is not None:
				torch.nn.init.constant_(m.bias, 0)
		
		if isinstance(m, nn.MultiheadAttention):
			print(f'Init method: {self.init_method}')
			if self.init_method == 'Normal':
				m.in_proj_weight.data.normal_(mean=0.0, std=0.02)
				m.out_proj.weight.data.normal_(mean=0.0, std=0.02)
			elif self.init_method == 'Xavier':
				torch.nn.init.xavier_uniform_(m.in_proj_weight, gain=np.sqrt(2))
				torch.nn.init.xavier_uniform_(m.out_proj.weight, gain=np.sqrt(2))
			else:
				Exception('Initialization method unknown')




































