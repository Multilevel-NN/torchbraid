import numpy as np
import matplotlib.pyplot as plt
import torch

from input_pipeline import sentences_from_conll_data, PAD_ID

class Dataset(torch.utils.data.Dataset):
	def __init__(self, **params):
		self.data = self.create_dataset(params)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

	def create_dataset(self, params):
		filename = params['filename']
		vocabs = params['vocabs']
		attributes_input = params['attributes_input']
		attributes_target = params['attributes_target']
		batch_size = params['batch_size']
		bucket_size = params['bucket_size']

		input_generator = sentences_from_conll_data(
			filename, vocabs, attributes_input, max_sentence_length=bucket_size)
		target_generator = sentences_from_conll_data(
			filename, vocabs, attributes_target, max_sentence_length=bucket_size)

		data = []
		ctr = 0
		for inputs in input_generator:
			sentence_nopad = torch.tensor(inputs).long()#, dtype=torch.int32)
			sentence = self.fill_till_(sentence_nopad, bucket_size, PAD_ID)

			targets = next(target_generator)
			labels_nopad = torch.tensor(targets).long()#, dtype=torch.int32)
			labels = self.fill_till_(labels_nopad, bucket_size, PAD_ID)

			data.append((sentence, labels))

			ctr += 1

		print(f'#inputs: {ctr}')

		return data

	def fill_till_(self, tensor, new_size, value):
		new_tensor = torch.cat(
			(
				tensor, 
				torch.full((new_size - len(tensor),), value).long()#, dtype=torch.int32)
			),
			axis=0
		)
		return new_tensor


def obtain_dataset(**params):
	dataset = Dataset(**params)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
	return dataset, dataloader









































