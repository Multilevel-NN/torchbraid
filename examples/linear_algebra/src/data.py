## The following classes, "Vocabulary" and "ParallelTextDataset",
##...were provided by the lecturer (prof. Irie, USI) to help the students.
##...Only a few modifications have been made.

import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DATASET_DIR = "../data/"

TRAIN_FILE_NAME = "train"
VALID_FILE_NAME = "interpolate"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"

TASK = 'algebra__linear_1d'

class Vocabulary:
  def __init__(
    self, pad_token="<pad>", unk_token='<unk>', eos_token='<eos>', 
    sos_token='<sos>',
  ):
    self.id_to_string = {}
    self.string_to_id = {}
    
    # add the default pad token
    self.id_to_string[0] = pad_token
    self.string_to_id[pad_token] = 0
    
    # add the default unknown token
    self.id_to_string[1] = unk_token
    self.string_to_id[unk_token] = 1
    
    # add the default unknown token
    self.id_to_string[2] = eos_token
    self.string_to_id[eos_token] = 2   

    # add the default unknown token
    self.id_to_string[3] = sos_token
    self.string_to_id[sos_token] = 3

    # shortcut access
    self.pad_id = 0
    self.unk_id = 1
    self.eos_id = 2
    self.sos_id = 3

  def __len__(self): return len(self.id_to_string)

  def add_new_word(self, string):
    self.string_to_id[string] = len(self.string_to_id)
    self.id_to_string[len(self.id_to_string)] = string

  # Given a string, return ID
  # if extend_vocab is True, add the new word
  def get_idx(self, string, extend_vocab=False):
    if string in self.string_to_id:
      return self.string_to_id[string]
    elif extend_vocab:  # add the new word
      self.add_new_word(string)
      return self.string_to_id[string]
    else:
      return self.unk_id

# Read the raw txt files and generate parallel text dataset:
# self.data[idx][0] is the tensor of source sequence
# self.data[idx][1] is the tensor of target sequence
# See examples in the cell below.
class ParallelTextDataset(Dataset):
  def __init__(
    self, src_file_path, tgt_file_path, src_vocab=None, tgt_vocab=None, 
    extend_vocab=False, device='cuda',
  ):
    (self.data, self.src_vocab, self.tgt_vocab, self.src_max_seq_length,
     self.tgt_max_seq_length) = self.parallel_text_to_data(
      src_file_path, tgt_file_path, src_vocab, tgt_vocab, extend_vocab,
      device,
    )

  def __getitem__(self, idx): return self.data[idx]
  def __len__(self): return len(self.data)

  def parallel_text_to_data(
    self, src_file, tgt_file, src_vocab=None, tgt_vocab=None, 
    extend_vocab=False, device='cuda',
  ):
    # Convert paired src/tgt texts into torch.tensor data.
    # All sequences are padded to the length of the longest sequence
    # of the respective file.

    assert os.path.exists(src_file)
    assert os.path.exists(tgt_file)

    if src_vocab is None:
      src_vocab = Vocabulary()

    if tgt_vocab is None:
      tgt_vocab = Vocabulary()
    
    data_list = []
    # Check the max length, if needed construct vocab file.
    src_max = 0
    with open(src_file, 'r') as text:
      for line in text:
        tokens = list(line)[:-1]  # remove line break
        length = len(tokens)
        if src_max < length:
          src_max = length

    tgt_max = 0
    with open(tgt_file, 'r') as text:
      for line in text:
        tokens = list(line)[:-1]
        length = len(tokens)
        if tgt_max < length:
          tgt_max = length
    tgt_max += 2  # add for begin/end tokens

    ## So that torch.stack((x, y)) in enc-dec Steplayers
    # src_max = max(src_max, tgt_max)
    # tgt_max = max(src_max, tgt_max) + 1
    src_max = 100
    tgt_max = src_max + 1

    src_pad_idx = src_vocab.pad_id
    tgt_pad_idx = tgt_vocab.pad_id

    tgt_eos_idx = tgt_vocab.eos_id
    tgt_sos_idx = tgt_vocab.sos_id

    # Construct data
    src_list = []
    print(f"Loading source file from: {src_file}")
    with open(src_file, 'r') as text:
      for line in text:#tqdm(text):
        seq = []
        tokens = list(line)[:-1]
        for token in tokens:
          seq.append(src_vocab.get_idx(
            token, extend_vocab=extend_vocab))
        var_len = len(seq)
        var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
        # padding
        new_seq = var_seq.data.new(src_max).fill_(src_pad_idx)
        new_seq[:var_len] = var_seq
        src_list.append(new_seq)

    tgt_list = []
    print(f"Loading target file from: {tgt_file}")
    with open(tgt_file, 'r') as text:
      for line in text:#tqdm(text):
        seq = []
        tokens = list(line)[:-1]
        # append a start token
        seq.append(tgt_sos_idx)
        for token in tokens:
          seq.append(tgt_vocab.get_idx(
            token, extend_vocab=extend_vocab))
        # append an end token
        seq.append(tgt_eos_idx)

        var_len = len(seq)
        var_seq = torch.tensor(seq, device=device, dtype=torch.int64)

        # padding
        new_seq = var_seq.data.new(tgt_max).fill_(tgt_pad_idx)
        new_seq[:var_len] = var_seq
        tgt_list.append(new_seq)

    # src_file and tgt_file are assumed to be aligned.
    assert len(src_list) == len(tgt_list)
    for i in range(len(src_list)):
      data_list.append((src_list[i], tgt_list[i]))

    print("Done.")
      
    return data_list, src_vocab, tgt_vocab, src_max, tgt_max

def obtain_data(batch_size, device, debug):
  src_file_path = DATASET_DIR + f"{TASK}/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
  tgt_file_path = DATASET_DIR + f"{TASK}/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

  if debug:
    src_file_path = DATASET_DIR + f"{TASK}_small/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
    tgt_file_path = DATASET_DIR + f"{TASK}_small/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

  training_data_set = ParallelTextDataset(
    src_file_path, tgt_file_path, extend_vocab=True, device=device,
  )

  # get the vocab
  src_vocab = training_data_set.src_vocab
  tgt_vocab = training_data_set.tgt_vocab

  src_file_path = DATASET_DIR + f"{TASK}/{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
  tgt_file_path = DATASET_DIR + f"{TASK}/{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

  validation_data_set = ParallelTextDataset(
      src_file_path, tgt_file_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
      extend_vocab=False, device=device,
  )

  training_data_loader = DataLoader(
    dataset=training_data_set  , batch_size=batch_size, shuffle=True, 
    drop_last=True,
  )
  validation_data_loader = DataLoader(
    dataset=validation_data_set, batch_size=batch_size, shuffle=True, 
    drop_last=True,
  )

  data_sets = {
    'training': training_data_set,
    'validation': validation_data_set,
  }
  data_loaders = {
    'training': training_data_loader,
    'validation': validation_data_loader,
  }
  source_vocabulary = src_vocab
  target_vocabulary = tgt_vocab
  source_max_length = training_data_set.src_max_seq_length
  target_max_length = training_data_set.tgt_max_seq_length

  return (
    data_sets, data_loaders, source_vocabulary, target_vocabulary,
    source_max_length, target_max_length,
  )




