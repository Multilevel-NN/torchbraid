import numpy as np
import torch

MAX_NUM = 1000000
N_DB = 100000
P_TR = .8
pos_names = '''units_tens_hundreds_thousands_ten thousands_hundred thousands'''.split('_')
fn_out_ = lambda s: f"../data/{s}.txt"

class Voc:
  def __init__(self):
    self.specials = '<pad> <unk> <sos> <eos>'.split()
    self.voc = dict(zip(self.specials, range(len(self.specials))))
    self.ctr = len(self.specials)

def generate_number(N): return np.random.randint(N)

def generate_quest(s):
  n = len(s)
  pos = np.random.randint(n)
  q = f'What is the {pos_names[pos]} digit of {s}?'
  a = s[-(pos+1)]
  return (q, a)

def generate_sample(ds, N, voc_src, voc_tgt, extend_voc):
  x = generate_number(N)
  sx = str(x)
  q, a = generate_quest(sx)
  q, a = process_qa(q, a, voc_src, voc_tgt, extend_voc)
  ds.append((q, a))

def process_qa(q_raw, a_raw, voc_src, voc_tgt, extend_voc):
  # print(q_raw, a_raw, voc_src, voc_tgt, extend_voc)
  q, a = [], []
  q.append(voc_src.voc['<sos>']); a.append(voc_src.voc['<sos>'])
  for (s, proc, voc) in [(q_raw, q, voc_src), (a_raw, a, voc_tgt)]:
    for char in s:
      if char in voc.voc: proc.append(voc.voc[char])
      elif extend_voc: 
        voc.voc[char] = voc.ctr
        voc.ctr += 1
        proc.append(voc.voc[char])
      else: proc.append(voc.voc['<unk>'])
  q.append(voc_src.voc['<eos>']); a.append(voc_src.voc['<eos>'])
  return q, a

def main():
  N_TR = int(N_DB*P_TR)

  ## Dataset
  ds_tr, ds_va = [], []
  voc_src, voc_tgt = Voc(), Voc()
  for i in range(N_TR):   # training
    generate_sample(ds_tr, MAX_NUM, voc_src, voc_tgt, extend_voc=True)
  for i in range(N_TR, N_DB): # validation
    generate_sample(ds_va, MAX_NUM, voc_src, voc_tgt, extend_voc=False)

  return ds_tr, ds_va, voc_src, voc_tgt


if __name__ == '__main__': _ = main()



































