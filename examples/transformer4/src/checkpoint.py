import os
import re 
import torch

dir_outputs = os.path.join('..', 'outputs')
dir_models_ = lambda instance: os.path.join(dir_outputs, instance, 'models')

instance = None
l_outputs = sorted(os.listdir(dir_outputs))
for nm in l_outputs:
  if nm.startswith('cont'): instance = nm; break
assert instance is not None
dir_models = dir_models_(instance)

def load_model(model):
  nm_model = None
  l_models = sorted(os.listdir(dir_models), reverse=True)
  for nm in l_models: 
    if nm.startswith('model') and nm.endswith('.pt'): nm_model = nm; break
  assert nm_model is not None

  path = os.path.join(dir_models, nm_model)
  # model = torch.jit.load(path)
  model.load_state_dict(torch.load(path))
  print(f'Model "{nm_model}" loaded.')
  return model

def save_model(model, datetime):
  fn = os.path.join(dir_models, f'model_{datetime}.pt')
  # torch.jit.script(model).save(fn)
  torch.save(model.state_dict(), fn)


























