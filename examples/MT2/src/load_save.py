import torch

def load(model, loading_path, optimizer=None):
  loading_path_cp1 = f'{loading_path}_cp1.pt'
  loading_path_cp2 = f'{loading_path}_cp2.pt'

  loading_copy = None
  try: 
    checkpoint = torch.load(loading_path_cp1)
  except:
    try: 
      checkpoint = torch.load(loading_path_cp2)
    except: raise Exception(f'Could not load the model from {loading_path}')
    else: loading_copy = 2
  else: loading_copy = 1

  model.load_state_dict(checkpoint['model_state'])

  optimizer_loaded = False
  if optimizer is not None and \
     (optimizer_state := checkpoint.get('optimizer_state', None)):
    optimizer.load_state_dict(optimizer_state)
    optimizer_loaded = True

  print(f'Model ' + ('and optimizer ' if optimizer_loaded else '')
      + f'loaded from {loading_path}_cp{loading_copy}.pt')

def save(model, processed_tokens_ctr, saving_path, optimizer=None):
  checkpoint = {
                        'model_state': model.state_dict(),
        'num_nonpad_processed_tokens': processed_tokens_ctr.non_pad,
         'num_total_processed_tokens': processed_tokens_ctr.total,
  }
  if optimizer is not None: 
    checkpoint['optimizer_state'] = optimizer.state_dict(),

  saving_path_cp1 = f'{saving_path}_cp1.pt'
  saving_path_cp2 = f'{saving_path}_cp2.pt'

  torch.save(checkpoint, saving_path_cp1)
  torch.save(checkpoint, saving_path_cp2)




