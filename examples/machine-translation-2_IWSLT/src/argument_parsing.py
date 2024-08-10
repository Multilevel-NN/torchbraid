import argparse
import datetime as dt
import numpy    as np
import os
from   types    import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument('--batch_first'             , action='store_true'       )
parser.add_argument('--batch_size'              , type=int  , default=  16  )  # 32 max
parser.add_argument('--d_model'                 , type=int  , default= 512  )
parser.add_argument('--debug'                   , action='store_true'       )
parser.add_argument('--dim_feedforward'         , type=int  , default=argparse.SUPPRESS)
parser.add_argument('--do_sample'               , action='store_true'       )
parser.add_argument('--download_tokenizers'     , action='store_true'       )
parser.add_argument('--drop_last'               , action='store_true'       )
parser.add_argument('--dropout'                 , type=float, default=   0.1)
parser.add_argument('--initialize_parameters'   , action='store_true'       )
parser.add_argument('--length_penalty'          , type=float, default=   1. )
parser.add_argument('--load_model'              , type=str  , default=argparse.SUPPRESS)
parser.add_argument('--log_dir'                 , type=str  , default= '.'  )
# parser.add_argument('--lr'                    , type=float, default=      )
# parser.add_argument('--momentum'              , type=float, default=      )
parser.add_argument('--monitoring_num_batches'  , type=int  , default=argparse.SUPPRESS, nargs='?', const=2000)
parser.add_argument('--monitoring_period'       , type=int  , default=argparse.SUPPRESS, nargs='?', const=  10)  # mins
parser.add_argument('--monitor_validation_bleu' , action='store_true'       )
parser.add_argument('--nhead'                   , type=int  , default=   8  )
parser.add_argument('--norm_first'              , action='store_true'       )
parser.add_argument('--num_beams'               , type=int  , default=   1  )
parser.add_argument('--num_layers'              , type=int  , default=   6  )
parser.add_argument('--num_return_sequences'    , type=int  , default=   1  )
parser.add_argument('--num_epochs'              , type=int  , default=  20  )
parser.add_argument('--num_warmup_steps'        , type=int  , default=4000  )
parser.add_argument('--patience'                , type=int  , default=  30  )
parser.add_argument('--seed'                    , type=int  , default=   0  )
parser.add_argument('--skip_tensorboard'        , action='store_true'       )
parser.add_argument('--skip_training'           , action='store_true'       )
parser.add_argument('--top_k'                   , type=int  , default=   1  )
parser.add_argument('--top_p'                   , type=int  , default=   1. )
args = parser.parse_args()

def get_config():
  retouch_args(args)
  config_keys = {
           'data': ('batch_size', 'download_tokenizers', 'drop_last'),
     'generation': ('do_sample', 'length_penalty', 'num_beams', 
                    'num_return_sequences', 'top_k', 'top_p'),
          'model': ('d_model', 'nhead', 'num_encoder_layers', 
                    'num_decoder_layers', 'dim_feedforward', 'dropout', 
                    'batch_first', 'norm_first', 'initialize_parameters'),
      'optimizer': ('num_warmup_steps',),#('lr', 'momentum'),
          'other': ('debug', 'loading_path', 'run_identifier', 'saving_path', 
                    'seed'),
    'tensorboard': ('log_dir', 'skip_tensorboard'),#, 'filename_suffix'),
       'training': ('monitoring_frequency', 'monitor_validation_bleu', 
                    'num_epochs', 'skip_training', 'patience'),
  }
  config_dict = {
    k: SimpleNamespace(**{arg_nm: getattr(args, arg_nm) for arg_nm in v})#args.__dict__[arg_nm] for arg_nm in v}
    for (k, v) in config_keys.items()
  }
  config = SimpleNamespace(**config_dict)
  print_config(config)
  return config

def print_config(config):
  max_key_length = max(map(len, config.__dict__.keys()))

  s = set()
  for k, namespace in config.__dict__.items():
    for k2 in namespace.__dict__.keys(): s.add(k2)
  max_key2_length = max(map(len, s))

  title = 'CONFIG:'
  print(title)

  for k, namespace in sorted(config.__dict__.items()):
    print(f'{k :>{len(title) + max_key_length}}:')

    for k2, v in sorted(namespace.__dict__.items()):
      print(f'{k2 :>{len(title) + max_key_length + 2 + max_key2_length}}: {v}')
  print()

def retouch_args(args):
  ## Modify/complement args
  if args.debug:
    args.batch_size = 5
    args.d_model    = 8
    args.nhead      = 2
    # args.monitoring_num_batches = 80
    # if hasattr(args, 'monitoring_period'): 
    #   _ = args.__dict__.pop('monitoring_period')
    args.skip_tensorboard = True

  num_layers = args.num_encoder_layers = \
               args.num_decoder_layers = args.__dict__.pop('num_layers')

  if not hasattr(args, 'dim_feedforward'): 
    args.dim_feedforward = 4 * args.d_model

  assert not(all(map(
    lambda arg: hasattr(args, arg), ['monitoring_num_batches', 'monitoring_period']
  ))), "You must input at most one of the following arguments: monitoring_num_batches, monitoring_period"
  args.monitoring_frequency = \
    (args.monitoring_num_batches, 'batches') \
                               if hasattr(args, 'monitoring_num_batches') else \
    (args.monitoring_period, 'minutes') \
                               if hasattr(args, 'monitoring_period') else \
    (  # unorthodox trick
      list(filter(lambda action: action.dest == 'monitoring_period',
                  parser.__dict__['_actions']))[0].const, 
     'minutes',
    )

  # args.filename_suffix = '_' + '_'.join([
  #   'bf' + str(args.batch_first            )[0]      ,
  #   'bs' + str(args.batch_size             ).zfill(4),
  #   'd'  + str(args.d_model                ).zfill(4),
  #   'ff' + str(args.dim_feedforward        ).zfill(4),
  #   'do' + str(args.dropout                )         ,
  #   'mf' + str(args.monitoring_frequency[0]).zfill(5) + args.monitoring_frequency[1][0],
  #   'H'  + str(args.nhead                  ).zfill(2),
  #   'N'  + str(     num_layers             ).zfill(4),
  #   'ep' + str(args.num_epochs             ).zfill(3),
  #   's'  + str(args.seed                   ).zfill(1),
  # ])
  args.run_identifier = '_'.join([dt.datetime.now().strftime('%Y%m%d%H%M%S'), 
                                 str(np.random.randint(0, 1000000000))])
  args.log_dir = os.path.join(args.log_dir, 
                              f'tensorboard_runs_{args.run_identifier}')
  if not args.skip_tensorboard and not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

  stored_models_dir = os.path.join('..', 'stored_models')
  args.saving_path = os.path.join(stored_models_dir, 
                                  f'checkpoint_{args.run_identifier}')
  args.loading_path = os.path.join(stored_models_dir, 
                                  f'checkpoint_{args.load_model}') \
                      if hasattr(args, 'load_model') else ''
    



