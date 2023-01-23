import os
import numpy as np
import torch

def download_UCI_Data(example_path):
  """
  Download and unzip the UCI HAR dataset to the UCI_HAR_Recurrent folder

  This creates a "UCI HAR Dataset" folder in the examples/UCI_HAR_Recurrent folder
  which contains the data necessary for training the example GRU network.

  If the dataset has already been download, this function does not download it again.
  """
  dataset_path = example_path + '/UCI HAR Dataset'

  # If the data hasn't been downloaded yet
  if not os.path.exists(dataset_path):
    import zipfile
    import urllib
    import shutil

    print('downloading UCI HAR dataset...')

    # Link to the UCI database
    download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    # Download the zip
    download_filename, _ = urllib.request.urlretrieve(download_url, example_path + '/UCIHARDataset.zip')
    # Upzip the file
    with zipfile.ZipFile(download_filename, 'r') as downloaded_zip:
        downloaded_zip.extractall(path=example_path)
    # Do some cleanup
    os.remove(download_filename)
    shutil.rmtree(example_path + '/__MACOSX')

    print('download complete')


def load_data(train, example_path):
  """
  Load the data from the UCI HAR data set.
  train: if true the training data is loaded, if false
         the test data is loaded.
  path: path to the data set directory.
  returns: An x,y pair where x is a rank 3
           array (samples,seq length,data size)
  """
  path = example_path + '/UCI HAR Dataset'

  if train:
    type_str = 'train'
  else:
    type_str = 'test'

  num_classes = 6
  d_path = path + '/' + type_str
  i_path = d_path + '/Inertial Signals/'

  # load label data
  y = np.loadtxt('%s/y_%s.txt' % (d_path,type_str))

  y_data = torch.tensor([int(y[i]-1) for i in range(y.shape[0])], dtype=torch.long)

  # load feature data
  body_x = np.loadtxt('%s/body_acc_x_%s.txt' % (i_path,type_str))
  body_y = np.loadtxt('%s/body_acc_y_%s.txt' % (i_path,type_str))
  body_z = np.loadtxt('%s/body_acc_z_%s.txt' % (i_path,type_str))

  gyro_x = np.loadtxt('%s/body_gyro_x_%s.txt' % (i_path,type_str))
  gyro_y = np.loadtxt('%s/body_gyro_y_%s.txt' % (i_path,type_str))
  gyro_z = np.loadtxt('%s/body_gyro_z_%s.txt' % (i_path,type_str))

  totl_x = np.loadtxt('%s/total_acc_x_%s.txt' % (i_path,type_str))
  totl_y = np.loadtxt('%s/total_acc_y_%s.txt' % (i_path,type_str))
  totl_z = np.loadtxt('%s/total_acc_z_%s.txt' % (i_path,type_str))

  x_data = np.stack([body_x,body_y,body_z,
                     gyro_x,gyro_y,gyro_z,
                     totl_x,totl_y,totl_z],axis=2)

  return torch.Tensor(x_data),y_data

class ParallelRNNDataLoader(torch.utils.data.DataLoader):
  """ Custom DataLoader class for distributing sequences across ranks """
  def __init__(self, comm, dataset, batch_size, shuffle=False):
    self.dataset = dataset
    self.shuffle = shuffle

    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    if comm.Get_rank()==0:
      # build a gneerator to build one master initial seed
      serial_generator = torch.Generator()
      self.initial_seed = serial_generator.initial_seed()
    else:
      self.serial_loader = None
      self.initial_seed = None
    # if rank==0

    # distribute the initial seed
    self.initial_seed = comm.bcast(self.initial_seed,root=0)

    # break up sequences
    x_block = [[] for n in range(num_procs)]
    y_block = [[] for n in range(num_procs)]
    if rank == 0:
      sz = len(dataset)
      for i in range(sz):
        x,y = dataset[i]
        x_split = torch.chunk(x, num_procs, dim=0)
        y_split = num_procs*[y]

        for p,(x_in,y_in) in enumerate(zip(x_split,y_split)):
          x_block[p].append(x_in)
          y_block[p].append(y_in)

      for p,(x,y) in enumerate(zip(x_block,y_block)):
        x_block[p] = torch.stack(x)
        y_block[p] = torch.stack(y)

    x_local = comm.scatter(x_block,root=0)
    y_local = comm.scatter(y_block,root=0)

    self.parallel_dataset = torch.utils.data.TensorDataset(x_local, y_local)

    # now setup the parallel loader
    if shuffle==True:
      parallel_generator = torch.Generator()
      parallel_generator.manual_seed(self.initial_seed)
      sampler = torch.utils.data.sampler.RandomSampler(self.parallel_dataset,generator=parallel_generator)
      torch.utils.data.DataLoader.__init__(self,self.parallel_dataset,batch_size=batch_size,sampler=sampler)
    else:
      torch.utils.data.DataLoader.__init__(self,self.parallel_dataset,batch_size=batch_size,shuffle=False)
