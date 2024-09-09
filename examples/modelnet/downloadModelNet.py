import os, requests, zipfile
import torch
from torch.utils.data import Dataset

def voxelgrid_to_tensor(voxelgrid, nx):
    img = torch.zeros((nx, nx, nx))
    for vox in voxelgrid.get_voxels():
        x, y, z = vox.grid_index
        img[x, y, z] = 1

    return img

def process(datadir, processed_dir, file, nx):
    import open3d

    # open off file and convert to voxel grid
    mesh = open3d.io.read_triangle_mesh(os.path.join(datadir, file))
    bounds = (min(mesh.get_min_bound()), max(mesh.get_max_bound()))
    dx = (bounds[1] - bounds[0])/(nx-1)
    out = open3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, dx, [bounds[0], bounds[0], bounds[0]], [bounds[1], bounds[1], bounds[1]])

    # convert to tensor and save to file
    out = voxelgrid_to_tensor(out, nx)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    torch.save(out, os.path.join(processed_dir, file[:-4] + f"_{nx}.pt"))
    

# download the dataset, generate the labels, and preprocess the data
def downloadModelNet(nx=31, train=True):
    modelnet_url = "http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    if not os.path.isfile("data/ModelNet10.zip"):
        print("Downloading ModelNet10.zip (this takes a few minutes)")
        with requests.get(modelnet_url, allow_redirects=True, stream=True) as r:
            tot_size = int(r.headers["Content-Length"])
            r.raise_for_status()
            if not os.path.exists("data"):
                os.mkdir("data")
            with open("data/ModelNet10.zip", "wb") as file:
                chunk_size = 8192
                for i, chunk in enumerate(r.iter_content(chunk_size=chunk_size)):
                    file.write(chunk)
                    print(f"progress: {100*(i+1)*chunk_size/tot_size : .2f}%", end="\r")
                print('\n')
        
    if not os.path.exists("data/ModelNet10"):
        print("Extracting ModelNet10.zip")
        zfile = zipfile.ZipFile("data/ModelNet10.zip")
        zfile.extractall("data/")

    # now generate the labels and preprocess the data
    print("Generating labels and preprocessing data")
    data_dir = "data/ModelNet10/"
    processed_dir = f"data/processed/"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    classes = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
    if train:
        groups = ["train", "test"]
    else:
        groups = ["test"]

    for group in groups:
        print(f"Processing {group} data")
        with open(f"data/{group}_labels.csv", "w") as label_file:
            for i_class, name in enumerate(classes):
                class_dir = data_dir + name + f"/{group}"
                class_processed = processed_dir + name + f"/{group}"
                class_files = os.listdir(class_dir)
                num_files = len(class_files)
                for i_file, filename in enumerate(class_files):
                    if filename[0] == '.':
                        continue

                    f = os.path.join(class_processed, filename[:-4] + f"_{nx}.pt")
                    label_file.write(f"{f},{i_class}\r\n")
                    if not os.path.isfile(f):
                        # convert .off file to voxels and save to file
                        process(class_dir, class_processed, filename, nx)
                    
                    print(f"{name}: {100*(i_file+1)/num_files : .2f}%", end="\r")
                print()


class ModelNet(Dataset):
    def __init__(self, train=True, transform=None):
        self.transform = transform

        labels_file = "data/test_labels.csv"
        if train:
            labels_file = "data/train_labels.csv"
        self.labels = []
        self.ptfiles = []
        with open(labels_file, "r") as f:
            for line in f:
                fname, cl = line.split(',')
                self.labels.append(int(cl))
                self.ptfiles.append(fname)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pt_path = self.ptfiles[idx]
        out = torch.load(pt_path)
        out = out.unsqueeze(0)
        if self.transform:
           out = self.transform(out)
        return out, self.labels[idx]

class RootLoaderIter:
  def __init__(self, cnt, elmt, last_elmt):
    self.cnt = cnt
    self.elmt = elmt
    self.last_elmt = last_elmt

  def __iter__(self):
    return self

  def __next__(self):
    if self.cnt > 1:
      self.cnt -= 1
      return self.elmt
    elif self.cnt == 1:
      self.cnt -= 1
      return self.last_elmt
    else:
      raise StopIteration

def batch_size(ten):
  """
  This convenience function is used in conjunction with the data loader (and RootLoader used
  in parallel) to extract a batch size. This is then used internally within torchbraid to optimize
  the computation of shapes.
  """
  if ten.dim()==0:
    return int(ten.item())
  return ten.shape[0]

class RootLoader:
  def __init__(self, batches, items,batch_size, device):
    self.batches = batches

    elmt = torch.tensor((batch_size), device=device)
    self.elmt = (elmt, elmt)

    # the last batch size can be the actual batchsize,
    # or some smaller fraction 
    if items % batch_size==0:
      last_batch = batch_size
    else:
      last_batch = items % batch_size

    last_elmt = torch.tensor((last_batch), device=device)
    self.last_elmt = (last_elmt, last_elmt)

    self.dataset = batches * [None]

  def __iter__(self):
    return RootLoaderIter(self.batches, self.elmt,self.last_elmt)

def root_loader(rank, loader, device):
  if rank == 0:
    return loader
  batches = len(loader)
  return RootLoader(batches, len(loader.dataset), loader.batch_size, device)
    

if __name__ == "__main__":
    downloadModelNet(nx=63, train=True)
