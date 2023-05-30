import os, requests, zipfile
import torch, open3d
from torch.utils.data import Dataset

def voxelgrid_to_tensor(voxelgrid, nx):
    img = torch.zeros((nx, nx, nx))
    for vox in voxelgrid.get_voxels():
        x, y, z = vox.grid_index
        img[x, y, z] = 1

    return img

def process(datadir, processed_dir, file, nx):
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
        print("Downloading ModelNet10.zip")
        r = requests.get(modelnet_url, allow_redirects=True)
        with open("data/ModelNet10.zip", "wb") as file:
            file.write(r.content)
        
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
        with open(f"data/{group}_labels.csv", "w") as label_file:
            for i, name in enumerate(classes):
                class_dir = data_dir + name + f"/{group}"
                class_processed = processed_dir + name + f"/{group}"
                for filename in os.listdir(class_dir):
                    if filename[0] == '.':
                        continue

                    f = os.path.join(class_processed, filename[:-4] + f"_{nx}.pt")
                    label_file.write(f"{f},{i}\r\n")
                    if not os.path.isfile(f):
                        # convert .off file to voxels and save to file
                        process(class_dir, class_processed, filename, nx)


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
    

if __name__ == "__main__":
    downloadModelNet(nx=31, train=False)