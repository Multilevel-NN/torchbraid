import numpy as np
import scipy
# import scipy.spatial
from scipy import interpolate
import string
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import matplotlib.pyplot as plt


class list2np(object):
    def __init__(self):
        pass

    def __call__(self, *args):
        args_array = ()
        for arg in args:
            args_array += (np.asarray(arg),)
        return args_array

    def __repr__(self):
        return self.__class__.__name__ + '()'
def saveMesh(xn, faces, pos, i=0, vmax=None, vmin=None):
    # xn of shape [points, features]
    # if with our net dim = 2 else 1
    if 1==1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(pos[:, 0].clone().detach().cpu().numpy(), pos[:, 1].clone().detach().cpu().numpy(),
                       pos[:, 2].clone().detach().cpu().numpy(),
                       c=xn.squeeze(0).norm(dim=1).clone().detach().cpu().numpy(), vmin=0.0, vmax=1.0)
        fig.colorbar(p)
        plt.savefig(
            "/yourpath/plots_wave/xn_norm_wave_layer_" + str(i))
        plt.close()

    mesh = trimesh.Trimesh(vertices=pos, faces=faces, process=False)
    colors = xn.squeeze(0).norm(dim=1).clone().detach().cpu().numpy() # xn.squeeze(0).clone().detach().cpu().numpy()[:, 0]
    if vmax is not None:
        colors[colors < vmin] = vmin
        colors[colors > vmax] = vmax
        add = np.array([[vmax], [vmin]], dtype=np.float).squeeze()
    else:
        colors[colors < 0.0] = 0.0
        colors[colors > 1.0] = 1.0
        add = np.array([[1.0], [0.0]], dtype=np.float).squeeze()
    vect_col_map2 = trimesh.visual.color.interpolate(colors,
                                                     color_map='jet')

    colors = np.concatenate((add, colors), axis=0)
    vect_col_map = trimesh.visual.color.interpolate(colors,
                                                    color_map='jet')
    vect_col_map = vect_col_map[2:, :]
    if xn.shape[0] == mesh.vertices.shape[0]:
        mesh.visual.vertex_colors = vect_col_map
    elif xn.shape[0] == mesh.faces.shape[0]:
        mesh.visual.face_colors = vect_col_map
        smooth = False

    trimesh.exchange.export.export_mesh(mesh,
                                        "/yourpath/plots_wave/xn_norm_wave_layer_" + str(
                                            i) + ".ply", "ply")

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()