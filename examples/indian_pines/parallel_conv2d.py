# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torchbraid
import torchbraid.utils
import torch.nn as nn
import torch.nn.functional as F
import statistics as stats
# from torchsummary import summary
# Numpy, scipy, scikit-image, spectral
import numpy as np
# Visualization
import seaborn as sns
import visdom
import sys
from mpi4py import MPI
import os
from deephyperx_master.utils import convert_to_color_, convert_from_color_,\
    display_dataset, explore_spectrums, plot_spectrums,\
    sample_gt, get_device
from deephyperx_master.datasets import get_dataset, HyperX, DATASETS_CONFIG
from deephyperx_master.models import get_model
# from deephyperx_master.models import train as hypertrain
# from deephyperx_master.models import test as test_prob
import argparse
from timeit import default_timer as timer
# from torch.nn import init


def convert_to_color(x, palette):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x, invert_palette):
    return convert_from_color_(x, palette=invert_palette)

def root_print(rank,s):
  if rank==0:
    print(s)


def compute_levels(num_steps, min_coarse_size, cfactor):
    from math import log, floor

    # we want to find $L$ such that ( max_L min_coarse_size*cfactor**L <= num_steps)
    levels = floor(log(num_steps / min_coarse_size, cfactor)) + 1

    if levels < 1:
        levels = 1
    return levels


def train(rank, args, model, train_loader, optimizer, epoch, compose, criterion):
    model.train()
    # criterion = nn.CrossEntropyLoss() #FIXME remove
    total_time = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = timer()
        optimizer.zero_grad()
        output = model(data)
        loss = compose(criterion, output, target)
        loss.backward()
        stop_time = timer()
        optimizer.step()
        # MPI.COMM_WORLD.Barrier()
        # print("Rank in train, batch idx ", MPI.COMM_WORLD.Get_rank(), batch_idx, data.shape, target.shape)

        total_time += stop_time - start_time
        if batch_idx % args.log_interval == 0:
            root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), total_time / (batch_idx + 1.0)))

        root_print(rank, 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
            epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                   100. * (batch_idx + 1) / len(train_loader), loss.item(), total_time / (batch_idx + 1.0)))


def test(rank, model, test_loader, compose, criterion, hyperparams, img):
  model.eval()
  test_loss = 0
  correct = 0
  # criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data, target
      output = model(data)
      test_loss += compose(criterion,output,target).item()

      output = MPI.COMM_WORLD.bcast(output,root=0)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  root_print(rank,'\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

# def test(rank, model, test_loader, compose, criterion, hyperparams, img):
#     model.eval()
#     ignored_labels = test_loader.dataset.ignored_labels
#     patch_size = hyperparams['patch_size']
#     center_pixel = hyperparams['center_pixel']
#     batch_size, device = hyperparams['batch_size'], hyperparams['device']
#     n_classes = hyperparams['n_classes']
#     probs = np.zeros(img.shape[:2] + (n_classes,))
#     test_loss = 0
#     correct = 0
#     total = 0
#     # criterion = nn.CrossEntropyLoss()
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data, target
#             output = model(data)
#             test_loss += compose(criterion, output, target).item()
#
#             print("rank ", rank, " before output ", output.shape)
#             output = MPI.COMM_WORLD.bcast(output, root=0)
#             print("rank ", rank, " after output ", output.shape)
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#
#             #commenting out this line causes the program to hang.
#             #so maybe the hanging is caused by something after this.
#             # correct += pred.eq(target.view_as(pred)).sum().item()
#             # print("rank ", rank, " correct ", correct)
#             # if rank == 0:
#             # _, pred = torch.max(output, dim=1) #This line causes mpi to hang
#             print("rank ", rank, " pred ,", pred.shape)
#             print("rank ", rank, " target ", target.shape)
#             break #with this the program no longer hangs. so the loop is what is making it take so long
#             # for out, tg in zip(pred.view(-1), target.view(-1)):
#             #     if out.item() in ignored_labels:
#             #         continue
#             #     else:
#             #         print("correct")
#             #         correct += out.item() == tg.item()
#             #         total += 1
#             #         print("num correct ", correct, total)
#
#                 # indices = [b[1:] for b in batch]
#                 # if patch_size == 1 or center_pixel:
#                 #     output = output.numpy()
#                 # else:
#                 #     output = np.transpose(output.numpy(), (0, 2, 3, 1))
#                 # for (x, y, w, h), out in zip(indices, output):
#                 #     if center_pixel:
#                 #         probs[x + w // 2, y + h // 2] += out
#                 #     else:
#                 #         probs[x:x + w, y:y + h] += out
#
#     # print("done with loop")
#     # test_loss /= len(test_loader.dataset)
#
#     # root_print(rank, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#     #     test_loss, correct, total,
#     #     100. * correct / total))
#
#     root_print(rank, 'done')


class ParallelNet(nn.Module):
    def __init__(self, out_channels=12, local_steps=8, Tf=1.0, max_levels=1, max_iters=1, print_level=0):
        super(ParallelNet, self).__init__()

        step_layer = lambda: StepLayer(out_channels)

        # print("Rank in parallel net ", MPI.COMM_WORLD.Get_rank())
        self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD, step_layer, local_steps, Tf, max_levels=max_levels, max_iters=max_iters)
        self.parallel_nn.setPrintLevel(print_level)
        self.parallel_nn.setCFactor(4)
        self.parallel_nn.setSkipDowncycle(True)
        self.parallel_nn.setNumRelax(1)  # FCF elsewehre
        self.parallel_nn.setNumRelax(0, level=0)  # F-Relaxation on the fine grid

        # this object ensures that only the LayerParallel code runs on ranks!=0
        compose = self.compose = self.parallel_nn.comp_op()

        # by passing this through 'compose' (mean composition: e.g. OpenLayer o channels)
        # on processors not equal to 0, these will be None (there are no parameters to train there)
        self.open_nn = compose(OpenLayer, out_channels)
        self.close_nn = compose(CloseLayer, out_channels)

    def forward(self, x):
        # by passing this through 'o' (mean composition: e.g. self.open_nn o x)
        # this makes sure this is run on only processor 0

        x = self.compose(self.open_nn, x)
        x = self.parallel_nn(x)
        x = self.compose(self.close_nn, x)

        return x
# end ParallelNet


class OpenLayer(nn.Module):
  def __init__(self,channels):
    super(OpenLayer, self).__init__()
    kernel_size = 3
    self.conv = nn.Conv2d(200, channels, kernel_size=kernel_size, padding=1)

  def forward(self, x):
    return F.relu(self.conv(x))


class CloseLayer(nn.Module):
  def __init__(self,channels):
    super(CloseLayer, self).__init__()
    ker_width = 3
    n_classes = 17

    self.features_size = 15680 #FIXME make this dynamic
    self.fc1 = nn.Linear(self.features_size, 128)
    self.fc2 = nn.Linear(128, n_classes)

  def forward(self, x):
    x = torch.flatten(x, 1)
    # x = x.view(-1, self.features_size)
    x = self.fc1(x)
    x = F.relu(x)
    # output = self.fc2(x) #FIXME this is different that serial example
    output = F.log_softmax(self.fc2(x), dim=1)
    return output
# end layer

class StepLayer(nn.Module):
  def __init__(self,channels):
    super(StepLayer, self).__init__()
    kernel_size = 3
    self.conv1 = nn.Conv2d(channels,channels, kernel_size=kernel_size, padding=1)
    self.conv2 = nn.Conv2d(channels,channels, kernel_size=kernel_size, padding=1)

  def forward(self, x):
    return F.relu(self.conv2(F.relu(self.conv1(x))))


def main():

    dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

    # Argument parser for CLI interaction
    parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                                 " various hyperspectral datasets")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default="IndianPines", choices=dataset_names,
                        help="Dataset to use.")
    parser.add_argument('--model', type=str, default="gary",
                        help="Model to train. Available:\n"
                        "SVM (linear), "
                        "SVM_grid (grid search on linear, poly and RBF kernels), "
                        "baseline (fully connected NN), "
                        "hu (1D CNN), "
                        "hamida (3D CNN + 1D classifier), "
                        "lee (3D FCN), "
                        "chen (3D CNN), "
                        "li (3D CNN), "
                        "he (3D CNN), "
                        "luo (3D CNN), "
                        "sharma (2D CNN), "
                        "boulch (1D semi-supervised CNN), "
                        "liu (3D semi-supervised CNN), "
                        "mou (1D RNN)")
    parser.add_argument('--folder', type=str, help="Folder where to store the "
                        "datasets (defaults to the current working directory).",
                        default="/Users/gjsaave/parnets/parnets/gjsaave/data/")
    parser.add_argument('--cuda', type=int, default=-1,
                        help="Specify CUDA device (defaults to -1, which learns on CPU)")
    parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
    parser.add_argument('--restore', type=str, default=None,
                        help="Weights to use for initialization, e.g. a checkpoint")

    # Dataset options
    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument('--training_sample', type=float, default=0.9,
                        help="Percentage of samples to use for training. careful - this should be a float representing the proportion of training data ")
    group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                        " (random sampling or disjoint, default: random)",
                        default='random')
    # Training options
    group_train = parser.add_argument_group('Training')
    group_train.add_argument('--epoch', type=int, help="Training epochs (optional, if"
                        " absent will be set by the model)")
    group_train.add_argument('--patch_size', type=int,
                        help="Size of the spatial neighbourhood (optional, if "
                        "absent will be set by the model)")
    group_train.add_argument('--lr', type=float,
                        help="Learning rate, set by the model if not specified.")
    group_train.add_argument('--class_balancing', action='store_true',
                        help="Inverse median frequency class balancing (default = False)")
    group_train.add_argument('--batch_size', type=int, default=50,
                        help="Batch size (optional, if absent will be set by the model")
    group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")
    # Data augmentation parameters
    group_da = parser.add_argument_group('Data augmentation')
    group_da.add_argument('--flip_augmentation', action='store_true',
                        help="Random flips (if patch_size > 1)")
    group_da.add_argument('--radiation_augmentation', action='store_true',
                        help="Random radiation noise (illumination)")
    group_da.add_argument('--mixture_augmentation', action='store_true',
                        help="Random mixes between spectra")

    parser.add_argument('--with_exploration', action='store_true',
                        help="See data exploration visualization and run visdom")

    # group_viz = parser.add_argument_group('Viz')
    # group_viz.add_argument('--run_visdom', type=bool, default=False,help="")
    parser.add_argument('--force-lp', action='store_true', default=False,
                        help='Use layer parallel even if there is only 1 MPI rank')
    parser.add_argument('--lp-levels', type=int, default=3, metavar='N',
                        help='Layer parallel levels (default: 3)')
    parser.add_argument('--lp-iters', type=int, default=2, metavar='N',
                        help='Layer parallel iterations (default: 2)')
    parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                        help='Layer parallel print level default: 0)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--steps', type=int, default=4, metavar='N',
                        help='Number of times steps in the resnet layer (default: 4)')

    rank = MPI.COMM_WORLD.Get_rank()
    procs = MPI.COMM_WORLD.Get_size()
    args = parser.parse_args()

    CUDA_DEVICE = get_device(args.cuda)

    SAMPLE_PERCENTAGE = args.training_sample
    FLIP_AUGMENTATION = args.flip_augmentation
    RADIATION_AUGMENTATION = args.radiation_augmentation
    MIXTURE_AUGMENTATION = args.mixture_augmentation
    DATASET = args.dataset
    MODEL = args.model
    # N_RUNS = args.runs
    PATCH_SIZE = args.patch_size
    DATAVIZ = args.with_exploration
    FOLDER = args.folder
    SAMPLING_MODE = args.sampling_mode
    CHECKPOINT = args.restore
    LEARNING_RATE = args.lr
    CLASS_BALANCING = args.class_balancing
    TEST_STRIDE = args.test_stride

    # some logic to default to Serial if on one processor,
    # can be overriden by the user to run layer-parallel
    if args.force_lp:
        force_lp = True
    elif procs > 1:
        force_lp = True
    else:
        force_lp = False

    torch.manual_seed(torchbraid.utils.seed_from_rank(args.seed, rank))

    if args.lp_levels == -1:
        min_coarse_size = 7
        args.lp_levels = compute_levels(args.steps, min_coarse_size, 4)

    local_steps = int(args.steps / procs)
    if args.steps % procs != 0:
        root_print(rank, 'Steps must be an even multiple of the number of processors: %d %d' % (args.steps, procs))
        sys.exit(0)

    if DATAVIZ:
        viz = visdom.Visdom(env=DATASET + ' ' + MODEL)
        if not viz.check_connection:
            print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
    else:
        viz = None

    hyperparams = vars(args)
    # Load the dataset
    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET,
                                                                   FOLDER)

    # Number of classes
    N_CLASSES = len(LABEL_VALUES)
    # Number of bands (last dimension of the image tensor)
    N_BANDS = img.shape[-1]

    if palette is None:
        # Generate color palette
        palette = {0: (0, 0, 0)}
        for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    invert_palette = {v: k for k, v in palette.items()}


    # Instantiate the experiment based on predefined networks
    hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    if DATAVIZ:
        display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)

    color_gt = convert_to_color(gt, palette)

    if DATAVIZ:
        # Data exploration : compute and show the mean spectrums
        mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES, viz,
                                           ignored_labels=IGNORED_LABELS)
        plot_spectrums(mean_spectrums, viz, title='Mean spectrum/class')

    if force_lp:
        root_print(rank, 'Using ParallelNet')
        _, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams) #FIXME get opt and hyperparams normally
        model = ParallelNet(out_channels=20, local_steps=local_steps,
                            max_levels=args.lp_levels,
                            max_iters=args.lp_iters,
                            print_level=args.lp_print)
        compose = model.compose
    else:
        root_print(rank, 'Using SerialNet')
        # model = SerialNet(channels=args.channels, local_steps=local_steps)
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
        compose = lambda op, *p: op(*p)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    results = []
    np.random.seed(args.seed)
    train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    # print("rank gt ", rank, train_gt)
    train_dataset = HyperX(img, train_gt, "gary", **hyperparams)
    # myidx = 15
    # print("Rank ", rank, train_dataset.__getitem__(myidx)[0][5, :3, :3], train_dataset.__getitem__(myidx)[1])
    # sys.exit()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   # pin_memory=hyperparams['device'],
                                   shuffle=True)
    test_dataset = HyperX(img, test_gt, "gary", **hyperparams)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                  batch_size=hyperparams['batch_size'],
                                  # pin_memory=hyperparams['device'],
                                  shuffle=False)

    epoch_times = []
    test_times = []
    for epoch in range(1, args.epoch + 1):
        start_time = timer()
        train(rank, args, model, train_loader, optimizer, epoch, compose, loss)
        end_time = timer()
        epoch_times += [end_time - start_time]

        start_time = timer()
        test(rank, model, test_loader, compose, loss, hyperparams, img)
        end_time = timer()
        test_times += [end_time - start_time]

    if force_lp:
        timer_str = model.parallel_nn.getTimersString()
        root_print(rank, timer_str)

    root_print(rank, 'TIME PER EPOCH: %.2e (1 std dev %.2e)' % (stats.mean(epoch_times), stats.stdev(epoch_times)))
    root_print(rank, 'TIME PER TEST:  %.2e (1 std dev %.2e)' % (stats.mean(test_times), stats.stdev(test_times)))

    # if rank == 0:
    #     probabilities = test_prob(model, img, hyperparams, "gary")
    #     prediction = np.argmax(probabilities, axis=-1)
    #     run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
    #
    #     mask = np.zeros(gt.shape, dtype='bool')
    #     for l in IGNORED_LABELS:
    #         mask[gt == l] = True
    #     prediction[mask] = 0
    #
    #     color_prediction = convert_to_color(prediction, palette)
    #     if DATAVIZ:
    #         display_predictions(color_prediction, viz, gt=convert_to_color(test_gt), caption="Prediction vs. test ground truth")
    #
    #     show_results(run_results, viz, label_values=LABEL_VALUES)


if __name__ == '__main__':
    main()