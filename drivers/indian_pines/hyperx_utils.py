from deephyperx_master.datasets import HyperX
import sys


class HyperXG(HyperX):

    def __init__(self, data, gt, **hyperparams):
        super(HyperXG, self).__init__(data, gt, **hyperparams)

    def __getitem__(self, i):
        data, label = HyperX.__getitem__(self, i)
        #Undo the unsqueeze that HyperX creates.  Needed for proper dimensions for torchbraid.
        if self.patch_size > 1:
            data = data.squeeze(0)

        return data, label