'''
After running deb_encdec.job:
'''

import matplotlib.pyplot as plt
import os
import re
import sys
import torch

from colors import get_colors

GRADS_DIR = 'grads'

NUM_NODES = 3
ITS = [1, 8, 16]  # 32]
NUM_BATCHES = [1, 8, 32, 64]
LAYER_IDS = list(range(-2, 12))

fig, axs = plt.subplots(2, 1)

# colors = get_colors(len(ITS)*len(NUM_BATCHES)).tolist() * 2
colors = get_colors(len(ITS)*len(NUM_BATCHES)) * 2
iter_colors = iter(colors)

variant = 'encdec'
for jj, its in enumerate(ITS):
    for num_batches in NUM_BATCHES:        
        g_dirs = {i: [] for i in LAYER_IDS}
        g_mods = {i: [] for i in LAYER_IDS}
        gT = torch.load(
            os.path.join(
                GRADS_DIR,
                f'grads_True_{its}_{num_batches}_rank0_{variant}.pt',
            )
        )
        gFs = []

        for rank in range(NUM_NODES):
            gFs.append(
                torch.load(
                    os.path.join(
                        GRADS_DIR,
                        f'grads_False_{its}_{num_batches}_rank{rank}_{variant}.pt',
                    )
                )
            )

        pattern = 'parallel_nn\.local_layers\.(\d+)\.layer\.'
        max_digit = -1
        current_max_digit = -1
        for j, gFj in enumerate(gFs):
            max_digit = current_max_digit
            for k in range(len(gFj)):
                nm, x = gFj[k]
                if (s := re.search(pattern, nm)):
                    i = int(s[1])
                    l = nm.split('.')
                    assert current_max_digit <= i + max_digit + 1
                    current_max_digit = i + max_digit + 1
                    l[2] = str(current_max_digit)
                    new_nm = '.'.join(l)
                    gFj[k] = (new_nm, x)

        # print(list(map(len, gFs)))

        gF = [
            *gFs[0][:-4],
            *sum(gFs[1:], start=[]),
            *gFs[0][-4:],
        ]

        # # print(f'{type(gT)=}\n{type(gF)=}')
        # print(f'{len(gT)=}\n{len(gF)=}')
        # print(f'{len(gT[0])=}\n{len(gF[0])=}')

        for ((nmT, pT), (nmF, pF)) in zip(gT, gF):
            lT = nmT.split('.')
            lF = nmF.split('.')

            if nmT.startswith('open_nn') or nmF.startswith('open_nn'):
                assert nmT.startswith('open_nn') and nmF.startswith('open_nn')
                idxT, idxF = -2, -2
                p_nmT, p_nmF = nmT, nmF

            elif nmT.startswith('close_nn') or nmF.startswith('close_nn'):
                assert nmT.startswith('close_nn') and nmF.startswith('close_nn')
                idxT, idxF = -1, -1
                p_nmT, p_nmF = nmT, nmF

            else:
                idxT = int(lT[1])
                idxF = int(lF[2])

                p_nmT = '.'.join(lT[3:])
                p_nmF = '.'.join(lF[3:])

            assert idxT == idxF and p_nmT == p_nmF, [nmT, nmF]

            pT, pF = pT.ravel(), pF.ravel()
            pT_norm, pF_norm = pT.norm(), pF.norm()
            pTu, pFu = pT/pT_norm, pF/pF_norm
            g_dirs[idxT].append((pTu @ pFu).abs().max())
            # g_dirs[idxT].append(pT_norm)
            g_mods[idxT].append((pT_norm - pF_norm).abs().max())
            # g_mods[idxT].append((pT_norm - pF_norm).abs().max()/pT_norm)

        g_dirs_maxs = {k: max(v) for (k, v) in g_dirs.items()}
        g_mods_maxs = {k: max(v) for (k, v) in g_mods.items()}

        kwargs = {
            'label': f'its={its}, num_batches={num_batches}'
        } if variant == 'encdec' else {}

        color = next(iter_colors)

        ax = axs[0]
        # ax.semilogy(
        ax.plot(
            g_dirs_maxs.keys(),
            g_dirs_maxs.values(),
            color=color,
            linestyle=['-', '-', '--'][jj],
            # linestyle='-' if variant=='encdec' else '--',
            marker='*',
            **kwargs,
        )

        ax = axs[1]
        ax.semilogy(
            g_mods_maxs.keys(),
            g_mods_maxs.values(),
            color=color,
            linestyle=['-', '-', '--'][jj],
            marker='*',
            **kwargs,
        )

for ax in axs:
    ax.vlines([-.5, 5.5], -1, 1, 'k')
    ax.set_xlabel('#Layer IDS (-2: OpenLayer, -1: CloseLayer). 0-5 Enc, 6-11 Dec')
    ax.grid()

ax = axs[0]
ax.set_ylabel('Grad. inf-norm unitary inner product with SerialNet')

ax = axs[1]
ax.set_ylabel('Grad. inf-norm norm difference with SerialNet')

fig.suptitle(f'Enc-dec, mpirun -n {NUM_NODES}')
ax.legend()
plt.show()



