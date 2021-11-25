import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.float_format', '{:.2e}'.format)




def read_residuals(file):
    nt = 0
    Tf = 0.
    iters = 0
    rs = []
    with open(file, 'r') as f:
        for line in f:
            if "Braid" in line:
                if "Begin" in line:
                    nt = int(line.split(',')[1].split(' ')[1])
                elif "conv" in line:
                    rs.append(float(line.split(',')[0].split(' ')[-1]))
            if "stop time" in line:
                Tf = float(line.split(' ')[-1])
            if "iterations" in line:
                iters = int(line.split(' ')[-1])
                return rs, nt, Tf


def mean_convergence(rs):
    window = 3
    if len(rs) < 4:
        window = len(rs) - 1

    m_first = (rs[window] / rs[0])**(1/window)
    m_last = (rs[-1] / rs[-1-window])**(1/window)

    return m_first, m_last


if __name__ == "__main__":
    cfls = [1, 2, 3, 4, 49]
    cfl_labels = ["0.1", "0.2", "0.3", "0.4", "0.49"]

    levels = [2, 3, 5]
    braid_str = "examples/mnist/experiments/{}level/braid_run_cfl_{}"
    torch_str = "examples/mnist/experiments/{}level/torch_braid_cfl_{}"

    braid_data = np.zeros((len(cfls), len(levels)))
    torch_data = np.zeros((len(cfls), len(levels)))

    for i, cfl in enumerate(cfls):
        for j, level in enumerate(levels):
            r_braid, nt_braid, Tf_braid = read_residuals(
                braid_str.format(level, cfl))
            r_torch, nt_torch, Tf_torch = read_residuals(
                torch_str.format(level, cfl))

            if np.abs(Tf_braid - Tf_torch) > 1e-6:
                print(f"Tf doesn't match for level {level}, cfl {cfl}!!")

            end = min(len(r_braid), len(r_torch))

            braid_data[i, j] = mean_convergence(r_braid[:end])[0]
            torch_data[i, j] = mean_convergence(r_torch[:end])[0]

    braid_chart = pd.DataFrame(braid_data, columns=levels, index=cfl_labels)
    torch_chart = pd.DataFrame(torch_data, columns=levels, index=cfl_labels)

    braid_chart.to_csv("examples/mnist/experiments/braid_diff_convergence.csv")
    torch_chart.to_csv("examples/mnist/experiments/torch_diff_convergence.csv")

    print(torch_chart)
