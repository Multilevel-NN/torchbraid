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
            if ("iterations" in line) and ("max" not in line) and ("Max" not in line):
                iters = int(line.split(' ')[-1])
                return rs, nt, Tf


def mean_convergence(rs):
    window = 4
    if len(rs) < 4:
        window = len(rs) - 1

    m_first = (rs[window] / rs[0])**(1/window)
    m_last = (rs[-1] / rs[-1-window])**(1/window)

    return m_first, m_last


if __name__ == "__main__":
    channels = (1, 4)
    levels = (2, 3)
    funcs = ("relu", "tanh")

    torch_str = "examples/mnist/experiments/trained_nets/{}_{}_{}l"
    torch_data = np.zeros((len(channels), len(levels)))

    # load data for no spatial coarsening
    def table(func, sc=False):
        if sc:
            append = "_sc.csv"
        else:
            append = ".csv"

        for i, channel in enumerate(channels):
            for j, level in enumerate(levels):
                r_torch, nt_torch, Tf_torch = read_residuals(torch_str.format(funcs[0], channel, level) + append)
                torch_data[i, j] = mean_convergence(r_torch)[0]

        torch_chart = pd.DataFrame(torch_data, columns=levels, index=channels)
        torch_chart.to_csv("torch_diff_convergence.csv")

        print("Mean initial convergence: (mean convergence rate for first 3 iterations)")
        print("=========================================================================")
        print("channels/max levels:")
        # display(HTML(torch_chart.to_html()))

    table(funcs[0], False)