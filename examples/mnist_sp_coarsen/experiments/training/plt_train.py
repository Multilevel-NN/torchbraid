import numpy as np
import matplotlib.pyplot as plt
plt.style.use(
    'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')


def parse_training(file):
    loss = []
    acc = []
    hold_res = []
    resconv = []
    with open(file, 'r') as f:
        for line in f:
            if "Train" in line:
                loss.append(float(line.split()[6]))
            if "Test" in line:
                r = line.split()[-2].split('/')
                acc.append(float(r[0])/float(r[1]))
            elif "conv" in line:
                hold_res.append(float(line.split(',')[0].split()[-1]))
            elif "Max. iterations reached" in line and len(hold_res) > 0:
                resconv.append((hold_res[-1]/hold_res[0])**(1/len(hold_res)))
                hold_res = []

        return np.array(loss), np.array(acc), resconv

def average_runs(fstr, seeds):
    losses, accuracies = [], []
    for seed in seeds:
        _loss, _acc, resconv = parse_training(dir + fstr.format(seed))
        losses.append(_loss)
        accuracies.append(_acc)
    loss = np.stack(losses)
    acc = np.stack(accuracies)
    return loss.mean(0), acc.mean(0), loss.std(0), acc.std(0)
    


if __name__ == "__main__":
    seeds = range(2, 17)
    dir = "examples/mnist/experiments/training/"
    # dir = "examples/mnist/experiments/training_fashion/"
    # dir = "./"
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

    loss, acc, sig_loss, sig_acc = average_runs("parallel_tanh_{}.out", seeds)
    epochs = range(1, len(acc)+1)
    loss_scale = np.linspace(1, len(acc), num=len(loss))
    axs[0].semilogy(loss_scale, loss, label="3 level, No SC", color="C0")
    axs[1].plot(epochs, acc, color="C0")
    axs[0].fill_between(loss_scale, loss+sig_loss, np.abs(loss-sig_loss), facecolor="C0", alpha=0.4)
    axs[1].fill_between(epochs, acc+sig_acc, acc-sig_acc, facecolor="C0", alpha=0.4)

    loss, acc, sig_loss, sig_acc = average_runs("parallel_sc_tanh_{}.out", seeds)
    epochs = range(1, len(acc)+1)
    loss_scale = np.linspace(1, len(acc), num=len(loss))
    axs[0].semilogy(loss_scale, loss, label="3 level, Deferred SC", color="C1")
    axs[1].plot(epochs, acc, color="C1")
    axs[0].fill_between(loss_scale, loss+sig_loss, np.abs(loss-sig_loss), facecolor="C1", alpha=0.4)
    axs[1].fill_between(epochs, acc+sig_acc, acc-sig_acc, facecolor="C1", alpha=0.4)

    loss, acc, sig_loss, sig_acc = average_runs("serial_tanh_{}.out", seeds)
    epochs = range(1, len(acc)+1)
    loss_scale = np.linspace(1, len(acc), num=len(loss))
    axs[0].semilogy(loss_scale, loss, label="serial", color="C2")
    axs[1].plot(epochs, acc, color="C2")
    axs[0].fill_between(loss_scale, loss+sig_loss, np.abs(loss-sig_loss), facecolor="C2", alpha=0.4)
    axs[1].fill_between(epochs, acc+sig_acc, acc-sig_acc, facecolor="C2", alpha=0.4)

    loss, acc, sig_loss, sig_acc = average_runs("parallel_sc_tanh_2lvl_{}.out", seeds)
    epochs = range(1, len(acc)+1)
    loss_scale = np.linspace(1, len(acc), num=len(loss))
    axs[0].semilogy(loss_scale, loss, label="2 level, SC", color="C3")
    axs[1].plot(epochs, acc, color="C3")
    axs[0].fill_between(loss_scale, loss+sig_loss, np.abs(loss-sig_loss), facecolor="C3", alpha=0.4)
    axs[1].fill_between(epochs, acc+sig_acc, acc-sig_acc, facecolor="C3", alpha=0.4)

    loss, acc, sig_loss, sig_acc = average_runs("parallel_sc_tanh_full_{}.out", seeds)
    epochs = range(1, len(acc)+1)
    loss_scale = np.linspace(1, len(acc), num=len(loss))
    axs[0].semilogy(loss_scale, loss, label="3 level, full SC", color="C4")
    axs[1].plot(epochs, acc, color="C4")
    axs[0].fill_between(loss_scale, loss+sig_loss, np.abs(loss-sig_loss), facecolor="C4", alpha=0.4)
    axs[1].fill_between(epochs, acc+sig_acc, acc-sig_acc, facecolor="C4", alpha=0.4)

    axs[1].set_xlabel("epoch", fontsize=14)
    axs[0].set_ylabel("training loss", fontsize=14)
    axs[1].set_ylabel("test accuracy", fontsize=14)
    axs[1].set_xticks(epochs[::2])

    axs[0].legend(fontsize=14)
    axs[0].set_title("Digits MNIST")
    plt.savefig("Overleaf/Beamer/04_12_22/training_history_sc.png", dpi=300)
    plt.show()

    # plt.figure()
    # plt.plot(range(len(resconv)), resconv, '.')
    # plt.show()
