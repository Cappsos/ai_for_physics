import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(dataset, width):
    # For the first few points we don't have a full window yet, so we just average
    # whatever we have so far; after that we use a proper sliding window of `width`.
    head = [np.average(dataset[0:i]) for i in range(1, width)]
    tail = [np.average(dataset[i:i + width]) for i in range(len(dataset) - width + 1)]
    return head + tail


def plot_movingAverage(dataset, name, width, path_to_save):
    y = moving_average(dataset, width)
    x = np.arange(len(y))
    plt.plot(x, y, 'r.-', label='Running average')
    plt.grid(linestyle=':')
    plt.legend()
    try:
        plt.savefig(path_to_save + name)
    except FileNotFoundError:
        os.makedirs(path_to_save)
        plt.savefig(path_to_save + name)
    plt.close()
