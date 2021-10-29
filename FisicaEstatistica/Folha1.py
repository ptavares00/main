import numpy as np
import matplotlib.pyplot as plt


def histogram(samples, inf, sup, n_bins):
    samples_transformed = ((samples - inf) / (sup - inf) * n_bins).astype(int)
    bins_values = np.zeros(n_bins, dtype=int)
    for i in samples_transformed:
        bins_values[i] += 1
    dx = (sup - inf) / n_bins
    bins_limits = np.linspace(inf + dx, sup - dx, n_bins)
    return bins_limits, bins_values / (dx * samples.size)


if __name__ == "__main__":

    # Problem 1
    for i in [1024, 4096, 16384]:
        samples = np.random.random_sample(i)
        x, y = histogram(samples, 0, 1, 128)
        plt.plot(x, y, label=f'{i}')
        plt.legend()
    plt.show()

    # Problem 2

