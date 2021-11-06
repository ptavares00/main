import numpy as np
import matplotlib.pyplot as plt
from FisicaEstatistica.TPC1.exponencial_sample import exponential_sample


def binary_search(x1, x2, f):
    p = 1e-6
    f1 = f(x1)
    while abs(x2 - x1) > p:
        average = (x1 + x2) / 2
        f_average = f(average)
        if f_average * f1 > 0:
            x1 = average
            f1 = f_average
        else:
            x2 = average
    return (x2 + x1) / 2


def golden_ratio(x1, x4, f, p=1e-6):
    z = (1 + np.sqrt(5)) / 2
    x2 = x4 - (x4 - x1) / z
    x3 = x1 + (x4 - x1) / z

    f1, f2, f3, f4 = f(x1), f(x2), f(x3), f(x4)

    while (x4 - x1) < p:
        if f(x2) < f(x3):
            x4, f4 = x3, f3
            x3, f3 = x2, f2
            x2 = x4 - (x4 - x1) / z
            f2 = f(x2)
        else:
            x1, f1 = x2, f2
            x2, f2 = x3, f3
            x3 = x1 + (x4 - x1) / z
            f3 = f(x3)
    return (x3 + x2) / 2


def exponential(x, alpha=1):
    return alpha / 2 * np.exp(-alpha * np.abs(x))


def gaussian_distribution(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


x_max = golden_ratio(-2, 0, lambda x: exponential(x) / gaussian_distribution(x))
ratio_max = gaussian_distribution(x_max) / exponential(x_max)
c = binary_search(0, 1, lambda c: 1 - c * ratio_max)


def von_neumann(sample, M, c=c):
    results = np.array([])
    while results.size < M:
        probability = c * gaussian_distribution(sample) / exponential(sample)
        index_results = np.where(np.random.random_sample(probability.size) < probability)
        results = np.concatenate((results, sample[index_results]))
    return results[0:M]


if __name__ == "__main__":
    n = 10**7
    gaussian_sample = von_neumann(exponential_sample(n), n)
    plt.hist(gaussian_sample, bins=500)
    plt.show()



