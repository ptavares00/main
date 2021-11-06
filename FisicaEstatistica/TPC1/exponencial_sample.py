import numpy as np
import matplotlib.pyplot as plt


def exponential_sample(n, alpha=1):
    x = np.random.random_sample(n)
    sign = np.random.choice([-1, 1], n)
    return sign * np.log(1 - x) / alpha


if __name__ == "__main__":
    sample = exponential_sample(100000)

    plt.hist(sample, bins=100)
    plt.show()
