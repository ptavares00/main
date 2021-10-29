import numpy as np
import matplotlib.pyplot as plt


def exponential_sample(x, alpha=1):
    signal = np.random.choice([-1, 1], x.size)
    return signal * np.log(1 - 2 * x) / alpha


x = np.random.random_sample(100000)
sample = exponential_sample(x)

plt.hist(sample, bins=100)
plt.show()
