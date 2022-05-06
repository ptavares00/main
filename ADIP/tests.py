from imports import *

x = np.array([[1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5]])

x_mean = np.mean(x, 0)
x_mean = np.tile(x_mean, (x.shape[0], 1))
m, n = np.shape(x)
print(n)
