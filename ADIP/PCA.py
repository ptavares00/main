from imports import *


def pca(x, c):
    """
    :param x: [np.array] with dimension MxN (M - no of subjects, N - number of pixels per subject)
    :param c: [int] number of principal components to return
    :return: [np.array] data with dimension MxC

    Instead of using eigen decomposition, this function uses SVD since it is faster for high-dimension data.
    """

    # centre the columns
    m, n = np.shape
    x = np.tile(np.mean(x, 0), (m, 1))

    # obtain eigenvalues
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    s = np.diag(s)
    eig_values = s @ s / (m - 1)

