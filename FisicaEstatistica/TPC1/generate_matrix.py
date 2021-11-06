from FisicaEstatistica.TPC1.von_neumann import *


def GUE(n, m=1):
    index_l = np.tril_indices(n)
    index_d = np.diag_indices(n)
    n_values = np.sum(np.arange(1, n + 1, 1))

    if m == 1:
        real_values = von_neumann(exponential_sample(n_values), n_values)
        complex_values = von_neumann(exponential_sample(n_values), n_values) * 1j
        matrix = np.zeros((n, n), dtype=complex)
        matrix[index_l] = real_values + complex_values
        matrix += matrix.conj().T
        matrix[index_d] /= 2
    else:
        matrix = np.zeros((m, n, n), dtype=complex)
        matrix_aux = np.zeros((n, n), dtype=complex)
        for i in range(m):
            real_values = von_neumann(exponential_sample(n_values), n_values)
            complex_values = von_neumann(exponential_sample(n_values), n_values) * 1j
            matrix_aux[:, :] = 0
            matrix_aux[index_l] = real_values + complex_values
            matrix_aux += matrix_aux.conj().T
            matrix_aux[index_d] /= 2
            matrix[i] = matrix_aux

    return matrix


def GOE(n, m=1):
    index_l = np.tril_indices(n)
    index_d = np.diag_indices(n)
    n_values = np.sum(np.arange(1, n + 1, 1))

    if m == 1:
        values = von_neumann(exponential_sample(n_values), n_values)
        matrix = np.zeros((n, n))
        matrix[index_l] = values
        matrix += matrix.T
        matrix[index_d] /= 2
    else:
        matrix = np.zeros((m, n, n))
        matrix_aux = np.zeros((n, n))
        for i in range(m):
            values = von_neumann(exponential_sample(n_values), n_values)
            matrix_aux[:, :] = 0
            matrix_aux[index_l] = values
            matrix_aux += matrix_aux.T
            matrix_aux[index_d] /= 2
            matrix[i] = matrix_aux

    return matrix


def histogram(samples, inf, sup, n_bins):
    samples_transformed = ((samples - inf) / (sup - inf) * n_bins).astype(int)
    bins_values = np.zeros(n_bins, dtype=int)
    for i in samples_transformed:
        bins_values[i] += 1
    dx = (sup - inf) / n_bins
    bins_limits = np.linspace(inf + dx, sup - dx, n_bins)
    return bins_limits, bins_values / (dx * samples.size)