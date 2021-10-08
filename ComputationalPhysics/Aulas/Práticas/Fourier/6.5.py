import numpy as np
import matplotlib.pyplot as plt


def f(n):
    if (int(2*n) + 1) % 2 == 0:
        return 1
    else:
        return -1


f = np.vectorize(f)
x = np.linspace(0, 1, 1000)
c = np.fft.rfft(f(x))
c[10:] = 0
f_changed = np.fft.irfft(c)
plt.plot(x, f(x))
plt.plot(x, f_changed)
plt.show()

# Explicação semelhante ao que se deu em sinais e sistemas. A função trata-se da soma de funções sinusoidais e só no
# limite k -> infinito (na transformada contínua) é que a expansão de fourier é igual à própria função. Aqui, ao
# anularmos coeficientes estamos a anular a contribuição de funções sinusoidais o que leva a uma forma diferente do
# gráfico.
