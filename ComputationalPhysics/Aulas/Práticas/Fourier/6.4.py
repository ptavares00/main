import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("dow.txt")
plt.plot(data)
plt.xlabel("dia útil"), plt.ylabel("valor de encerramento")
plt.show()

c = np.fft.rfft(data)
percentage = 0.02
aux = int(len(c) * percentage)
c[aux:] = 0
data_2 = np.fft.irfft(c)

plt.Figure()
plt.plot(data)
plt.plot(data_2)
plt.show()

# torna a curva mais smooth e quanto maior a percentagem de 0's mais diferente a curva é. Perde-se informação.
