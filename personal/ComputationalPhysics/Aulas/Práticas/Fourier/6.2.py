import numpy as np
import matplotlib.pyplot as plt

# Ler dados
data = np.loadtxt("sunspots.txt")
x = data[:, 0]; y = data[:, 1]

# Gráfico
plt.Figure()
plt.plot(x, y)
plt.xlabel("mês"); plt.ylabel("nº manchas solares")
plt.grid(); plt.show()

# Potência espetral
c = np.fft.rfft(y)
plt.Figure()
plt.plot(abs(c)**2)
plt.xlabel("k"); plt.ylabel("potência espetral, |ck|^2")
plt.grid(); plt.show()

# Potência espetral aproximado
c = np.fft.rfft(y)
plt.Figure()
plt.plot(abs(c)**2)
plt.xlim(-20, 100)
plt.xlabel("k"); plt.ylabel(r"$|c_k|^2$")
plt.grid(); plt.show()

print(len(y) / 22.5)

# Os dados têm um período entre 100 e 150 meses.
# O pico corresponde a k = ~22.5.
# Note-se que omega = k*2*pi/N, logo T = N/k = 140 meses.

