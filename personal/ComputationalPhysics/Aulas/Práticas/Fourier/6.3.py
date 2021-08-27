import numpy as np
import matplotlib.pyplot as plt

piano = np.loadtxt("piano.txt")
trumpet = np.loadtxt("trumpet.txt")

dt = 1 / 44100
f_max = 1 / (2 * dt)  # Acho que o 2 aparece de este f_max ser na verdade f_max - f_min = 2 * f_max
frequencies = np.linspace(0, f_max, len(piano))
frequencies = frequencies[:len(piano)//2 + 1]

coefficients_piano = np.fft.rfft(piano)
plt.Figure(), plt.xlim(0, 1000)
plt.xlabel("frequency, Hz")
plt.plot(frequencies, abs(coefficients_piano))
plt.grid(), plt.show()
# Dá valores próximos de 261 Hz. A nota é 261 Hz.

coefficients_trumpet = np.fft.rfft(trumpet)
plt.Figure(), plt.xlim(0, 1000)
plt.xlabel("frequency, Hz")
plt.plot(frequencies, abs(coefficients_trumpet))
plt.grid(), plt.show()
# O 1º pico corresponde a 261 Hz, mas o maior não. Não percebo o que é suposto responder para este caso.
