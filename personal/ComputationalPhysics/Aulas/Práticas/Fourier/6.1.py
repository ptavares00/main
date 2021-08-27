import numpy as np
import cmath as cm
import matplotlib.pyplot as plt


# Transformada de Fourier
def fourier(y):
    size = len(y)
    c = np.zeros(size, complex)
    for k in range(size//2 + 1):
        for n in range(size):
            c[k] += y[n] * cm.exp(-1j * (2 * cm.pi * n * k) / size)
    c[size//2 + 1:] = c[size//2 - 1:0:-1].conjugate()
    return c


# Onda quadrada
def square_wave(n):
    return np.ones(n, complex)


# Onda dente de serra
def tooth(n):
    return np.arange(n)


# Onda sinusoidal
def sinusoidal(n):
    aux = np.arange(n)
    a = np.sin(np.pi * aux / n)
    b = np.sin(20 * np.pi * aux / n)
    return a * b


number = 1000

# Gráfico onda quadrada
plt.Figure()
# plt.plot(abs(fourier(square_wave(number))), label="|ck| mine")
plt.plot(abs(np.fft.fft(square_wave(number))), label="|ck| numpy")
plt.title("Onda quadrada")
plt.xlabel("n")
plt.grid()
plt.legend()
plt.show()

# Gráfico onda dente de serra
plt.Figure()
# plt.plot(abs(fourier(tooth(number))), label="|ck| mine")
plt.plot(abs(np.fft.fft(tooth(number))), label="|ck| numpy")
plt.title("Onda dente de serra")
plt.xlabel("n")
plt.grid()
plt.legend()
plt.show()

# Gráfico onda sinusoidal
plt.Figure()
# plt.plot(abs(fourier(sinusoidal(number))), label="|ck| mine")
plt.plot(abs(np.fft.fft(sinusoidal(number))), label="|ck| numpy")
plt.title("Onda sinusoidal")
plt.xlabel("n")
plt.grid()
plt.legend()
plt.show()

# tanto a fft como a rfft usam a fast fourier transform, mas a rfft trata o input como se fossem reais, ou seja,
# só calcula os 1ºs N//2 + 1 coeficientes, os restantes podem-se obter através do conjugado destes. Contudo, como a
# amplitude é o que nos interessa, o gráfico vai ser só uma reflexão dos N//1 + 2 coeficientes (exceto o último que não
# é novamente calculado). Assim, não tem grande interesse o cálculo dos mesmos, como se pode verificar nos gráficos
# obtidos.
