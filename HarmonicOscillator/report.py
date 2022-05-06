import numpy as np
from scipy.integrate import odeint


def f(t, t0):
    return t*(t0-t)/t0**2 if t < t0 else 0


def system_of_eq(v, t, q, t0):
    z, y = v
    dvdt = [y, -y/q - z + f(t, t0)]
    return dvdt


t0 = 1
q = 1
t_interval = np.linspace(0, 30, 1000)

f = np.vectorize(f)
z0 = 0
dz0 = 0
v0 = [z0, dz0]
solution_num = odeint(system_of_eq, v0, t_interval, args=(q, t0))
