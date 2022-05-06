import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt
from analitical_solution import AnalyticalSolution
import warnings

warnings.filterwarnings('ignore')
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["figure.figsize"] = (12, 8)
mpl.rcParams["legend.fontsize"] = 15
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12


def f(t, t0):
    return t*(t0-t)/t0**2 if t < t0 else 0


def system_of_eq(v, t, q, t0):
    z, y = v
    dvdt = [y, -y/q - z + f(t, t0)]
    return dvdt


# Choose parameters
t0_list = [0.1, 1, 10]
q_list = [-1, 1, 10**8]
t_interval = np.linspace(0, 30, 1000)

# Avoid repetitions in the for loop
f = np.vectorize(f)
z0 = 0
dz0 = 0
v0 = [z0, dz0]

for q in q_list:
    c_list = ["y", "r", "c"]
    for t0, c in zip(t0_list, c_list):

        # Numerical Solution
        solution_num = odeint(system_of_eq, v0, t_interval, args=(q, t0))

        # Analytical Solution
        solver = AnalyticalSolution()
        solver.fit(q, t0)
        solution_ana = solver.predict(t_interval)

        # Generate Graph
        plt.plot(t_interval, solution_num[:, 0], label=f"Numerical, T' = {t0}", c=c)
        plt.plot(t_interval, solution_ana, c="k", label="Analytical", linestyle=(0, (5, 6.5)))
    if q > 10**3:
        plt.title(fr"Behaviour for Q = {q:.0e}")
    else:
        plt.title(fr"Behaviour for Q = {q}")
    plt.xlabel(r"$\tau$")
    plt.ylabel("z")
    plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid()
    plt.savefig(fr"C:\Users\user\OneDrive - Universidade do Porto\Desktop\RPs\Ringdown of an Harmonic Oscillator\graph_{q}.png")
    plt.show()

