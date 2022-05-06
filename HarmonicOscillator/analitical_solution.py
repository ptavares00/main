from sympy import Function, Eq, symbols, diff, lambdify, solve, dsolve, latex
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt


class AnalyticalSolution:
    def __init__(self):
        # Define variables and a function x(t)
        self.m, self.t, self.q, self.w, self.t0, self.f0 = symbols("m t Q omega t_0 f_0")
        x = Function("x")

        # # # Solution for 0 < t < T
        # Define ODE and solve
        ode = Eq(self.m * diff(x(self.t), self.t, 2) + (self.m * self.w / self.q) * diff(x(self.t), self.t)
                 + (self.m * self.w ** 2) * x(self.t), self.f0 * self.t * (self.t0 - self.t) / self.t0 ** 2)
        solution_1 = dsolve(ode, x(self.t))

        # Apply initial conditions for 0 < t < T
        C1, C2 = symbols("C1 C2")
        bc1 = Eq(solution_1.rhs.subs({self.t: 0}), 0)
        bc2 = Eq(diff(solution_1.rhs, self.t).subs({self.t: 0}), 0)
        constant_initial_cond = solve([bc1, bc2], [C1, C2])
        self.solution_1 = solution_1.subs({C1: constant_initial_cond[C1], C2: constant_initial_cond[C2]})

        # # # Solution for t > T
        ode = Eq(self.m * diff(x(self.t), self.t, 2) + (self.m * self.w / self.q) * diff(x(self.t), self.t)
                 + (self.m * self.w ** 2) * x(self.t), 0)
        self.solution_2 = dsolve(ode, x(self.t))

    def fit(self, q, t0):
        # Update parameters
        self.q_chosen = q
        self.t0_chosen = t0

        # Construct function to predict for 0 < t < T
        specific_solution = self.solution_1.subs({self.m: 1, self.f0: 1, self.w: 1, self.t0: t0, self.q: q})
        self.lamb_x = lambdify(self.t, specific_solution.rhs)

        # Apply initial conditions for t > T
        C1, C2 = symbols("C1 C2")
        init_x = self.lamb_x(t0)
        dx = diff(specific_solution.rhs, self.t)
        init_dx = dx.subs({self.t: t0}).evalf()
        bc1 = Eq(self.solution_2.rhs.subs({self.t: t0}), init_x)
        bc2 = Eq(diff(self.solution_2.rhs, self.t).subs({self.t: t0}), init_dx)
        constant_initial_cond = solve([bc1, bc2], [C1, C2])
        self.solution_2 = self.solution_2.subs({C1: constant_initial_cond[C1], C2: constant_initial_cond[C2]})

        # Construct function to predict for t > T
        specific_solution_0 = self.solution_2.subs({self.m: 1, self.f0: 1, self.w: 1, self.t0: t0, self.q: q})
        self.lamb_x_0 = lambdify(self.t, specific_solution_0.rhs)

    def predict(self, t_array):
        x_array = np.zeros(t_array.size)
        for i in np.arange(t_array.size):
            if t_array[i] < self.t0_chosen:
                x_array[i] = self.lamb_x(t_array[i])
            else:
                x_array[i] = self.lamb_x_0(t_array[i])
        return x_array


if __name__ == '__main__':
    q = 1
    t0 = 10
    t_array = np.linspace(0, 15, 500)
    solver = AnalyticalSolution()
    solver.fit(q, t0)
    x_array = solver.predict(t_array)
    plt.plot(t_array, x_array)
    plt.show()

