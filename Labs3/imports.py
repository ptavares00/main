import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as st
import math
from scipy.optimize import curve_fit, minimize
from statsmodels.graphics.gofplots import qqplot
from tabulate import tabulate
from numba import jit, njit


def round_number(number, digits):
    '''
    Rounds a number to a certain number of significant digits. Source:
    https://www.kite.com/python/answers/how-to-round-a-number-to-significant-digits-in-python
    :param number: number to round
    :param digits: number of significant digits
    :return: rounded number
    '''
    return np.round(number, digits - int(math.floor(math.log10(abs(number)))) - 1)


def linest(x: np.array, y: np.array):
    '''
    Prints the values to make the table in excel. Can't make everything in python. sy calculation based on
    https://gist.github.com/jhjensen2/eda0963937b556b8282abed317963384
    :param x: x values
    :param y: y values
    :return: linregress output (slope, stderr, intercept, intercept_stderr, rvalue)
    '''
    table = [['m', 0, 0, 'b'],
             ['sm', 0, 0, 'sb'],
             [r'$r^{2}$', 0, 0, '2sy']
             ]
    res = st.linregress(x, y)
    m = res.slope
    sm = res.stderr
    b = res.intercept
    sb = res.intercept_stderr
    r = res.rvalue**2
    p = np.poly1d([m, b])
    yp = p(x)
    residual_ss = np.sum((y - yp) ** 2)
    dofreedom = len(yp) - 2
    sy = np.sqrt(residual_ss / dofreedom)
    table[0][1:3] = [m, b]
    table[1][1:3] = [round_number(sm, 1), round_number(sb, 1)]
    table[2][1:3] = [r, round_number(2 * sy, 1)]
    print(tabulate(table, tablefmt='grid'))
    return {'m': m, 'b': b, 'r2': r, 'sm': sm, 'sb': sb, '2sy': 2*sy, 'sy': sy}
