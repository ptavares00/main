import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

# Load the data to the variable data
data = np.array([1, 2, 3])

mean = np.mean(data)
std = np.std(data, ddof=1)

mu, sigma = st.norm.fit(data)
x = np.linspace(data.min(), data.max(), len(data))
pdf = st.norm.pdf(x, mu, sigma)

plt.hist(data)
plt.plot(x, pdf, label=f'N({mu},{sigma})')
plt.legend()
plt.show()
