# %matplotlib inline
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

np.random.seed(9876789)

# OLS estimation¶
# Artificial data:

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

# Our model needs an intercept so we add a column of 1s:
X = sm.add_constant(X)
y = np.dot(X, beta) + e

# Fit and summary:

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
