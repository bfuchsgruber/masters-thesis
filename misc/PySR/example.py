import numpy as np
from pysr import *
from matplotlib import pyplot as plt

X = 2 * np.random.randn(100, 5)
y = 1 / X[:, [0, 1, 2]]
model = PySRRegressor(
    binary_operators=["+", "*"],
    unary_operators=["inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1/x},
)
# X = Training data --> might be Weather eg
# Y = Target Values --> maybe the PV generation
model.fit(X, y)

plt.scatter(y[:, 0], model.predict(X)[:, 0])
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.show()

pass