import numpy as np
import matplotlib.pyplot as plt
from reward_over_time import data

y = data
x = list(range(len(data)))

plt.scatter(x, y)
plt.show()
