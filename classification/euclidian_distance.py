from math import sqrt
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

plot1 = [1,3]
plot2 = [2,5]
euclidean_distance = sqrt((plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2 )
print(euclidean_distance)

dataset = { 'k': [ [1,2], [2,3], [3,1] ], 'r': [ [6,5], [7,7], [8,6] ]}
new_features = [5,7]

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100)
plt.show()