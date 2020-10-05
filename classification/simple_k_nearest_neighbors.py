from math import sqrt
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import pandas as pd 
import random

style.use('fivethirtyeight')

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    # find the closest k points
    votes = [i[1] for i in sorted(distances)[:k]]
    # find the most common group in the closest k points
    vote_result = Counter(votes).most_common(1)[0][0]
    # calculate the confidence of the result
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence

# load in dataset
df = pd.read_csv('classification/breast-cancer-wisconsin.data')
# replace missing data
df.replace('?', -99999, inplace=True)
# drop the id column; not relevant to the prediction
df.drop(['id'], 1, inplace=True)
# standardize data types
full_data = df.astype(float).values.tolist()
# shuffle around the dataset order
random.shuffle(full_data)

# split out the dataset into testing and training
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

# test and calc accuracy
correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

accuracy = correct/total
print('Accuracy:', accuracy)
