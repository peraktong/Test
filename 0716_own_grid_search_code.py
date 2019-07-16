import itertools
# try xgboost:
from xgboost import XGBClassifier
import numpy as np

params = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
          "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
          "n_estimators": [10, 30, 50, 70]}

X_train = np.linspace(1, 13, 100)
y_train = np.linspace(0, 12, 100)

count = 0
# grid search
keys, values = zip(*params.items())
for v in itertools.product(*values):
    # print(keys, v)

    print("Doing %d" % count)
    experiment = dict(zip(keys, v))

    experiment['gpu_id'] = 0
    experiment['max_bin'] = 512
    experiment['tree_method'] = 'gpu_hist'

    model = XGBClassifier(**experiment)
    print(model)
    count += 1











