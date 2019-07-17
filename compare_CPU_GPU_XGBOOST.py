
import sklearn
import numpy as np

X = np.ones((10000,5))
Y = np.ones(10000)
# split into 8:2 train test:
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.2)


### XGBOOST:
import time


# try xgboost:
from xgboost import XGBClassifier


time_start = time.time()
params={}
params['gpu_id'] = 0
params['max_bin'] = 4096
params['tree_method'] = 'gpu_hist'
model = XGBClassifier(n_estimators=10000,verbose=2,n_jobs=-1,**params)

model.fit(X_train,y_train)
# predict:

Y_predict = model.predict(X)
print("Time it takes using GPU=%.2f"%(time.time()-time_start))

# CPU:


time_start = time.time()
params={}
#params['gpu_id'] = 0
#params['max_bin'] = 1024
#params['tree_method'] = 'gpu_hist'
model = XGBClassifier(n_estimators=10000,verbose=2,n_jobs=-1)

model.fit(X_train,y_train)
# predict:

Y_predict = model.predict(X)
print("Time it takes using CPU=%.2f"%(time.time()-time_start))

