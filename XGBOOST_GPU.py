
# try xgboost:
from xgboost import XGBClassifier
params={}


# remove these if there is no gpu:
# assign GPU id here
params['gpu_id'] = 0
params['max_bin'] = 16
params['tree_method'] = 'gpu_hist'



model = XGBClassifier(n_estimators=10000,verbose=2,n_jobs=-1,**params)

model.fit(X_train,y_train)