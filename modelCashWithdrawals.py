import numpy as np
import pandas as pd
import time
#take time
tic = time.time()
#import the data
df = pd.read_csv("training_data.csv")
#pick training and validation set randomly
train=df.sample(frac=0.9,random_state=10)
df_val=df.drop(train.index)
df=train
N=df.shape[0]
true_vals = df_val.TRX_COUNT



from sklearn.metrics import mean_squared_error
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# the parameters are optimized using grid search
regr = RandomForestRegressor(n_estimators=150, max_depth=12)
#grid search
#param_grid = {'n_estimators': [100,120,140,150,170], 'max_depth': list(np.arange(10,15,1))}
#regr = GridSearchCV(regr, param_grid, cv=10)

# Train the model using the training set
regr.fit(df[['IDENTITY','REGION','DAY','MONTH','YEAR','TRX_TYPE']], df['TRX_COUNT'])


# Make predictions using the validation set
y_pred = regr.predict(df_val[['IDENTITY','REGION','DAY','MONTH','YEAR','TRX_TYPE']])



# print RMSE on validation set
print("RMSE is:")
print(np.sqrt(mean_squared_error(true_vals, y_pred)))
# print MAE on validation set
print("MAE is:")
print(np.mean(abs(y_pred - true_vals)))


# import test points
test = pd.read_csv("test_data.csv")
# predict TRX_COUNT using the trained random forest estimator
test_pred = regr.predict(test[['IDENTITY','REGION','DAY','MONTH','YEAR','TRX_TYPE']])
# write the predictions to a csv file
pd.DataFrame(test_pred).to_csv("test_predictions.csv",header=None,index=None)
# take time again
toc=time.time()
