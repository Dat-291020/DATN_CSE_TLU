from preprocessing.read import load_csv, window_generate, split, split_y, power_set
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import os
import matplotlib.pyplot as plt
from model.model import normal_lstm, stack_lstm, mtl_mtv_lstm, model_mtl_mtv, mtl_mtv_GRU
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error, explained_variance_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from keras.models import Model

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit, TimeSeriesSplit


# load data and set the datetime index for data frame
datadir = 'data/data_stock.csv'

# imf1 = pd.read_csv("IMF/imfs/IMF-0.csv",names=['x1'])
# imf2 = pd.read_csv("IMF/imfs/IMF-1.csv",names=['x2'])
# imf3 = pd.read_csv("IMF/imfs/IMF-2.csv",names=['x3'])
# imf4 = pd.read_csv("IMF/imfs/IMF-3.csv",names=['x4'])
i5 = pd.read_csv("IMF/imfs/IMF-4.csv",names=['x5'])
i6 = pd.read_csv("IMF/imfs/IMF-5.csv",names=['x6'])
i7 = pd.read_csv("IMF/imfs/IMF-6.csv",names=['x7'])
i8 = pd.read_csv("IMF/imfs/IMF-7.csv",names=['x8'])

features = []
data = load_csv(datadir, features = features)
# data = data.set_index(pd.to_datetime(data['Date'], format='%m/%d/%Y'))
data = data.drop(['Date'], axis=1)
data = data.dropna()
data=pd.concat([data,i8,i7,i6,i5],axis=1)
# preparing the train, valid and test data
## set the milestone for dividing data by time
## đặt các mốc để chia dữ liệu thành train/validdate/test
valid_index=1000
test_index=1200
end_index=1500
## set the lag time and horizon
LAG = 5
HORIZON = 3
##công cụ chuẩn hóa dữ liệu
X_scaler = StandardScaler()
y_scaler = StandardScaler()
main_target_scaler = StandardScaler()

result = pd.DataFrame()
result[['algorithm', 'LAG', 'Horizon', 'rmse', 'mae','mape', 'r2']] =["","","","","","",""]

# =========================================================

X_scaler = StandardScaler()
y_scaler = StandardScaler()
main_target_scaler = StandardScaler()

input_features = data.columns
targets = [data.columns[0]] # y

X = data[input_features]
y = data[targets]

## tạo train data
X_train = X.copy().loc[X.index < valid_index]
y_train = y.copy().loc[y.index < valid_index]

# fit công cụ chuẩn hóa theo dữ liệu train
X_scaler.fit(X_train)
y_scaler.fit(y_train)
main_target_scaler.fit(y_train[targets[0]].values.reshape(-1,1))

# dùng hàm tự định nghĩa split để chia dữ liệu thành
# train/valid/test data, đồng thời tạo các chuỗi dữ liệu độ dài LAG để phục vụ dự báo
X_tr, X_v, X_t = split(data, input_features, X_scaler, 
                                 valid_index, test_index, end_index, time_lag=LAG, freq=None)
y_tr, y_v, y_t = split_y(data, targets, y_scaler, 
                                 valid_index, test_index, end_index, horizon=HORIZON, freq=None)
# add_tr, add_v, add_t = split(imfkolq, add_features, add_scaler, 
#                              valid_index, test_index, end_index,time_lag=LAG, freq=None)
# ==========================================================

# sp lstm
train = pd.concat([X_tr, y_tr], axis=1) 
train = train.loc[:,~train.columns.duplicated()].copy()
train = train.dropna()

valid = pd.concat([X_v, y_v], axis=1)
valid = valid.loc[:,~valid.columns.duplicated()].copy()
valid = valid.dropna()

test = pd.concat([X_t, y_t], axis=1)
test = test.loc[:,~test.columns.duplicated()].copy()
test = test.dropna()

in_features = X_tr.columns
out_features = y_tr.columns

# train, valid, test for lstm
Xtrain = train[in_features].values
Xtrain = Xtrain.reshape(Xtrain.shape[0], LAG, len(input_features))
ytrain = []
for feature in out_features:
    ytrain.append(train[feature].values.reshape(-1,1))

Xvalid = valid[in_features].values
Xvalid = Xvalid.reshape(Xvalid.shape[0], LAG, len(input_features))
yvalid = []
for feature in out_features:
    yvalid.append(valid[feature].values.reshape(-1,1))

Xtest = test[in_features].values
Xtest = Xtest.reshape(Xtest.shape[0], LAG, len(input_features))
ytest = []
for feature in out_features:
    ytest.append(test[feature].values.reshape(-1,1))

# Train, Test for RF, SVR...
XTrain = np.concatenate([Xtrain.reshape(-1, len(in_features)), 
                          Xvalid.reshape(-1, len(in_features))], axis=0)
XTest = Xtest.reshape(-1, len(in_features))
yTrain = np.concatenate([ytrain[0], yvalid[0]], axis=0).ravel()
yTest = ytest[0].ravel()

# ==========================================================

# nash-sutcliffe efficiency
def nse(targets,predictions):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(predictions))**2))

# ==========================================================

## gọi công cụ chia dữ liệu để chia 2 fold với mỗi fold có cỡ test =100
cv = TimeSeriesSplit(n_splits=2, test_size=100)


# svr
C_range = np.logspace(-3, 3, 7)
gamma_range = np.logspace(-5, 2, 8)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(SVR(), param_grid=param_grid, cv=cv,
                    scoring = 'neg_mean_absolute_percentage_error',
                    verbose=0)
grid.fit(XTrain, yTrain)

print("The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_))

model_svr = SVR(kernel='rbf', C=grid.best_params_['C'], 
          gamma = grid.best_params_['gamma'])

yPredSVR = model_svr.fit(XTrain, yTrain).predict(XTest)
y_test_svr = main_target_scaler.inverse_transform(yTest.reshape(-1,1))
y_pred_svr = main_target_scaler.inverse_transform(yPredSVR.reshape(-1,1))

print('svr')
print('rmse: ' + str(mean_squared_error(y_test_svr, y_pred_svr, squared=False)))
print('r2: '+ str(r2_score(y_test_svr, y_pred_svr)))
print('mae: ' + str(mean_absolute_error(y_test_svr, y_pred_svr)))
print('mape: '+ str(mean_absolute_percentage_error(y_test_svr, y_pred_svr)))

result.loc[result.shape[0]] = ['SVR', LAG, HORIZON,
                               mean_squared_error(y_test_svr, y_pred_svr, squared=False),
                               mean_absolute_error(y_test_svr, y_pred_svr),
                               mean_absolute_percentage_error(y_test_svr, y_pred_svr),
                               r2_score(y_test_svr, y_pred_svr),
                               ]

# rf
n_estimators_range = [100, 200, 500, 1000]###pham vi 
max_depth_range = np.array(range(5, 25, 5))###do sau
rf_param_grid = dict(n_estimators=n_estimators_range, max_depth=max_depth_range)

grid_rf = GridSearchCV(RandomForestRegressor(), param_grid=rf_param_grid, cv=cv,
                    scoring = 'neg_mean_squared_error',  verbose=0)
grid_rf.fit(XTrain, yTrain)
print("The best parameters of randomforest are %s with a score of %0.2f"
    % (grid_rf.best_params_, grid_rf.best_score_))

model_rf = RandomForestRegressor(n_estimators=grid_rf.best_params_['n_estimators'],
                                 max_depth=grid_rf.best_params_['max_depth'])
y_pred_rf = model_rf.fit(XTrain, yTrain).predict(XTest)
y_test_rf = main_target_scaler.inverse_transform(yTest.reshape(-1,1))
y_pred_rf = main_target_scaler.inverse_transform(y_pred_rf.reshape(-1,1))

print('rf')
print('rmse: ' + str(mean_squared_error(y_test_rf, y_pred_rf, squared=False)))
print('r2: '+ str(r2_score(y_test_rf, y_pred_rf)))
print('mae: ' + str(mean_absolute_error(y_test_rf, y_pred_rf)))
print('mape: '+ str(mean_absolute_percentage_error(y_test_rf, y_pred_rf)))

result.loc[result.shape[0]] = ['RF', LAG, HORIZON,
                               mean_squared_error(y_test_rf, y_pred_rf, squared=False),
                               mean_absolute_error(y_test_rf, y_pred_rf),
                               mean_absolute_percentage_error(y_test_rf, y_pred_rf),
                               r2_score(y_test_rf, y_pred_rf),
                               ]

#lstm
earlystop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
file_path = 'checkpoint/weights-improvement-{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(monitor="val_loss",
                              filepath=file_path,
                              verbose=0,
                              save_weight_only=True,
                              save_best_only=True)
###
model_lstm = stack_lstm(features_count=len(input_features), lag_time=LAG)

## train
model_lstm.fit(Xtrain, ytrain[0], 
          batch_size=32, 
          epochs=500, 
          validation_data=(Xvalid, yvalid[0]),
          callbacks=[earlystop, checkpoint],
          verbose = 2)
## dự báo
y_pred_lstm = model_lstm.predict(Xtest)
y_test_lstm = main_target_scaler.inverse_transform(ytest[0].reshape(-1,1))
y_pred_lstm = main_target_scaler.inverse_transform(y_pred_lstm.reshape(-1,1))

print('lstm')
print('rmse: ' + str(mean_squared_error(y_test_lstm, y_pred_lstm, squared=False)))
print('r2: '+ str(r2_score(y_test_lstm, y_pred_lstm)))
print('mae: ' + str(mean_absolute_error(y_test_lstm, y_pred_lstm)))
print('mape: '+ str(mean_absolute_percentage_error(y_test_lstm, y_pred_lstm)))

result.loc[result.shape[0]] = ['lstm', LAG, HORIZON,
                               mean_squared_error(y_test_lstm, y_pred_lstm, squared=False),
                               mean_absolute_error(y_test_lstm, y_pred_lstm),
                               mean_absolute_percentage_error(y_test_lstm, y_pred_lstm),
                               r2_score(y_test_lstm, y_pred_lstm),
                               ]