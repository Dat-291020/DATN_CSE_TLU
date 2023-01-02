
import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, LSTM, Dense, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Activation, Reshape, Concatenate
from tensorflow.keras.models import Model

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def normal_lstm(features_count = 1, lag_time = 1):
    x = Input(shape = (lag_time, features_count))
    lstm = LSTM(32)(x)
    dense1 = Dense(16)(lstm)
    output = Dense(1)(dense1)
    model = Model(inputs = x, outputs = output)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape', 'mae'])
    model.summary()
    return model

def stack_lstm(features_count = 1, lag_time = 1):
    x = Input(shape = (lag_time, features_count))
    lstm1 = LSTM(64, return_sequences = True)(x)
    lstm2 = LSTM(32)(lstm1)
    dense1 = Dense(64, name = 'spec_out0')(lstm2)
    output = Dense(1)(dense1)
    model = Model(inputs = x, outputs = output)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape', 'mae'])
    model.summary()
    return model

def mtl_mtv_lstm(features_count, addition_features, lag_time, batch_size=32):
    x = Input(shape = (lag_time, features_count))
    lstm1 = LSTM(16, return_sequences = True)(x)
    lstm2 = LSTM(32, return_sequences = True)(lstm1)
    shared_dense = Dense(64)(lstm2)
    # main task
    sub1 = GRU(units=(lag_time*addition_features), name="task1")(shared_dense)
    #addition task for main imf
    sub2 = LSTM(units=16, name="task2")(shared_dense)
    sub3 = LSTM(units=16, name="task3")(shared_dense)
    # merge sub1 with addition input imfs
    sub1 = Reshape((lag_time, addition_features))(sub1)
    addition_input = Input(shape=(lag_time, addition_features), name='add_input')
    concate = Concatenate(axis=-1)([sub1, addition_input])
    #perform mtl
    out1 = Dense(8, name="spec_out1")(concate)
    out1 = Flatten()(out1)
    out1 = Dense(1, name="out1")(out1)

    out2 = Dense(8, name="spec_out2")(sub2)
    out2 = Dense(1, name="out2")(out2)

    out3 = Dense(8, name="spec_out3")(sub3)
    out3 = Dense(1, name="out3")(out3)

    outputs = [out1, out2, out3]
    # define model
    model = Model(inputs = [x, addition_input], outputs = outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape', 'mae'], loss_weights=[0.5, 0.01, 0.01])
    model.summary()
    return model

def model_mtl_mtv(horizon=1, nb_train_samples=512, batch_size=32, 
                  feature_count=11, lag_time=6, auxiliary_feature_count=12):

    x = Input(shape=(lag_time, feature_count), name="input_layer")

    lstm1 = GRU(16, return_sequences=True)(x)
    # lstm2 = GRU(32, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub1 = GRU(units=(lag_time*auxiliary_feature_count), name="task1")(shared_dense)
    sub2 = GRU(units=16, name="task2")(shared_dense)
    sub3 = GRU(units=16, name="task3")(shared_dense)

    sub1 = Reshape((lag_time, auxiliary_feature_count))(sub1)
    auxiliary_input = Input(shape=(lag_time, auxiliary_feature_count), name='aux_input')

    concate = Concatenate(axis=-1)([sub1, auxiliary_input])
    # out1_gp = Dense(1, name="out1_gp")(sub1)
    out1 = Dense(8, name="spec_out1")(concate)
    out1 = Flatten()(out1)
    out1 = Dense(1, name="out1")(out1)

    out2 = Dense(8, name="spec_out2")(sub2)
    out2 = Dense(1, name="out2")(out2)

    out3 = Dense(1, name="spec_out3")(sub3)
    out3 = Dense(1, name="out3")(out3)

    outputs = [out1, out2, out3]

    model = Model(inputs=[x, auxiliary_input], outputs=outputs)

    # adam optimizsor is good
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape', 'mae'], loss_weights=[0.5, 0.01, 0.01])
    model.summary()
    return model

def mtl_mtv_GRU(horizon=1, nb_train_samples=512, batch_size=32, targets=[], 
                  feature_count=11, lag_time=6, auxiliary_feature_count=12):
    x = Input(shape=(lag_time, feature_count), name="input_layer")

    lstm1 = GRU(16, return_sequences=True)(x)
    # lstm2 = GRU(32, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub = []
    for i in range(len(targets)):
        if i==0:
            subi = GRU(units=(lag_time*auxiliary_feature_count), name="task0")(shared_dense)
        else:
            subi = GRU(units=16, name="task"+str(i))(shared_dense)
        sub.append(subi)

    sub[0] = Reshape((lag_time, auxiliary_feature_count))(sub[0])
    auxiliary_input = Input(shape=(lag_time, auxiliary_feature_count), name='aux_input')

    concate = Concatenate(axis=-1)([sub[0], auxiliary_input])
    # out1_gp = Dense(1, name="out1_gp")(sub1)
    out1 = Dense(8, name="spec_out0")(concate)
    out1 = Flatten()(out1)
    out1 = Dense(1, name="out0")(out1)
    
    out=[]
    out.append(out1)
    for i in range(1, len(targets)):
        outi = Dense(8, name="spec_out"+str(i))(sub[i])
        outi = Dense(1, name="out"+str(i))(outi)
        out.append(outi)

    outputs = out

    model = Model(inputs=[x, auxiliary_input], outputs=outputs)
    
    weights=[]
    for i in range(len(out)):
        if i==0:
            weights.append(0.5)
        else:
            weights.append(0.1)

    # adam optimizsor is good
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape', 'mae'], loss_weights=weights)
    # model.summary()
    return model