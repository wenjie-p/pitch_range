#!/usr/bin/env python
# coding: utf-8

# In[30]:


import os
import sys
from datetime import datetime
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, LSTM, Bidirectional
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
import codecs
import h5py

np.random.seed(42)

def init_lstm(loss, num_steps, input_dim, hidden_size = 50):

    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences = True, input_shape = (num_steps, input_dim)))
    #model.add(Bidirectional(LSTM(hidden_size, input_shape = (num_steps, input_dim), return_sequences = False)))
    model.add(LSTM(hidden_size, return_sequences = True ))
    model.add(LSTM(hidden_size, return_sequences = False ))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss = loss, optimizer = "rmsprop")


    print(model.summary())

    return model

def gen_data_for_test(fp, num_steps, skip_steps, op):
    
    data = np.loadtxt(fp, "float")
    dataX, dataY = [], []
    cur = 0
    print("processing file: {} with: {}".format(fp, len(data)))
    while cur+num_steps<len(data):
        samples = data[cur:cur+num_steps]
        cur+=skip_steps
        x = samples[:,:op]
        y = samples[0][op:]
        dataX.append(x)
        dataY.append(y)
        
    return np.array(dataX), np.array(dataY), len(dataX)

def evaluate_manually(fin, fout, md, num_steps, skip_steps, cur_ans, ave_ans):
    
    X, Y, spe = gen_data_for_test(fin, num_steps, skip_steps, -4)
    
    Yhat = md.predict(X)
    
    yhat_tot = 0
    fp = codecs.open(fout, "w", encoding = "utf8")
    for i in range(spe):
        yhat = Yhat[i][0]
        y = Y[i][cur_ans]
        yhat_tot += yhat
        line = "{} {}\n".format(y, yhat)
        fp.write(line)
    
    local_yhat = yhat_tot/spe
    glob_y = Y[0][ave_ans]
    local_y = Y[0][cur_ans]
    fp.write("glob_y {} local_y {} local_yhat {}\n".format(glob_y, local_y, local_yhat))
    fp.close()
    
    return glob_y, local_y, local_yhat


def predict(fmd, src, des, cur_ans, ave_ans):
    
    loss = "mean_absolute_error"
    fdic = {}
    num_steps = 30
    input_dim = 40
    hidden_size = 100
    skip_steps = 5
    md = init_lstm(loss, num_steps, input_dim, hidden_size)
    md.load_weights(fmd)
    
    fp = codecs.open(des+"/info.txt", "w", encoding = "utf8")
    for spk in os.listdir(src):
        ps = src+os.sep+spk
        pso = des+os.sep+spk
        if spk not in fdic:
            fdic[spk] = {}
        if not os.path.exists(pso):
            os.makedirs(pso)
        for tone in ["1", "2", "3", "4"]:
            fin = ps+os.sep+tone
            fout = pso+os.sep+tone
            
            glob_y, local_y, local_yhat = evaluate_manually(fin, fout, md, num_steps, skip_steps, cur_ans, ave_ans)
            line = "spk {} tone {} glob_y {} local_y {} local_yhat {}\n".format(spk, tone, glob_y, local_y, local_yhat)
            fp.write(line)
            
            
    fp.close()


# In[ ]:


prefix = "/disk2/pwj/workspace/pitch-range/src/model"
mean_md = prefix+"/base_mean_mds/model_v10.h5"
src = prefix+"/103_input"
mean_des = prefix+"/103_mean_output"
##predict mean
predict(mean_md, src, mean_des, 0, 2)

#predict std
std_md = prefix+"/base_span_mds/model_v10.h5"
std_des = prefix+"/103_span_output"
predict(std_md, src, std_des, 1, 3)


# In[ ]:




