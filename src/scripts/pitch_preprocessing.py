#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import codecs
from sklearn.preprocessing import MinMaxScaler
import os
import random

scaler = MinMaxScaler(feature_range=(-1, 1))

def load_data(fin):
    
    fp = codecs.open(fin, "r", encoding = "utf8")
    data = fp.readlines()
    fp.close()
    
    return data

def extract_params(data):
    
    params = []
    for line in data:
        line = line.strip().split()[1:]
        param = [float(e) for e in line]
        params.append(param)
    
    return params

def get_scaler_from_train(data):
    
    train_scaler = scaler.fit(data)
    
    return train_scaler

def save2file(data_scaled, raw_data, fout):
    
    fp = codecs.open(fout, "w", encoding = "utf8")
    for idx in range(len(raw_data)):
        spk = raw_data[idx].strip().split()[0]
        param = data_scaled[idx].tolist()
        param = " ".join([str(e) for e in param])
        fp.write("{} {}\n".format(spk, param))
        
    fp.close()
    
def scale_data(src, des):
    
    train_data = load_data(src+"/train")
    
    train_params = extract_params(train_data)
    
    scaler = get_scaler_from_train(train_params)
    
    train_params_scaled = scaler.transform(train_params)
    save2file(train_params_scaled, train_data, des+"/train")
    
    test_data = load_data(src+"/test")
    test_params = extract_params(test_data)
    test_params_scaled = scaler.transform(test_params)
    save2file(test_params_scaled, test_data, des+"/test")
    
def save2file(fout, data):
    
    print("the length of {} is {}".format(fout, len(data)))
    
    with codecs.open(fout, "w", encoding = "utf8") as f:
        for line in data:
            f.write(line)
            
def data_split(src, des):
    
    all_pit = []
    for usg in os.listdirs(src):
        fin = src+os.sep+usg
        data = load_data(fin)
        for line in data:
            all_pit.append(line)
            
    tot = len(all_pit)
    trn_tot = int(tot*0.8)
    dev_tot = int(tot*0.1)
    random.shuffle(all_pit)
    
    trn = all_pit[:trn_tot]
    dev = all_pit[trn_tot: trn_tot+dev_tot]
    tes = all_pit[trn_tot+dev_tot:]
    
    save2file(des+"/train" trn)
    save2file(des+"/dev", dev)
    save2file(tes+"/test", tes)
    


# In[ ]:


data_split("/disk2/pwj/workspace/pitch-range/src/data-prep/aishell-2-params", "/disk2/pwj/workspace/pitch-range/src/data-prep/aishell-2-params-8-1-1")


# In[ ]:




