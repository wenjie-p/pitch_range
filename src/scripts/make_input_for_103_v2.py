#!/usr/bin/env python
# coding: utf-8

# In[1]:


import codecs
import os
import numpy as np

def load_data(fin):
    
    fp = codecs.open(fin, "r", encoding = "utf8")
    data = fp.readlines()
    fp.close()
    
    return data

def make_data_idxs(data):
    
    idxs = []
    for i in range(len(data)):
        line = data[i]
        if "[" in line:
            idxs.append(i)
            
    return idxs

def load_dic(src):
    
    dic = {}
    for f in os.listdir(src):
        if f.startswith("tone"):
            tone = f.split("_")[1]
            fin = src+os.sep+f
            data = load_data(fin)
            for line in data:
                line = line.strip().split()
                spk = line[0]
                mean = float(line[1])
                std = float(line[2])
                if spk not in dic:
                    dic[spk] = {}
                if tone not in dic[spk]:
                    dic[spk][tone] = {}
                dic[spk][tone]["mean"] = mean
                dic[spk][tone]["std"] = std
                
    return dic

def make_batches(samples, mean, std, uttid):
    
    tot = len(samples)
    num_steps = 30
    if tot < num_steps:
        print("Utterance: {} with length: {}".format(uttid, tot))
        return
    
    cur = 0
    batches = []
    while cur+num_steps<tot:
        sample = samples[cur:cur+num_steps]
        batch = []
        for s in sample:
            s = s.strip().split()
            vec = [float(v) for v in s]
            vec.extend([mean, std])
            batch.append(vec)
        batches.append(batch)
        cur += 1
        
    return batches

def make_input_for_103(src, vad_feats, des):
    
    data = load_data(vad_feats)
    idxs = make_data_idxs(data)
    targets_dic = load_dic(src)
    
    fdic = {}
    for spk in targets_dic:
        fdic[spk] = {}
        for tone in targets_dic[spk]:
            path = des+"/"+spk
            if not os.path.exists(path):
                os.makedirs(path)
            fout = path+"/"+tone
            if tone not in fdic[spk]:
                fdic[spk][tone] = {}
            fdic[spk][tone]["fp"] = fout
            fdic[spk][tone]["data"] = []
    
    for i in range(len(idxs)):
        beg = idxs[i]
        uttid = data[beg].strip().split()[0]
        spk = uttid.split("_")[0]
        tone = uttid[-1]
        beg += 1
        end = 0
        mean = targets_dic[spk][tone]["mean"]
        std  = targets_dic[spk][tone]["std"]
        if i < len(idxs)-1:
            end = idxs[i+1] - 1
        else:
            end = len(data) - 2
        samples = data[beg: end]
        batches = make_batches(samples, mean, std, uttid)
        fdic[spk][tone]["data"].extend(batches)
#         for sample in samples:
#             sample = sample.strip().split()
#             if len(sample) != 40:
#                 print("Error! in file:{}".format(uttid))
#                 print(sample)
#                 exit(0)
#             vec = [float(v) for v in sample]
#             vec.extend([mean, std])
#             fdic[spk][tone]["data"].append(vec)
            
    for spk in fdic:
        for tone in fdic[spk]:
            data = fdic[spk][tone]["data"]
            fout = fdic[spk][tone]["fp"]
            np.savetxt(fout, data)
                
    


# In[ ]:


make_input_for_103("/disk1/wenjie/workspace/pitch_range/src/data/103/103_targets", "/disk1/wenjie/workspace/pitch_range/src/103_after_vad", "/disk1/wenjie/workspace/pitch_range/src/data/103_input_v2")


# In[ ]:




