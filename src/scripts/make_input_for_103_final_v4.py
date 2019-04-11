#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import codecs
import os
import numpy as np
import h5py

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
    data = load_data(src)
    for line in data:
        line = line.strip().split()
        spk = line[0]
        mean = float(line[1])
        std = float(line[2])
        if spk not in dic:
            dic[spk] = {}
        for tone in ["1", "2", "3", "4"]:
            dic[spk][tone] = {}
            dic[spk][tone]["mean"] = mean
            dic[spk][tone]["std"] = std
                
    return dic

def make_batches(samples, mean, std, ltd_mean, ltd_std, uttid):
    
    tot = len(samples)
    num_steps = 30
    if tot < num_steps:
        print("Utterance: {} with length: {}".format(uttid, tot))
        return []
    
    cur = 0
    batches = []
    while cur+num_steps<tot:
        sample = samples[cur:cur+num_steps]
        batch = []
        for s in sample:
            s = s.strip().split()
            vec = [float(v) for v in s]
            vec.extend([mean, std, ltd_mean, ltd_std])
            batch.append(vec)
        batches.append(batch)
        cur += 1
        
    return np.array(batches)

def build_utt_dic(src):
    
    dic = {}
    for f in os.listdir(src):
        fin = src+os.sep+f
        uttid = f.replace(".pitch", "")
        data = load_data(fin)
        pitches = [float(e.strip()) for e in data]
        pitches = [p for p in pitches if p > 10]
        mean = np.mean(pitches)
        std = np.std(pitches)
        dic[uttid] = {}
        dic[uttid]["mean"] = mean
        dic[uttid]["std"] = std
        
    return dic

def make_input_for_103(src, vad_feats, des):
    
    data = load_data(vad_feats)
    print("feats loaded...")
    idxs = make_data_idxs(data)
    print("idxs built ...")
    tdic = load_dic(src)
    print("targets dic loaded...")
#    udic = build_utt_dic("/home/pwj/103/wav")
#    print("uttdic built...")
    prefix = "/home/pwj/103/wav"
    fdic = {}

    dic = set()
    spkdic = {}
    for i in range(len(idxs)):
        beg = idxs[i]
        uttid = data[beg].strip().split()[0]
        spk = uttid.split("_")[0]
        if spk not in tdic:
            print("spk: {} not in targets".format(spk))
            continue
        if spk not in dic:
            print("processing spk: {}".format(spk))
            dic.add(spk)
        if spk not in spkdic:
            spkdic[spk] = 0
        tone = uttid[-1]
        if tone not in set(["1", "2", "3", "4"]):
            continue

        if spk not in fdic:
            fdic[spk] = {}
        if tone not in fdic[spk]:
            fdic[spk][tone] = {}
            path = des+"/"+spk
            if not os.path.exists(path):
                os.makedirs(path)
            fout = path+"/"+tone+"h5"
            hdf = h5py.File(fout, "w")
            dst = hdf.create_dataset("data", shape = (0, 30, 44), maxshape = (None, 30, 44), chunks = True, dtype = "f")
            fdic[spk][tone]["data"] = dst
            fdic[spk][tone]["fp"] = hdf
        beg += 1
        end = 0
        mean = tdic[spk][tone]["mean"]
        std  = tdic[spk][tone]["std"]
        uttdata = load_data("{}/{}.pitch".format(prefix, uttid))
        uttpitches = [float(e.strip()) for e in uttdata]
        pitches = [np.log10(p) for p in uttpitches if p > 10]
        ltd_mean = np.mean(pitches)
        ltd_std = np.std(pitches)
        if i < len(idxs)-1:
            end = idxs[i+1] - 1
        else:
            end = len(data) - 2
        samples = data[beg: end]
        batches = make_batches(samples, mean, std, ltd_mean, ltd_std, uttid)
        if len(batches) == 0:
            spkdic[spk] += 1
            continue
        print(batches.shape, fdic[spk][tone]["data"].shape)
        fdic[spk][tone]["data"].resize((fdic[spk][tone]["data"].shape[0]+batches.shape[0]), axis = 0)
        fdic[spk][tone]["data"][-batches.shape[0]:] = batches
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
            fdic[spk][tone]["fp"].close()

    fp = codecs.open("/home/pwj/103_spk_less_300ms.json", "w", encoding = "utf8")
    json.dump(spkdic, fp)
    fp.close()
    print("Done with data processing")
                
    


# In[ ]:


src = "/home/pwj/103/103_all_test.targets"
feats = "/disk2/pwj/workspace/pitch-range/src/103_after_vad"
des = "/disk2/pwj/workspace/pitch-range/src/model/103_input_final_v8/"
make_input_for_103(src, feats, des)

