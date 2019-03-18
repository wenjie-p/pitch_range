#!/usr/bin/env python
# coding: utf-8

# In[7]:

import sys


import h5py
import numpy as np
#des = "../"
#hdf = h5py.File(des+"aaghhag.h5", "w")
#dst = hdf.create_dataset("data", shape = (0,2,1), maxshape=(None, 2,1), chunks = True)
#ele = np.array([[[1],[4]],[[2],[5]]])
#print(ele.shape)
#dst.resize((ele.shape[0]+dst.shape[0]), axis = 0)
#dst[-ele.shape[0]:] = ele
#for i in range(3):
#    dst.resize((ele.shape[0]+dst.shape[0]), axis = 0)
#    dst[-ele.shape[0]:] = ele
#hdf.close()
#
#
## In[ ]:


import codecs

#fin = "/disk2/pwj/workspace/pitch-range/data/aishell-2/train"
#fp = codecs.open(fin, "r", encoding = "utf8")
#data = fp.readlines()
#fp.close()
#
#idxs = []
#for i in range(len(data)):
#    line = data[i]
#    if "[" in line:
#        idxs.append(str(i))
#print(len(idxs))
#with codecs.open("/disk2/pwj/workspace/pitch-range/data/aishell-2/train.idx", "w", encoding = "utf8") as f:
#    idxs_ = "\n".join(idxs)
#    f.write(idxs_)
    


# In[2]:


import numpy as np
import random

random.seed(42)


def build_dic(fin):
    
    data = load_data(fin)
    tot = len(data)
    trn = int(tot*0.8)
    trn_data = data[:trn]
    dev_data = data[trn:]
    print("train: {} dev: {}".format(len(trn_data), len(dev_data)))
    dic = {}
    for ele in trn_data:
        ele = ele.strip().split()
        spk = ele[0]
        mean = ele[1]
        std  = ele[2]
        dic[spk] = {}
        dic[spk]["usg"] = "train"
        dic[spk]["mean"] = mean
        dic[spk]["std"] = std
        
    for ele in dev_data:
        ele = ele.strip().split()
        spk = ele[0]
        mean = ele[1]
        std  = ele[2]
        dic[spk] = {}
        dic[spk]["usg"] = "dev"
        dic[spk]["mean"] = mean
        dic[spk]["std"] = std
        
    return dic
   
    
def gen_sen_vec(data, idxs, dic):
    
    tot = len(idxs)
    idxs = [int(j) for j in idxs]

    
    for i in range(tot):
        beg = idxs[i]
        info = data[beg].strip().split()
        uttid = info[0]
        spk = uttid[1:6]
        if spk not in dic:
            continue
        usg = dic[spk]["usg"]
        # 
        mean = float(dic[spk]["mean"])
        std  = float(dic[spk]["std"])
        if i == tot - 1:
            end = tot
        else:
            end = idxs[i+1]
            
        beg += 1
        samples = data[beg:end-1]
        batches = make_batches(samples, mean, std)
        
        yield batches, usg, uttid, spk
        
def make_batches(samples, mean, std):
    
    batches = []
    cur = 0
    tot = len(samples)
    vecs = []
    for sample in samples:
        vec = sample.strip().split()
        if len(vec) != 40:
            print("Error, vec dim is {}".format(len(vec)))
        #vec = [float(e) for e in vec]
        vec.extend([mean, std])
        vec = [float(v) for v in vec]
        vecs.append(vec)
        
    while cur+ 30<tot:
        batch = vecs[cur:cur+30]
        batches.append(batch)
        cur+=5
    
    return np.array(batches)

def load_data(fin):
    fp = codecs.open(fin, "r", encoding = "utf8")
    data = fp.readlines()
    fp.close()
    return data

def work(src, targets, des):
    dic = build_dic(targets+"/train")
    print("Spk dic built...")
    data = load_data(src+"/train")
    print("Train data loaded...")
    idxs = load_data(src+"/train.idx")
    print("Train idxs loaded...")
    
    generator = gen_sen_vec(data, idxs, dic)
    print("Starts generates data...")
    hdf_trn = h5py.File(des+"/train.h5", "w")
    dst_trn = hdf_trn.create_dataset("data", shape = (0, 30, 42), maxshape = (None, 30, 42), chunks = True, dtype = "f")

    hdf_dev = h5py.File(des+"/dev.h5", "w")
    dst_dev = hdf_dev.create_dataset("data", shape = (0, 30, 42), maxshape = (None, 30, 42), chunks = True, dtype = "f") 

    sen = 0
    trn_spk = set()
    dev_spk = set()
    for i in range(len(idxs)):
        try: 
            batches, usg, uttid, spk = next(generator)
        except Exception as e:
            print("e")
            break
        if len(batches.shape) != 3:
            print("{} donot make a batch".format(uttid))
            continue
#        print(batches.shape)
        if usg == "train":
            if spk not in trn_spk:
                trn_spk.add(spk)
            dst_trn.resize((dst_trn.shape[0]+batches.shape[0]), axis = 0)
            dst_trn[-batches.shape[0]:] = batches
            print("processing train")
            sen += 1
        elif usg == "dev":
            if spk not in dev_spk:
                dev_spk.add(spk)
            dst_dev.resize((dst_dev.shape[0]+batches.shape[0]), axis = 0)
            dst_dev[-batches.shape[0]:] = batches
            print("processing dev")
            sen += 1

    hdf_trn.close()
    hdf_dev.close()
    print("Summary: done with {} sentences, trn_spk: {} dev_spk: {}".format(sen, len(trn_spk), len(dev_spk)))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: {} vad_feats targets des".format(sys.argv[0]))
        exit(0)

    work(sys.argv[1], sys.argv[2], sys.argv[3])

