#!/usr/bin/env python
# coding: utf-8

# In[6]:


import codecs
import os
import numpy as np
import json


def load_data(fin):
    
    fp = codecs.open(fin, "r", encoding = "utf8")
    data = fp.readlines()
    fp.close()
    
    return data

def get_pitches(data):
       
    pitches = [float(e.strip()) for e in data]
    pitches = [np.log10(e) for e in pitches if e > 10]
    
    return pitches  

def make_targets_for_103(src, des):
    
    toneinfo = {}
    spkinfo = {}
    for f in os.listdir(src):
        fin = src+os.sep+f
        data = load_data(fin)
        pitches = get_pitches(data)
        spk = f.split("_")[0]
        tone = f.replace(".pitch", "")[-1]
        try:
            val = int(tone)
        except:
            #print("errors occur in {}".format(f))
            tone = 0
            continue
            
        if spk not in spkinfo:
            print("processing spk {}".format(spk))
            spkinfo[spk] = []
        if tone not in toneinfo:
            toneinfo[tone] = {}
        if spk not in toneinfo[tone]:
            toneinfo[tone][spk] = []
        spkinfo[spk].extend(pitches)
        toneinfo[tone][spk].extend(pitches)
    
    fspk = des+"/103_all_test.targets"
    fp = codecs.open(fspk, "w", encoding = "utf8")

    tdic = {}
    for spk in spkinfo:
        pitches = spkinfo[spk]
        mean = np.mean(pitches)
        std = np.std(pitches)
        line = "{} {} {}\n".format(spk, mean, std)
        fp.write(line)
        t_mean = []
        t_std  = []
        tdic[spk] = {}
        for t in ["1", "2", "3", "4"]:
            pitches = toneinfo[t][spk]
            tt_mean = np.mean(pitches)
            tt_std  = np.std(pitches)
            tdic[spk][t] = {"mean": tt_mean, "std": tt_std}
        
    fp.close()
    
    for tone in ["1", "2", "3", "4"]:
        fout = "{}/103_{}_.targets".format(des, tone)
        fp = codecs.open(fout, "w", encoding = "utf8")
        print("writting data into file {}".format(fout))
        for spk in toneinfo[tone]:
            pitches = toneinfo[tone][spk]
            mean = np.mean(pitches)
            std = np.std(pitches)
            line = "{} {} {}\n".format(spk, mean, std)
            fp.write(line)
        fp.close()
    
    ftone = des+"/103_all_tone.json"
    ft = codecs.open(ftone, "w", encoding = "utf8")
    json.dump(tdic, ft)
    ft.close()
    


# In[7]:


prefix = "/home/pwj/103/"
src = prefix+"/wav"

make_targets_for_103(src, prefix)

