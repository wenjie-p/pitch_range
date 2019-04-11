#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import codecs
import numpy as np

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


def load_data(fin):
    
    fp = codecs.open(fin, "r", encoding = "utf8")
    data = fp.readlines()
    fp.close()
    
    return data

def calculate_mape_per_batch(data):
    
    ltd_mape, lstm_mape = [], []
    for i in range(len(data)-1):
        line = data[i].strip().split()
        lstm_val = float(line[-1])
        ltd_val = float(line[5])
        lstm_mape.append(lstm_val)
        ltd_mape.append(ltd_val)
    
    return np.mean(ltd_mape), np.mean(lstm_mape)

def calculate_mape_for_103(src, fout):
    
    #udic = build_utt_dic("/home/pwj/103/wav")
    
    tot_mape = {"lstm": {}, "ltd": {}}
    tone_mape = {}
    lstm_mape = {}
    ltd_mape = {}
    for spk in os.listdir(src):
        if spk == "info.txt":
            continue
        dspk = src+os.sep+spk
        tot_mape["lstm"][spk] = []
        tot_mape["ltd"][spk] = []
        
        lstm_mape[spk] = {}
        ltd_mape[spk] = {}
        for tone in os.listdir(dspk):
            fin = dspk+os.sep+tone
            data = load_data(fin)
            ltd_val, lstm_val = calculate_mape_per_batch(data)
            lstm_mape[spk][tone] = lstm_val
            ltd_mape[spk][tone] = ltd_val

            tot_mape["lstm"][spk].append(lstm_val)
            tot_mape["ltd"][spk].append(ltd_val)
            
            if tone not in tone_mape:
                tone_mape[tone] = {"lstm": [], "ltd": []}
            tone_mape[tone]["lstm"].append(lstm_val)
            tone_mape[tone]["ltd"].append(ltd_val)
    
    fp = codecs.open(fout, "w", encoding = "utf8")
    lstm_tot, ltd_tot = [], []
    for spk in ltd_mape:
        print(spk)
        for tone in ["1h5", "2h5", "3h5", "4h5"]:
            line = "spk {} tone {} ltd_mape {} lstm_mape {}\n".format(spk, tone, ltd_mape[spk][tone], lstm_mape[spk][tone])
            fp.write(line)
        lstm_val = np.mean(tot_mape["lstm"][spk])
        ltd_val = np.mean(tot_mape["ltd"][spk])
        line = "spk {} ltd_mape {} lstm_mape {}\n".format(spk, ltd_val, lstm_val )
        fp.write(line)
        lstm_tot.append(lstm_val)
        ltd_tot.append(ltd_val)

    line = "Summary ltd_mape {} lstm_mape {}\n".format(np.mean(ltd_tot), np.mean(lstm_tot))
    fp.write(line)
    for tone in tone_mape:
        lstm_val = np.mean(tone_mape[tone]["lstm"])
        ltd_val = np.mean(tone_mape[tone]["ltd"])
        line = "Tone {} ltd_mape {} lstm_mape {}\n".format(tone, ltd_val, lstm_val)
        fp.write(line)
    fp.close()


# In[6]:


prefix = "/disk2/pwj/workspace/pitch-range/src/model/"
src1 = prefix+"103_mean_output_final_v5"
fout1 = prefix+"mape_103_mean_final_v5"
src = prefix+"103_span_output_final_v5"
fout = prefix+"mape_103_span_final_v5"
calculate_mape_for_103(src, fout)
calculate_mape_for_103(src1, fout1)

