#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import codecs

def load_data(fin):
    
    fp = codecs.open(fin, "r", encoding = "utf8")
    data = fp.readlines()
    fp.close()
    
    return data

def make_wavscp_for_301(src, des, fout):
    
    #data = load_data(fin)
    wavscp = []
    for f in os.listdir(src):
        #line = line.strip().split()
        if f.startswith("C"):
            continue
        spk = f.split("_")[0]
        if not spk[3] == "1":
            continue
        fsrc = src+os.sep+f
        #f   = fsrc.split("/")[-1]
        fdes = des+os.sep+f.replace(".wav", ".pitch")
        gender = spk[1]
        line = "{} {} {}\n".format(fsrc, gender, fdes)
        wavscp.append(line)
    
    fp = codecs.open(fout, "w", encoding = "utf8")
    for line in wavscp:
        fp.write(line)
    fp.close()


def make_wavscp_for_103(src, des, fout):
    
    fp = codecs.open(fout, "w", encoding = "utf8")
    utt2spk = []
    spk2utt = {}
    for f in os.listdir(src):
        gender = "F"
        if f.startswith("M"):
            gender = "M"
        fsrc = src+os.sep+f
        fdes = des+os.sep+f.replace(".wav", ".pitch")
       
        line = "{} {} {}\n".format(fsrc, gender, fdes)
        try:
            fp.write(line)
        except Exception as e:
             print("Processing file: {} falied".format(f))
    fp.close()
    
def make_103_L2_wavscp_for_kaldi(src, des):
    
    fp = codecs.open(des+"/wav.scp", "w", encoding = "utf8")
    ft = codecs.open(des+"/text", "w", encoding = "utf8")
    fu = codecs.open(des+"/utt2spk", "w", encoding = "utf8")
    fs = codecs.open(des+"/spk2utt", "w", encoding = "utf8")
    wav = []
    text = []
    utt2spk = []
    spk2utt = {}
    for f in os.listdir(src):
        if "monosyllable" not in f:
            continue
        fin = src+os.sep+f
        spk = f.split("_")[0]    
        uttid = f.replace(".wav", "")
        line = "{} {}\n".format(uttid, fin)
        wav.append(line)
        text_line = "{} thisis fake\n".format(uttid)
        text.append(text_line)
        utt2spk_line = "{} {}\n".format(uttid, spk)
        utt2spk.append(utt2spk_line)
                
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(uttid)
                
    gar = set()
    wav.sort()
    for w in wav:
        try:
            fp.write(w)
        except :
            gar.add(w.strip().split()[0])
            print(w)
            continue
    
    #fp.write("".join(wav))
    fp.close()

    spk2utt = sorted(spk2utt.items(), key = lambda x:x[0])
    for spkinfo in spk2utt:
        spk = spkinfo[0]
        utts = spkinfo[1]
        utts = [u for u in utts if u not in gar]
        utts.sort()
        utts = " ".join(utts)
        line = "{} {}\n".format(spk, utts)
        try:
            fs.write(line)
        except:
            print(line)
            continue
    fs.close()
    
    
    text.sort()
    for t in text:
        tt = t.strip().split()[0]
        if tt in gar:
            continue
        
        ft.write(t)
    ft.close()
    
    utt2spk.sort()
    for u in utt2spk:
        uttid = u.strip().split()[0]
        if uttid in gar:
            print(uttid)
            continue
        fu.write(u)
    fu.close()


# In[ ]:


#make_wavscp_for_301("/adddisk/DB_public/301/oriWav/", "/home/pwj/301/wav/", "/home/pwj/301/301_wav_v3.scp")
#make_wavscp_for_103("/adddisk/DB_public/103corpus/wav", "/home/pwj/103/wav", "/home/pwj/103/103_wav.scp")
#make_103_L2_wavscp_for_kaldi("/home/pwj/103_L2", "/home/pwj/103_L2_out/")
make_103_L2_wavscp_for_kaldi("/adddisk/DB_public/103corpus/wav/", "/home/pwj/103_v2/")


# In[ ]:




