import os
import json
import sys
import codecs
import matplotlib.pyplot as plt
import numpy as np

def savesta(sta, fsta):
    f = codecs.open(fsta, "w", encoding = "utf8")
    f.close()

def write2file(dic, fsta):

    dis = sorted(dic.items(), key = lambda x:x[1], reverse = True)
    with codecs.open(fsta, "w", encoding = "utf8") as f:
        for ele in dis:
            line = "frequency: {} with count: {}".format(ele[0], ele[1])
            f.write(line+"\n")
            

def data_sta(vec):
    dic = {}
    b = 50
    while b <= 500:
        sub = []
        for e in vec:
            if b-e<=5 and b-e>=0:
#                e = np.log(e)
                sub.append(e)
        dic[b] = sub
        b+= 5
    
    new = []
    for k,v in dic.items():
        if len(v) > 300:
            new.extend(v)
    new = np.array(new)
    miu = new.mean()
    sigma = new.std()

    new = [(x-miu)/sigma for x in new]
    return new

def statistics(fin):

    print(fin)
    f = codecs.open(fin, "r", encoding = "utf8")
    data = f.readline()
    f.close()
    
    data = [float(x) for x in data.strip().split()]
    return data

#    data = data_sta(data)
    #write2file(dic, fsta)

def display(data, fout):

    fig, ax = plt.subplots()
    tot = len(data)
    num_bins = 100
    n, bins, patches = ax.hist(data, num_bins, density = 1)
    fig.savefig(fout)
    plt.close("all")

if __name__ == "__main__":
    f = codecs.open("spk.dic", "r", encoding = "utf8")
    dic = json.load(f)
    f.close()
    spk = {"M": [], "F": []}
    tot = []
    for usg in os.listdir("pitches"):
        
        dr = "pitches/"+usg
        for info in os.listdir(dr):
            fin = dr+"/"+info
            fout = "./pics/"+info.replace(".info", ".png")
            fsta = "sta/"+info.replace("info", "sta")
            sid = info.replace(".info", "").replace("S", "")
            gender = dic[sid]
            sta = statistics(fin)
            spk[gender].extend(sta)
            tot.extend(sta)
            display(sta, fout)
    #        savesta(sta, fsta)
    display(spk["M"], "./male.png")
    display(spk["F"], "./female.png")
    display(tot, "./tot.png")
