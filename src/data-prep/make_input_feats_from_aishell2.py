import os
import random
import codecs
import sys

random.seed(42)

def split_data(dic):

    train = dic["train"]
    tot = len(train)
    dev = random.sample(train, len(train) * 0.2)
    test = dic["test"]

    new = {}

    for ele in dev:
        spk = ele["spk"]
        mean = ele["mean"]
        std = ele["std"]
        new[spk] = {}
        new[spk]["mean"] = mean
        new[spk]["std"] = std
        new[spk]["usg"] = "dev"
    co = 0
    for ele in train:
        spk = ele["spk"]
        mean = ele["mean"]
        std = ele["std"]
        if spk in new:
            co += 1
            continue
        new[spk] = {}
        new[spk]["mean"] = mean
        new[spk]["std"] = std
        new[spk]["usg"] = "train"
    if co != tot:
        print("Error when doing split data co: {} vs tot:{}".format(co, tot))
        exit(0)
    for ele in test:
        spk = ele["spk"]
        mean = ele["mean"]
        std = ele["std"]
        new[spk] = {}
        new[spk]["mean"] = mean
        new[spk]["std"] = std
        new[spk]["usg"] = "test"

    return new

def load_data(f):

    fp = codecs.open(f, "r", encoding = "utf8")
    content = fp.readlines()
    fp.close()

    return content

def make(fdic, sdic, content):

    mean = std = 0
    fp = ""
    for line in content:
        line = line.replace("]", "")
        line = line.strip().split()
        if len(line) < 20:
            spk = line[0][1:6]
            if spk not in sdic:
                print("{} not in the dic!".format(spk))
            mean = sdic[spk]["mean"]
            std = sdic[spk]["std"]
            usg = sdic[spk]["usg"]
            fp = fdic[usg]
        else:
            line.extend([mean, std])
            line = [str(e) for e in line]
            fp.write(" ".join(line) + "\n")


def make_samples_for_training(feats, sdic, des):
    
    fdic = {}
    for usg in ["train", "dev", "test"]:
        fout = des+"/"+usg
        fdic[usg] = codecs.open(fout, "w", encoding = "utf8")

    for usg in os.listdir(feats):
        fin = feats+os.sep+usg
        content = load_data(fin)
        make(fdic, sdic, content)
    
    for usg in fdic:
        fdic[usg].close()

def make_spk_info_dic(dr):

    dic = {}
    means = []
    stds = []
    for usg in os.listdir(dr):
        fin = dr+os.sep+usg
        content = load_data(fin)
        dic[usg] = []
        for line in content:
            line = line.strip().split()
            spk = line[0]
            mean = float(line[1])
            std = float(line[2])

            ele = {}
            ele["mean"] = mean
            ele["std"] = std
            ele["spk"] = spk
            dic[usg].append(ele)
            means.append(mean)
            stds.append(std)
    print("The max of mean and std is {} {}".format(max(means), max(stds)))
    print("The min of mean and std is {} {}".format(min(means), min(stds)))

    dic = split_data(dic)

    return dic

def make_samples(feats, targets, des):

    sdic = make_spk_info_dic(targets)
    
    make_samples_for_training(feats, sdic, des)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: {} vad_feats targets_per_spk des".format(sys.argv[0]))
        exit(0)

    make_samples(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
