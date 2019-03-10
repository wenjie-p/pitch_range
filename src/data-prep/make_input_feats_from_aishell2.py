import os
import codecs
import sys

def load_data(f):

    fp = codecs.open(f, "r", encoding = "utf8")
    content = fp.readlines()
    fp.close()

    return content

def make(fout, sdic, content):

    mean = std = 0
    fp = codecs.open(fout, "w", encoding = "utf8")
    for line in content:
        line = line.replace("]", "")
        line = line.strip().split()
        if len(line) < 20:
            spk = line[0][1:6]
            if spk not in sdic:
                print("{} not in the dic!".format(spk))
            mean = sdic[spk]["mean"]
            std = sdic[spk]["std"]
        else:
            line.extend([mean, std])
            fp.write(" ".join(line) + "\n")

    fp.close()

def make_samples_for_training(feats, sdic, des):

    for usg in os.listdir(feats):
        fin = feats+os.sep+usg
        content = load_data(fin)
        fout = des+os.sep+usg
        make(fout, sdic, content)

def make_spk_info_dic(dr):

    dic = {}
    means = []
    stds = []
    for usg in os.listdir(dr):
        fin = dr+os.sep+usg
        content = load_data(fin)
        for line in content:
            line = line.strip().split()
            spk = line[0]
            mean = float(line[1])
            std = float(line[2])

            if spk in dic:
                print("Error! {} in usg: {} already in the dic !".format(spk, usg))

            dic[spk] = {}
            dic[spk]["mean"] = mean
            dic[spk]["std"] = std

            means.append(mean)
            stds.append(std)
    print("The max of mean and std is {} {}".format(max(means), max(stds)))
    print("The min of mean and std is {} {}".format(min(means), min(stds)))
    return dic

def make_samples(feats, targets, des):

    sdic = make_spk_info_dic(targets)
    
    make_samples_for_training(feats, sdic, des)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: {} vad_feats targets_per_spk des".format(sys.argv[0]))
        exit(0)

    make_samples(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
