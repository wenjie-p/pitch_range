import os
import sys
import codecs
import numpy as np

def write2file(params, fout):

    f = codecs.open(fout, "w", encoding = "utf8")
    for line in params:
        f.write(line+"\n")

    f.close()

def get_parameters(fin):

    with codecs.open(fin, "r", encoding = "utf8") as f:
        data = f.readline()
        data = [np.log(float(x)) for x in data.strip().split()]
        data = np.array(data)
        miu = "{: >.3f}".format(data.mean())
        sigma = "{: >.3f}".format(data.std())

        return (miu, sigma)

def make_targets(pdr, des):

    for usg in os.listdir(pdr):
        dr1 = pdr+"/"+usg
        params = []
        for spk in os.listdir(dr1):
            fin = dr1+"/"+spk
            miu, sigma = get_parameters(fin)
            key = spk.replace(".info", "")
            params.append(" ".join([key, miu, sigma]))

        fout = des+"/"+usg
        write2file(params, fout)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("This script statistics the mean and deviation from the samples.")
        print("Usage: {} piches des".format(sys.argv[0]))
        exit(1)

    make_targets(sys.argv[1], sys.argv[2])
