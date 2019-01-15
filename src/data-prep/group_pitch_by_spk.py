import codecs
import os
import sys
import logging
import numpy as np

def main(src, des):

    for usg in os.listdir(src):
        dr1 = src+os.sep+usg
        if not os.path.isdir(dr1):
            continue
        for spk in os.listdir(dr1):
            dr2 = dr1+os.sep+spk
            pitches = []
            for f in os.listdir(dr2):
                fp = dr2+os.sep+f
                fp =codecs.open(fp, "r", encoding = "utf8")
                content = fp.readlines()
                fp.close()

                for line in content[1:]:
                    line = line.strip().split(",")
                    p = 0
                    try:
                        p = float(line[1])
                    except:
                        continue
                    pitches.append(p)
            pitches = np.array(pitches)
#            miu = pitches.mean()
#            sigma = pitches.var()
#
            dr = "/".join([des, usg])
            if not os.path.isdir(dr):
                os.makedirs(dr)
            fout = dr+"/"+spk+".info"
            fp = codecs.open(fout, "w", encoding = "utf8")
#            fp.write(miu+"\n")
#            fp.write(sigma+"\n")
            pitches = ["{: .3f}".format(x) for x in pitches]
            fp.write(" ".join(pitches))
            fp.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("This script grouping the pitch data in terms of individual speaker.")
        print("Usage: {} src des".format(sys.argv[0]))
        exit(1)

    main(sys.argv[1], sys.argv[2])
