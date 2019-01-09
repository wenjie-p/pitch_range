import codecs
import os
import sys
import logging
import numpy as np

def main(src, des):

    for usg in os.lisrdir(src):
        dr1 = src+os.sep+usg
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
            miu = pitches.mean()
            sigma = pitches.var()

            if not os.isdir(des+os.sep+usg):
                os.mkdirs(des+os.sep+usg)
            fout = des+os.sep+usg+spk+".info"
            fp = codecs.open(fout, "w", encoding = "utf8")
            fp.write(miu+"\n")
            fp.write(sigma+"\n")
            fp.close()


if __name__ == "__main__":
    if len(sys.argv) != :
        print("Usage: {} src des".format(sys.argv[0]))
        exit(1)

    main(sys.argv[1], sys.argv[2])
