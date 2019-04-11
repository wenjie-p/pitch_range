import os
import codecs
import sys
import matlab.engine as me
import matlab
import scipy.io
import numpy as np

def load_data(fin):

    f = codecs.open(fin, "r", encoding = "utf8")
    data = f.readlines()
    f.close()

    return data

def main(scp):

    data = load_data(scp)
    params = []
    bot = top = 0
    for line in data:
        line = line.strip().split()
        fin = line[0]
        gender = line[1]
        fout = line[2]

        des = fout.split("/")[:-1]
        des = "/".join(des)

        if gender == "M":
            bot = 50.0
            top = 300.0
            #continue
        else:
            #continue
            bot = 75.0
            top = 500.0
        #param = [[fin], [bot], [top], [fout]]
        #param = [fin, des, fout, bot, top]
        param = [fin, des, fout, bot, top]
        params.append(param)

    eng = me.start_matlab()
    eng.f0_extraction_parallel(params, nargout = 0)
    eng.quit()

    print("Done with f0 extraction...")

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: {} usg.scp".format(sys.argv[0]))
        exit(0)

    main(sys.argv[1])
