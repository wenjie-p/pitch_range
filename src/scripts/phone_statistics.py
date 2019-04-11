import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import os
import codecs 

def get_endpoint(x):

    return max(x), min(x)


def postprocessing(info):
    
    info = sorted(info.items(), key = lambda x:x[0])
    x = [xx[0] for xx in info]
    y = [yy[1] for yy in info] 

    return x, y


def draw_his(dic, des):

    for phn in dic:
        info = dic[phn]
        x, y = postprocessing(info)
        x_max, x_min = get_endpoint(x)
        y_max, y_min = get_endpoint(y)
        
        fig, ax = plt.subplots()
        left = np.array(x)
        right = np.array(x)
        bottom = np.zeros(len(left))
        top = bottom + y

        XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T
        barpath = path.Path.make_compound_path_from_polys(XY)
        patch = patches.PathPatch(barpath)
        ax.add_patch(patch)

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom.min(), top.max())
        plt.xlabel("Duration distribution of phone: {}".format(phn))
        plt.ylabel("Count")
        #plt.show()
        
        fig.savefig("{}/{}.png".format(des, phn))
        plt.close("all")



def statistics_phone(data):

    dic = {}

    for line in data:
        line = line.strip().split()
        phn = line[-1]
        dur = int((float(line[-2]) - float(line[-3])) * 1000)
        if phn not in dic:
            dic[phn] = {}
        section = dur/5
        if section not in dic[phn]:
            dic[phn][section] = 0
        dic[phn][section] += 1

    return dic


def draw(data, des):

    dic = statistics_phone(data)

    draw_his(dic, des)

def load(f):

    fp = codecs.open(f, "r", encoding = "utf8")
    data = fp.readlines()
    fp.close()

    return data


def main(ori, des):

    for f in os.listdir(ori):
        fp = ori + os.sep + f
        data = load(fp)
        draw(data, des)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print ("This script is for data statistics")
        print ("Usage: {} data info".format(sys.argv[0]))
        exit(1)

    main(sys.argv[1], sys.argv[2])
