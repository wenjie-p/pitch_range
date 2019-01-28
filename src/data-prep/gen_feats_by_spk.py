import sys
import os
import codecs

def preprocessing_dir(path):
    
    if os.path.isdir(path):
        for f in os.listdir(path):
            fp = path+"/"+f
            os.unlink(fp)
    else:
        os.makedirs(path)

def gen_feats(fin, des, usg):

    # first, we should clear the data in the des folder.
    path = des+"/"+usg
    preprocessing_dir(path)
    
    spk_dic = {}

    with codecs.open(fin, "r", encoding = "utf8") as f:
        content = f.readlines()
        cur_spk = ""
        sen = 0
        for line in content:
            line = line.strip().split()
            # fbank with dim 23
            if len(line) < 23:
                spk = line[0][7:11]
                cur_spk = spk
                sen = (sen+1)%4
                print("Dealing with spk: {}".format(spk))
                fout = path+"/"+spk
                if spk not in spk_dic:
                    fout = codecs.open(fout, "w", encoding = "utf8")
                    spk_dic[spk] = fout
                continue
            if line[-1] == "]":
                line[-1] = str(sen)
            else:
                line.append(str(sen))
            line = " ".join(line) + "\n"
            spk_dic[cur_spk].write(line)

        for spk in spk_dic:
            spk_dic[spk].close()

def gen_feats_by_spk(src, des):

    for usg in os.listdir(src):
        fin = src+"/"+usg
        gen_feats(fin, des, usg)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} raw_data feats_by_spk".format(sys.argv[0]))
        exit(0)

    gen_feats_by_spk(sys.argv[1], sys.argv[2])
