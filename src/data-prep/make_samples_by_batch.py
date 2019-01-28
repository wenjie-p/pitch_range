import os
import sys
import codecs

def load_data(fin):

    f = codecs.open(fin, "r", encoding = "utf8")
    data = f.readlines()
    f.close()

    return data
def preprocessing_dir(dr):

    if os.path.isdir(dr):
        for fp in os.listdir(dr):
            f = dr+"/"+fp
            os.unlink(f)
    else:
        os.makedirs(dr)

def make_samples(fdic, out, num_steps, skip_steps, input_dim):

    preprocessing_dir(out)
    
    fout_dic = {}
    tot = 0
    for spk in fdic:
        fin = fdic[spk]["fin"]
        data = load_data(fin)
        usg = fdic[spk]["usg"]
        miu = fdic[spk]["miu"]
        std = fdic[spk]["std"]
        fout = out+"/"+usg
        suffix = " {} {}\n".format(miu, std)

        if usg not in fout_dic:
            fout = codecs.open(fout, "w", encoding = "utf8")
            fout_dic[usg] = fout
        f = fout_dic[usg]

        for line in data:
            line = line.strip()
            input_dim_ac = len(line.split())
            # input dim is 40
            if input_dim_ac != 43:
                print("Error: input_dim in cfg: {} vs actural input_dim: {}".format(input_dim, input_dim_ac))
            line = line + suffix
            f.write(line)

    for k in fout_dic:
        f = fout_dic[k]
        f.close()

def load_targets(dr):

    fdic = {}

    for f in os.listdir(dr):
        fp = dr+"/"+f
        data = load_data(fp)
        for line in data:
            line = line.strip().split()
            spk = line[0]
            miu = float(line[1])
            std = float(line[2])
            if spk in fdic:
                print("Error! {} in the dic !".format(spk))
            fdic[spk] = {}
            fdic[spk]["miu"] = miu
            fdic[spk]["std"] = std

    return fdic

def make_samples_helper(feats, targets):

    fdic  = load_targets(targets)

    fd = {}
    for usg in os.listdir(feats):
        pu = feats + "/" + usg
        for spk in os.listdir(pu):
            fin = pu+"/"+spk
            fd[spk] = {}
            fd[spk]["fin"] = fin
            fd[spk]["miu"] = fdic[spk]["miu"]
            fd[spk]["std"] = fdic[spk]["std"]
            fd[spk]["usg"] = usg
    
    return fd

def load_cfg(fp):

    with codecs.open(fp, "r", encoding = "utf8") as f:
        content = f.readlines()
        params = {}
        for line in content:
            line = line.strip().split("=")
            params[line[0]] = int(line[1])

        return params["num_steps"], params["skip_steps"], params["input_dim"] 

def make_samples_by_batch(feats, targets, out):

    num_steps, skip_steps, input_dim = load_cfg("../conf/nn.conf")

    fdic = make_samples_helper(feats, targets)

    make_samples(fdic, out, num_steps, skip_steps, input_dim)
    

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: {} feats_dir, targets_dir, output_dir".format(sys.argv[0]))
        exit(0)

    make_samples_by_batch(sys.argv[1], sys.argv[2], sys.argv[3])
