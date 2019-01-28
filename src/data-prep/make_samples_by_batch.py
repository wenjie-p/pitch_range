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
    zeros = ["0.0"] * (input_dim - 2)
    
    padding_line = " ".join(zeros)

    fout_dic = {}
    counter = {"train": 0, "dev": 0, "test": 0}
    for spk in fdic:
        fin = fdic[spk]["fin"]
        data = load_data(fin)
        usg = fdic[spk]["usg"]
        length = len(data)
        cur_idx = 0
        miu = fdic[spk]["miu"]
        std = fdic[spk]["std"]
        fout = out+"/"+usg
        y = "{} {}".format(miu, std)+" "+padding_line

        if usg not in fout_dic:
            fout = codecs.open(fout, "w", encoding = "utf8")
            fout_dic[usg] = fout
        f = fout_dic[usg]

        while (cur_idx + num_steps) < length:

            batch = data[cur_idx: cur_idx+num_steps]
            for line in batch:
                input_dim_ac = len(line.strip().split())
                if input_dim_ac != input_dim:
                    print("Error: input_dim in cfg: {} vs actural input_dim: {}".format(input_dim, input_dim_ac))
                f.write(line)
                counter[usg]+=1
            f.write(y+"\n")
            cur_idx += skip_steps

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
