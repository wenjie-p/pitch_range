import os
import sys
import codecs

def load_data(fin):

    f = codecs.open(fin, "r", encoding = "utf8")
    data = f.readlines()
    f.close()

    return data

def make_samples(fdic, out, num_steps, skip_steps):
    
    for spk in fdic:
        fin = fdic[spk]["fin"]
        data = load_data(fin)
        usg = fdic[spk]["usg"]
        length = len(data)
        cur_idx = 0
        miu = fdic[spk]["miu"]
        std = fdic[spk]["std"]
        fout = out+"/"+usg
        y = "{} {}".format(miu, std)
        f = codecs.open(fout, "a", encoding = "utf8")
        while cur_idx + num_steps < length:
            batch = data[cur_idx: cur_idx+num_steps]
            for line in batch:
                f.write(line)
            f.write(y+"\n")
            cur_idx += skip_steps

        f.close()

def make_samples_helper(feats, targets):
    miu, std = load_targets(targets)

    fd = {}
    for usg in os.listdir(feats):
        pu = feats + "/" + usg
        for spk in os.listdir(pu):
            fin = pu+"/"+spk
            k = ""
            fd[k] = {}
            fd[k]["fin"] = fin
            fd[k]["miu"] = miu[k]
            fd[k]["std"] = std[k]
            fd[k]["usg"] = usg
    
    return fd

def make_samples_by_batch(feats, targets, out, num_steps, skip_steps):

    fdic = make_samples_helper(feats, targets)

    make_samples(fdic, out, num_steps, skip_steps)
    

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: {} feats_dir, targets_dir, output_dir, num_steps, skip_steps".format(sys.argv[0]))
        exit(0)

    make_samples_by_batch(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
