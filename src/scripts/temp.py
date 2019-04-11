import codecs

fin = "./1"
fp = codecs.open(fin, "r", encoding = "utf8")
data = fp.readlines()
fp.close()

dic = {}

for line in data:
    line = line.strip().split("_")
    spk = line[0]
    tt = line[1]
    if spk not in dic:
        dic[spk] = {}

    if tt not in dic[spk]:
        dic[spk][tt] = 0

    dic[spk][tt] += 1

path = "/disk2/pwj/workspace/pitch-range/src/model/103_mean_output_final_v4"
spks = set()
import os
for s in os.listdir(path):
    if s != "info.txt":
        spks.add(s)

mono = 0
for spk in spks:
    for tt in dic[spk]:
        line = "spk {} type {} total {}".format(spk, tt, dic[spk][tt])
        if tt.startswith("mono"):
            mono+= dic[spk][tt]
        print(line)
print(mono/35.0)
