import os
import logging
import sys
import subprocess
import codecs
from multiprocessing import Process
import datetime
import json

def process_data(name, fin, gender, des, fout, logger):

    praat = "/home/pwj/workspace/third_party/praat"
    pitch_bot = "50"
    pitch_top = "300"
    
    if gender == "F":
        pitch_bot = "75"
        pitch_top = "500"
    
    if not os.path.isdir(des):
        os.makedirs(des)
    try:
        subprocess.call([praat, "--run", "ExtractF0.praat", fin, fout, des, pitch_top, pitch_bot])
        logger.info("Processor: {} processing file: {} successfully.".format(name, fin))
    except Exception as e:
        logger.warning("Processor: {} processing file: {} with an exception: {}".format(e))


def parallel_work(name, src, gdic, des, logger):

    for spk in src:
        for wav in os.listdir(spk):
            fin = spk+os.sep+wav
            spker = spk.split("/")[-1].replace("S", "")
            gender = gdic[spker]
            fdr = des+os.sep+spker
            fout = wav.replace(".wav", "")
            process_data(name, fin, gender, fdr, fout, logger)

def build_dic(wav):

    dic = {}

    fp = wav+"/../../resource_aishell/speaker.info"
    f = codecs.open(fp, "r", encoding = "utf8")
    content = f.readlines()
    f.close()

    for line in content:
        line = line.strip().split()
        dic[line[0]] = line[1]

    f = codecs.open("spk.dic", "w", encoding = "utf8")
    json.dump(dic, f)
    f.close()

    return dic

def split_data(wav):
    chunks = {}

    for usg in os.listdir(wav):
        spks = []
        dr1 = wav+os.sep+usg
        for spk in os.listdir(dr1):
            dr2 = dr1 + os.sep +spk
            spks.append(dr2)
        tot = len(spks)
        gap = int(tot/8)
        
        chunks[usg] = [spks[x: x+gap] for x in range(0, tot, gap)]

    return chunks

def work(chunks, des, gdic, logger):
    
    procs = []

    threads = []
    for i in range(len(chunks)):
        chunk = chunks[i]
        tname = str(i)
        params = (tname, chunk, gdic, des, logger)
        proc = Process(target = parallel_work, args = params)
        procs.append(proc)
        proc.start()
    
    for p in procs:
        p.join()


def pitch_extraction(wav, des):
    # Get the spkr's gender
    flog = des+os.sep+"pitch_extraction.log"
    logging.basicConfig(filename = flog, level = logging.DEBUG)
    logging.info(datetime.datetime.now())
    chunks = split_data(wav)

    gdic = build_dic(wav)
    #return
    for usg in chunks:
        out = des+os.sep+usg
        work(chunks[usg], out, gdic, logging)

    logging.info(datetime.datetime.now())

if __name__ == "__main__":
    if len(sys.argv) != 3 :
        print("Usage: {} data des".format(sys.argv[0]))
        exit(1)

    pitch_extraction(sys.argv[1], sys.argv[2])
