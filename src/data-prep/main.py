import os
import logging
import sys
import subprocess
import codecs
import threading
import time

def process_data(name, fin, gender, des, fout, logger):

    praat = "/home/pwj/workspace/third_party/praat"
    pitch_bot = "50"
    pitch_top = "300"

    if gender == "F":
        pitch_bot = "75"
        pitch_top = "500"
    try:
        subprocess.call([praat, "--run", "ExtractF0.praat", fin, fout, des, pitch_top, pitch_bot])
        logger.info("Thread: {} processing file: {} successfully.".format(name, fin))
    except Exception as e:
        logger.warning("Thread: {} processing file: {} with an exception: {}".format(e))


class myThread(threading.Thread):

    def __init__(self, name, src, gender,  des, logger):
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.data = data
        self.src = src
        self.des = des
        self.logger = logger
        self.gender

    def run(self):

        for wav in src:
            fin = src+os.sep+wav
            fout = wav.replace(".wav", "")
            process_data(self.name, fin, self.gender, self.des, fout, self.logger)

def build_dic(wav):

    dic = {}

    fp = wav+"/../../resource_aishell/speaker.info"
    f = codecs.open(fp, "r", encoding = "utf8")
    content = f.readlines()
    f.close()

    for line in content:
        line = line.strip().split()
        dic[line[0]] = line[1]

    return dic

def split_data(wav):
    spkers = []

    for usg in os.listdir(wav):
    dr1 = wav+os.sep+usg
    for spk in os.listdir(dr1):
        dr2 = dr1 + os.sep +spk
        spkers.append(dr2)
    chunks = [spkers[x: x+100] for x in range(0, len(spkers), 100)]

    return chunks

def work(chunks, des, gdic):
    
    log = des+os.sep+"log"
    logging.basicConfig(filename = log, level = logging.DEBUG)
    logging.info("Start pitch extraction...")

    tb = time.time()
    threads = []
    for i in range(len(chunks)):
        chunk = chunks[i]
        spk = chunk.split("/")[-1].replace("S", "")
        gender = gdic[spk]

        tname = "Thread-{}".foramt(str(i))
        thread = myThread(tname, chunk, gender, des, logger)
        thread.start()
        threads.append(thread)
    
    for t in threads:
        t.join()

    te = time.time()
    cost = (te-tb)/60
    logging("Done with pitch extraction at the cost of {}min".format(str(cost)))

def pitch_extraction(wav, des):
    # Get the spkr's gender
    flog = des+os.sep+"pitch_extraction.log"
    logging.basicConfig(filename = flog, level = logging.DEBUG)

    logging.info("Start building the dictionary of spker's gender...")
    gender_dic = build_dic(wav)
    logging.info("Done with building the dictionary...")

    logging.info("Start pitch extraction...")
    chunks = split_data(wav)

    work(chunks, des)
    

if __name__ == "__main__":
    if len(sys.argv) != 3 :
        print("Usage: {} data des".format(sys.argv[0]))
        exit(1)

    pitch_extraction(sys.argv[1], sys.argv[2])
