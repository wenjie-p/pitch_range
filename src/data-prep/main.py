import os
import logging
import sys
import subprocess
import codecs

def build_dic(wav):

    dic = {}

    fp = "../../resource_aishell/speaker.info"
    f = codecs.open(fp, "r", encoding = "utf8")
    content = f.readlines()
    f.close()

    for line in content:
        line = line.strip().split()
        dic[line[0]] = line[1]

    return dic

def main(wav, des):
    # Get the spkr's gender
    flog = des+os.sep+"pitch_extraction.log"
    logging.basicConfig(filename = flog, level = logging.DEBUG)

    logging.info("Start building the dictionary of spker's gender...")
    gender_dic = build_dic(wav)
    logging.info("Done with building the dictionary...")

    praat = ""
    pitch_bot = "50"
    pitch_top = "300"

    logging.info("Start pitch extraction...")
    for usg in os.listdir(wav):
        dr1 = wav+os.sep+ug
        for spk in os.listdir(dr1):
            spkout = des+os.sep+usg+os.sep+spk
            if not os.path.isdir(spkout):
                os.makedirs(spkout)
            dr2 = dr1 + os.sep +spk
            for fwav in os.listdir(dr2):
                fwav = dr2 + os.sep + fwav
                gender = gender_dic[spk]
                if gender == "F":
                    pitch_bot = "75"
                    pitch_top = "500"
                fout = fwav.replace(".wav", ".pitch")
                logging.info("Pitch extraction: {} {}".format(spk, fwav))
                subprocess.call([praat, "--run", "ExtractF0.praat", fwav, fout, spkout, pitch_top, pitch_bot])
    logging.info("Done with pitch extraction successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 3 :
        print("Usage: {} data des".format(sys.argv[0]))
        exit(1)

    main(sys.argv[1], sys.argv[2])
