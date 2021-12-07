import re
import os
import numpy as np
from scipy.io.wavfile import read as readwav
import pandas as pd

def extract_meta(filename):

    filename = filename.split("/")[-1]
    pat = r"EM(\d+)_RPM(\d+)_D(\d+)_Nr\((\d+)\)\.wav"
    lab = ["cycles","rpm","dist","special","filename"]
    res = re.findall(pat, filename)
    meta = list(map(int,res[0]))
    meta.append(filename)
    meta = dict(zip(lab,meta))
    
    return meta

def gen_dataset(path):

    df = pd.DataFrame()

    for root,_, filenames in os.walk(path):
        for file in filenames:
            if file.split(".")[-1] == "wav":
                filepath = f"{root}/{file}"
                samplef, data = readwav(filepath)
                meta = extract_meta(filepath)
                meta["samplef"] = samplef
                meta['length'] = int(len(data))
                df = df.append(dict({'raw audio' : data}, **meta), ignore_index = True)

    return df


def get_fft(data, length, samplef):
    return

    
