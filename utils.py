import re
import os
import numpy as np
from scipy.io.wavfile import read as readwav
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

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

    limits = [1, 4.5] #seconds
    samplef = df.iloc[0]["samplef"]
    df["clip"] = df["raw audio"].apply(lambda x: x[int(limits[0]*samplef):int(limits[1]*samplef)])
    return df


def get_fft(data, samplef):

    T = 1/samplef
    N = data.shape[0]

    ps = fft(data)
    freqs = fftfreq(N, T)[:N//2]

    psnorm = 2.0/N * np.abs(ps[0:N//2])
    return freqs, psnorm
