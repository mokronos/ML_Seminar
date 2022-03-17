import numpy as np
import pandas as pd
import librosa
import scipy.signal

def gen_dataset():

    path = "data/MLSeminarSchraube/Audioaufnahmen/"
    prefix = "schraubeM"
    suffix = "_objectnr_1.wav"

    # only use those 6 diameters for now, "12" is really noisy
    # sizes = [3, 4]
    # sizes = [3, 4, 5, 6, 8, 10]
    sizes = [3, 4, 5, 6, 8, 10, 12]
    # sizes = [10, 12]

    filenames = [("{}"*4).format(path, prefix, size, suffix) for size in sizes] 

    data, sr = [], []
    data_raw = []
    for file in filenames:
        data_temp, sr_temp = librosa.load(file, sr=44100)
        data_raw.append(data_temp)
        rms = get_rms(data_temp)
        data.append(rms)
        sr.append(sr_temp)
        print("{} loaded".format(file))

    long_df = pd.DataFrame({"data_raw": data_raw, "rms": data, "sr": sr, "diameter": sizes})


    return long_df

def get_rms(data):

    # spectrum und phase berechnen
    spectrum, phase = librosa.magphase(librosa.stft(data))
    # rms berechnen
    rms_raw = librosa.feature.rms(S=spectrum)
    # rms glätten
    rms_hat = scipy.signal.savgol_filter(rms_raw, 51, 3)
    rms = rms_hat[0]

    return rms
    
def gen_cuts(start=0, amount=15, length=300, space=180):
    cuts = []
    
    for i in range(amount):
        cuts.append(start + (length + space) * i)
        cuts.append(start + (length + space) * i + length)
    return cuts

def cut(long_df):
    
    #cut beginning and end off
    long_df["rms_cut"] = long_df["rms"].apply(lambda x: x[2740:10434])
    long_df["max"] = long_df["rms_cut"].apply(lambda x: np.percentile(-x, 70))
    long_df["minima_rms"] = long_df.apply(lambda x: scipy.signal.find_peaks(-x["rms_cut"], width = 30, height = x["max"], distance=150)[0], axis=1)
    long_df["cuts"] = [gen_cuts(start=2935, length=280, space=230)]*long_df.shape[0]
    return long_df

def cut_samples(long_df):
    samples_df = pd.DataFrame(columns= ["raw_sample", "diameter"]) 
    for i in range(long_df.shape[0]):
        samples = cut_peaks(long_df["data_raw"][i],long_df["cuts"][i])
        dia = [long_df["diameter"][i]] * len(samples)
        samples_df = samples_df.append(pd.DataFrame({"raw_sample": samples, "diameter": dia}), ignore_index=True)

    return samples_df

def cut_peaks(rms, peaks):
    rms_samples = []
    for i in range(0,len(peaks),2):
        rms_samples.append(rms[peaks[i]*512:peaks[i+1]*512])
    
    return rms_samples
