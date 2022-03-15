import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import scipy.signal

def gen_dataset():

    path = "data/MLSeminarSchraube/Audioaufnahmen/"
    prefix = "schraubeM"
    suffix = "_objectnr_1.wav"

    # only use those 6 diameters for now, "12" is really noisy
    # sizes = [3, 4]
    sizes = [3, 4, 5, 6, 8, 10]
    # sizes = [3, 4, 5, 6, 8, 10, 12]
    # sizes = [10, 12]

    filenames = [("{}"*4).format(path, prefix, size, suffix) for size in sizes] 

    data, sr = [], []
    data_raw = []
    for file in filenames:
        data_temp, sr_temp = librosa.load(file, sr=44100)
        data_raw.append(data_temp)
        rms = get_rms(data_temp, sr_temp)
        data.append(rms)
        sr.append(sr_temp)
        print("{} loaded".format(file))

    long_df = pd.DataFrame({"data_raw": data_raw, "rms": data, "sr": sr, "diameter": sizes})


    return long_df

def get_rms(data, sr):

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
    # long_df["rms"] = long_df["rms"].apply(lambda x: x[2740:10434])
    # long_df["raw_data_cut"] = long_df["raw_data"].apply(lambda x: x[1370:5217])
    long_df["max"] = long_df["rms"].apply(lambda x: np.percentile(-x, 83))
    long_df["minima_rms"] = long_df.apply(lambda x: scipy.signal.find_peaks(-x["rms"], width = 30, height = x["max"], distance=150)[0], axis=1)
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

def add_feat(samples_df):
    
    return samples_df


def plot_raw(long_df, column="data_raw"):

    count = long_df.shape[0]
    fig, axs = plt.subplots(count, figsize=(8, 4*count), tight_layout = True)
    for i in range(count):
        axs[i].plot(long_df[column][i])
        axs[i].set_title("M{}".format(long_df["diameter"][i]))
        axs[i].set_xlabel("time")
        axs[i].set_ylabel("data_raw")
    plt.show()


def plot_rms(long_df, column="rms", peaks=True):

    count = long_df.shape[0]
    fig, axs = plt.subplots(count, figsize=(8, 4*count), tight_layout = True)
    for i in range(count):
        axs[i].plot(long_df[column][i])
        if peaks:
            # axs[i].plot(long_df["minima_rms"][i], long_df[column][i][long_df["minima_rms"][i]], "x", color="red")
            axs[i].plot(long_df["cuts"][i], long_df[column][i][long_df["cuts"][i]], "x", color="red")
            # axs[i].hlines(-long_df["max"][i], xmin=0, xmax=len(long_df[column][i]), color="green")
        axs[i].set_title("M{}".format(long_df["diameter"][i]))
        axs[i].set_xlabel("time")
        axs[i].set_ylabel("smoothed rms")
    plt.show()
