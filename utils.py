import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import scipy.signal

def gen_dataset():

    path = "data/MLSeminarSchraube/Audioaufnahmen/"
    prefix = "schraubeM"
    suffix = "_objectnr_1.wav"
    sizes = [3, 4, 5, 6, 8, 10, 12]

    filenames = [("{}"*4).format(path, prefix, size, suffix) for size in sizes] 

    data, sr = librosa.load(filenames[1])

    return data, sr

    
def cut(data, sr):
    
    # spectrum und phase berechnen
    spectrum, phase = librosa.magphase(librosa.stft(data))
    # rms berechnen
    rms_raw = librosa.feature.rms(S=spectrum)
    # rms glätten
    rms_hat = scipy.signal.savgol_filter(rms_raw, 51, 3)
    rms = rms_hat[0]

    plt.plot(rms_hat[0])
    plt.show()

data, sr = gen_dataset()
cut(data, sr)
