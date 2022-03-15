import pandas as pd
import numpy as np
import scipy.signal
import scipy.stats
import librosa


# preprocessing functions
def raw(x):
    return x

def welch(x):
    return scipy.signal.welch(x)[1]

def rms(x):
    S, _ = librosa.magphase(librosa.stft(x))
    rms_raw = librosa.feature.rms(S=S)
    rms_smoothed = scipy.signal.savgol_filter(rms_raw, 51, 3)[0]
    return rms_smoothed

def diff(x):
    rms_smoothed = rms(x)
    diff_raw = np.diff(rms_smoothed)
    diff_smoothed = scipy.signal.savgol_filter(diff_raw, 51, 3)
    return diff_smoothed

def mfcc(x):
    return librosa.feature.mfcc(y=x)[0]
def poly_feat(x):
    return librosa.feature.poly_features(y=x)[0]
def spec_cent(x):
    return librosa.feature.spectral_centroid(y=x)[0]
def spec_band(x):
    return librosa.feature.spectral_bandwidth(y=x)[0]
def spec_cont(x):
    return librosa.feature.spectral_contrast(y=x)[0]
def spec_flat(x):
    return librosa.feature.spectral_flatness(y=x)[0]
def spec_roll(x):
    return librosa.feature.spectral_rolloff(y=x)[0]


# postprocessing functions
def q1(x):
    return np.quantile(x, 0.25)

def q3(x):
    return np.quantile(x, 0.75)

def mode(x):
    return scipy.stats.mode(x)[0][0]

def zerocross(x):
    return np.sum(librosa.zero_crossings(x, pad=False))

preprocessing = [welch, rms, diff, mfcc, poly_feat, spec_cent, spec_band, spec_cont, spec_flat, spec_roll]

postprocessing = [np.mean, np.std, np.amax, np.argmax, np.amin, np.argmin, np.median, scipy.stats.skew, scipy.stats.kurtosis, q1, q3, mode, scipy.stats.iqr, zerocross]


def add_features(samples, pre, post):

    for process in pre:
        samples["{}".format(process.__name__)] = samples["raw_sample"].apply(process)
        for func in post:
            samples["{}_{}".format(process.__name__, func.__name__)] = samples["{}".format(process.__name__)].apply(func)
    return samples

samples = pd.read_pickle("samples.pkl")
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
samples = add_features(samples, preprocessing, postprocessing)
features = samples.copy()
features["diameter"] = features["diameter"].astype(int)
print(features)
features = features.select_dtypes(exclude="object")
features.to_pickle("features.pkl")
