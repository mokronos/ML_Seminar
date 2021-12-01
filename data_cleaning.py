import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import re

filename = "./data/01_EM_Prüfmessungen_Audiofiles/#1/EM1_RPM3000_D0_Nr(1).wav"
sr, data = wavfile.read(filename)
duration = data.shape[0]/sr

x = [re.split(filename,i) for i in ["EM\d","EM"]]
print(x)





smplr = 44100
dur = 5
def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

x, y = generate_sine_wave(400, smplr, dur)
_, y2 = generate_sine_wave(400, smplr, dur)

max_val = np.amax(y)
norm_y = np.int16((y/max_val) * 32767)

wavfile.write("data/sine.wav", smplr, norm_y)

N = smplr * dur
N2 = data.shape[0]

yf = fft(data)
xf = fftfreq(N2, 1 / sr)

plt.plot(xf, np.abs(yf))



