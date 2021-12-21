import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn import svm

#pd.set_option("display.max_rows", None, "display.max_columns", None)

path = './data'
df = gen_dataset(path)
df = df.sort_values(by=['rpm'])
plots = df.shape[0] 

t = range(int(df.iloc[0]['length']))
#fig, axes = plt.subplots(20,10,figsize=(100,100),sharex='row')
#for idx, audio in enumerate(df['raw audio'][:plots]):
#    axes[idx%20, idx//20].scatter(t,audio)
#    axes[idx%20, idx//20].set_ylim([-10000,10000])
#    axes[idx%20, idx//20].title.set_text(df['filename'][idx])
#    print(idx)

train_x = df["clip"].tolist()
train_y = df["rpm"].values
samplef = df["samplef"][0]

#x, y = get_fft(train_x[0],samplef)
fig, axes = plt.subplots(20,10, figsize=(100,100))
for idx, sample in enumerate(train_x):
    x, y = get_fft(sample, samplef)
    axes[idx%20, idx//20].plot(x,y)
    #axes[idx%20, idx//20].set_ylim([0,samplef//2])
    axes[idx%20, idx//20].title.set_text(df["filename"][idx])
    print(idx)

plt.savefig("test.png")
