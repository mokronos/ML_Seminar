import numpy as np
import matplotlib.pyplot as plt
from utils import *

#pd.set_option("display.max_rows", None, "display.max_columns", None)

path = './data'

df = gen_dataset(path)
df = df.sort_values(by=['rpm'])
print(df)
plots = df.shape[0] 

t = range(int(df.iloc[0]['length']))
#fig, axes = plt.subplots(20,10,figsize=(100,100),sharex='row')
#for idx, audio in enumerate(df['raw audio'][:plots]):
#    axes[idx%20, idx//20].scatter(t,audio)
#    axes[idx%20, idx//20].set_ylim([-10000,10000])
#    axes[idx%20, idx//20].title.set_text(df['filename'][idx])
#    print(idx)

print(df['rpm'].value_counts())

plt.scatter(t, df['raw audio'][0])
plt.show()
