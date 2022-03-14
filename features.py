import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean


samples = pd.read_pickle("samples.pkl")

pd.set_option('display.max_columns', None)

# single = samples[samples["diameter"] == 8]
single = samples

single["mean"] = single["rms_sample"].apply(mean)
pd.set_option('display.max_rows', None)
print(single)

for index, row in single.iterrows():
    plt.plot(row["rms_sample"])
plt.show()
