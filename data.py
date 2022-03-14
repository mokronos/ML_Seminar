from utils import gen_dataset, cut, cut_samples, plot_rms, plot_raw
import pandas as pd



long_df = gen_dataset()
# plot_rms(long_df, peaks=False)
long_df = cut(long_df)
samples_df = cut_samples(long_df)
pd.set_option('display.max_columns', None)
print(samples_df)
samples_df.to_pickle("samples.pkl")
