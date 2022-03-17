from utils import gen_dataset, cut, cut_samples, get_rms
import pandas as pd



long_df = gen_dataset()
long_df = cut(long_df)
long_df.to_pickle("long_df.pkl")

samples_df = cut_samples(long_df)
samples_df["rms"] = samples_df["raw_sample"].apply(lambda x: get_rms(x))
samples_df.to_pickle("samples.pkl")

pd.set_option('display.max_columns', None)
