# from utils import plot_rms, plot_raw
import pandas as pd
import matplotlib.pyplot as plt

def plot_raw(long_df, column="data_raw"):

    count = long_df.shape[0]
    width = 20
    fig, axs = plt.subplots(count, figsize=(width, width), tight_layout = True)
    for i in range(count):
        axs[i].plot(long_df[column][i])
        axs[i].set_title("M{}".format(long_df["diameter"][i]))
        axs[i].set_xlabel("time in frames")
        axs[i].set_ylabel("data_raw")
    plt.savefig('figures/raw_data.png')


def plot_rms(long_df, column="rms", peaks=True):

    count = long_df.shape[0]
    width = 20
    fig, axs = plt.subplots(count, figsize=(width, width), tight_layout = True)
    for i in range(count):
        axs[i].plot(long_df[column][i])
        if peaks:
            axs[i].plot(long_df["cuts"][i], long_df[column][i][long_df["cuts"][i]], "x", color="red")
            # axs[i].plot(long_df["minima_rms"][i], long_df[column][i][long_df["minima_rms"][i]], "x", color="red")
            # axs[i].hlines(-long_df["max"][i], xmin=0, xmax=len(long_df[column][i]), color="green")
        axs[i].set_title("M{}".format(long_df["diameter"][i]))
        axs[i].set_xlabel("time in frames")
        axs[i].set_ylabel("smoothed rms")
    plt.savefig('figures/rms_cuts.png')

def plot_samples(samples, sizes):

    width = 20
    fig, axs = plt.subplots(3,3, figsize=(width, width), tight_layout = True)
    fig.delaxes(axs[2,1])
    fig.delaxes(axs[2,2])
    for i, size in enumerate(sizes):

        temp = samples[samples["diameter"] == size]
        temp = temp.reset_index(drop=True)

        for j in range(temp.shape[0]):

            axs[i//3, i%3].plot(temp["rms"][j])

        axs[i//3, i%3].set_title("M{}".format(size))
        axs[i//3, i%3].set_xlabel("time in frames")
        axs[i//3, i%3].set_ylabel("smoothed rms")


    plt.savefig('figures/rms_samples.png')


samples = pd.read_pickle("samples.pkl")
long_df = pd.read_pickle("long_df.pkl")
pd.set_option('display.max_columns', None)

# long_df = long_df[long_df["diameter"]<7]
long_df = long_df.reset_index(drop=True)
# plot_rms(long_df, peaks=True)

sizes = [3, 4, 5, 6, 8, 10, 12]
plot_samples(samples, sizes)
