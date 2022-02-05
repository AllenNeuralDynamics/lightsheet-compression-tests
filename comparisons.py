import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.plotting
from sklearn.preprocessing import MinMaxScaler


def klb_vs_zarr():

    klb_metrics = "/allen/scratch/aindtemp/cameron.arshadi/klb-compression-metrics_data.h5.csv"
    zarr_metrics = "/allen/scratch/aindtemp/cameron.arshadi/compression-metrics_data.h5.csv"

    klb_df = pd.read_csv(klb_metrics, index_col='test_number')
    # Blosc uses up to 8 threads by default
    # Extract relevant observations from klb csv
    klb_df = klb_df[klb_df['threads'] == 8]
    klb_df['format'] = 'klb'

    zarr_df = pd.read_csv(zarr_metrics, index_col='test_number')
    level = 1
    shuffle = 0
    zarr_df = zarr_df[zarr_df['level'] == level]
    zarr_df = zarr_df[zarr_df['shuffle'] == shuffle]
    zarr_df['format'] = 'zarr'

    merged_df = pd.concat([klb_df, zarr_df], ignore_index=True).dropna(axis=1)
    merged_df.reset_index(inplace=True)
    merged_df.to_csv("./test.csv")

    # parallel coordinates
    # standardize features
    cols = ['cpu_utilization', 'compress_time', 'storage_ratio']
    scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(merged_df[cols]), columns=cols)
    scaled_df['compressor_name'] = merged_df['compressor_name']
    plt.figure(1, figsize=(13, 9), dpi=300)
    pd.plotting.parallel_coordinates(
        scaled_df,
        'compressor_name',
        cols=cols,
        colormap='jet'
    )
    plt.title(f"blosc (lvl={level}, shuffle={shuffle}) vs klb")
    plt.savefig('./paracoords.png')

    # pair plot
    plt.figure(2, figsize=(18, 18), dpi=300)
    labels = merged_df['compressor_name'].unique().tolist()
    color_palette = [plt.cm.get_cmap('jet')(i / len(labels)) for i in range(1, len(labels) + 1)]
    color_map = dict(zip(labels, color_palette))
    color_idx = merged_df['compressor_name'].map(lambda x: color_map.get(x))
    pandas.plotting.scatter_matrix(merged_df, color=color_idx, figsize=(18, 18), alpha=0.7)
    # hack a legend
    handles = [plt.plot([], [], color=c, ls="", marker=".", markersize=np.sqrt(10))[0] for c in color_palette]
    plt.legend(handles, labels, loc=(1.02, 0))
    plt.suptitle(f"blosc (lvl={level}, shuffle={shuffle}) vs klb")
    plt.tight_layout()
    plt.savefig("./pairplot.png")

    plt.show()


if __name__ == "__main__":
    klb_vs_zarr()
