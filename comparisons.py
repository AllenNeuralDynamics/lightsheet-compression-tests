import itertools

import matplotlib.pyplot as plt
import pandas as pd


def barplot(metrics_file, ax, x_key, y_key, secondary_y_key=None, bar_width=0.2, **kwargs):
    zarr_df = pd.read_csv(metrics_file, index_col='test_number')
    if secondary_y_key is not None:
        ax2 = ax.twinx()
    for i, compressor in enumerate(zarr_df['compressor_name'].unique().tolist()):
        df = zarr_df[zarr_df['compressor_name'] == compressor]
        # txtbox = ""
        if 'shuffle' in kwargs:
            df = df[df['shuffle'] == kwargs['shuffle']]
            # txtbox += f"shuffle={kwargs['shuffle']}\n"
        if 'level' in kwargs:
            df = df[df['level'] == kwargs['level']]
            # txtbox += f"level={kwargs['level']}\n"
        if 'trunc' in kwargs:
            df = df[df['trunc'] == kwargs['trunc']]
            # txtbox += f"trunc={kwargs['trunc']}\n"
        if 'chunk_scale' in kwargs:
            df = df[df['chunk_factor'] == kwargs['chunk_scale']]
            # txtbox += f"chunk scale={kwargs['chunk_scale']}\n"
        mean_df = df.groupby([x_key]).mean().sort_index()
        x = mean_df.index.to_numpy()
        y = mean_df[y_key].to_numpy()
        yerr = df.groupby([x_key]).std().sort_index()[y_key].to_numpy()
        ax.bar(x + (i * bar_width), y, bar_width, yerr=yerr, label=compressor)
        ax.set_xticks(x)
        if secondary_y_key is not None:
            t = mean_df[secondary_y_key].to_numpy()
            ax2.plot(x, t)
            ax2.set_ylabel(secondary_y_key)
    # ax.text(0, 0, txtbox, color='black', rotation=0, bbox=dict(facecolor=None, boxstyle="square,pad=0.3"))
    ax.set_ylabel(y_key)
    ax.set_xlabel(x_key)


if __name__ == "__main__":
    x_keys = ['level', 'shuffle', 'trunc']
    y_keys = ['storage_ratio', 'compress_bps', 'cpu_utilization']
    default_values = {
        'shuffle': 1,
        'trunc': 0,
        'level': 9
    }
    keys = list(itertools.product(x_keys, y_keys))
    n = len(keys)
    # Make a square-ish grid
    import math
    rows = int(math.sqrt(n))
    cols = int(n / rows) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(9,13))
    for i, xy_keys in enumerate(keys):
        defaults_no_x = default_values.copy()
        defaults_no_x.pop(xy_keys[0])  # remove varying parameter from defaults
        barplot(
            metrics_file=r"Y:\aindtemp\cameron.arshadi\compression-metrics_data.h5.csv",
            ax=axes.ravel()[i],
            x_key=xy_keys[0],
            y_key=xy_keys[1],
            secondary_y_key=None,
            bar_width=0.2,
            **defaults_no_x
        )
    axes.ravel()[0].legend()
    plt.tight_layout()
    plt.show()
