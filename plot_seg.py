import json
import math
import numbers
import os

import numpy as np
import pandas as pd
import tifffile
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize


def read_params(param_file):
    with open(param_file, 'r') as f:
        return pd.DataFrame.from_dict(json.load(f), orient='index')


def filter_df(df, args):
    for key, val in args.items():
        df = df[df[key] == val]
    return df


def plot_segmentations(segdir, param_file, ground_truth, outfile='./seg_examples.png', sort_error=True, **kwargs):
    # Filter segmentations by given keys
    df = filter_df(read_params(param_file), kwargs)
    if sort_error:
        df = df.sort_values('adapted_rand_error')

    true_seg = tifffile.imread(ground_truth)

    # Make a square-ish grid
    n = df.shape[0] + 1  # include the ground-truth image
    rows = int(math.sqrt(n))
    cols = int(n / rows) + 1

    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)

    # Try to evenly space parameter annotations on each image
    label_step = 1.0 / len(df.columns)
    fontsize = 10

    # MIP of ground-truth segmentation
    axes.ravel()[0].imshow(np.max(true_seg, axis=0), cmap='gray', aspect='auto')
    axes.ravel()[0].set_title(f"ground truth")
    # Now plot the test segmentations
    for i in range(df.shape[0]):
        test_seg = tifffile.imread(os.path.join(segdir, df.index[i]))
        # MIP of test segmentation
        axes.ravel()[i + 1].imshow(np.max(test_seg, axis=0), cmap='gray', aspect='auto')
        # Annotate compression parameters and accuracy metrics
        for j, column in enumerate(df):
            val = df[column].iloc[i]
            val = round(val, 4) if isinstance(val, numbers.Number) else val
            axes.ravel()[i + 1].annotate(
                f"{column}={val}",
                xy=(0, j * label_step),
                color='red',
                xycoords='axes fraction',
                fontsize=fontsize
            )
    for ax in axes.ravel():
        ax.axis("off")
    # plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.show()


def plot_skeletons(segdir, param_file, ground_truth, outfile='./skel_examples.png', sort_error=True, **kwargs):
    # Filter segmentations by given keys
    df = filter_df(read_params(param_file), kwargs)
    if sort_error:
        df = df.sort_values('adapted_rand_error')

    true_seg = tifffile.imread(ground_truth)

    rows = df.shape[0] + 1  # include ground-truth
    cols = 2
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    # MIP of ground-truth segmentation
    axes[0][0].imshow(np.max(true_seg, axis=0), cmap='gray')
    axes[0][0].set_title(f"ground truth")
    axes[0][1].imshow(np.max(skeletonize(true_seg), axis=0), cmap='gray')
    for i in range(df.shape[0]):
        test_seg = tifffile.imread(os.path.join(segdir, df.index[i]))
        axes[i + 1][0].imshow(np.max(test_seg, axis=0), cmap='gray')
        axes[i + 1][1].imshow(np.max(skeletonize(test_seg), axis=0), cmap='gray')

    plt.savefig(outfile, dpi=600)
    plt.show()


def main():
    output_image_dir = "./images"
    seg_params_file = os.path.join(output_image_dir, "seg_params.json")
    ground_truth_file = './images/true_seg.tif'
    plot_segmentations(output_image_dir,
                       seg_params_file,
                       ground_truth_file,
                       # plot kwargs
                       compressor_name='blosc-zstd',
                       shuffle=0,
                       level=1,
                       )


if __name__ == "__main__":
    main()
