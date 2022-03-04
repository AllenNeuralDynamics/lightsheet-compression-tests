import json
import math
import numbers
import os

import numpy as np
import pandas as pd
import tifffile
from matplotlib import pyplot as plt
from skimage import color
from skimage.morphology import skeletonize


def read_params(param_file):
    with open(param_file, 'r') as f:
        return pd.DataFrame.from_dict(json.load(f), orient='index')


def query_df(df, queries):
    if queries is not None:
        for q in queries:
            df = df.query(q)
    return df


def plot_parameters(imdir, param_file, reference_imfile, df_queries=None, annotate_keys=None, sort_key=None,
                    max_project=True, outfile='./param_examples.png'):
    """imdir may be a directory of compressed images or segmentations, all that's required
    is a parameter json file mapping an image filename (basename) to its compression or segmentation parameters"""
    # Filter images by given queries
    param_df = query_df(read_params(param_file), df_queries)
    if sort_key is not None:
        param_df = param_df.sort_values(sort_key)

    input_data = tifffile.imread(reference_imfile)

    # Make a square-ish grid
    n = param_df.shape[0] + 1  # include the ground-truth image
    rows = int(math.sqrt(n))
    cols = int(n / rows) + 1
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(18, 18))

    # MIP of ground-truth segmentation
    if max_project:
        input_data = np.max(input_data, axis=0)
    axes.ravel()[0].imshow(color.label2rgb(input_data), aspect='auto')
    axes.ravel()[0].set_title(f"input data")
    # Now plot the test segmentations
    for i in range(param_df.shape[0]):
        test_data = tifffile.imread(os.path.join(imdir, param_df.index[i]))
        # MIP of test segmentation
        if max_project:
            test_data = np.max(test_data, axis=0)
        axes.ravel()[i + 1].imshow(color.label2rgb(test_data), aspect='auto')
        # Annotate compression parameters and accuracy metrics
        filtered_keys = [k for k in annotate_keys if k in param_df]
        df_keys = param_df[filtered_keys]
        # Try to evenly space parameter annotations on each image
        label_step = 1.0 / len(df_keys.columns)
        fontsize = 10
        for j, column in enumerate(df_keys):
            val = df_keys[column].iloc[i]
            val = round(val, 4) if isinstance(val, numbers.Number) else val
            axes.ravel()[i + 1].annotate(
                f"{column}={val}",
                xy=(0, j * label_step),
                color='orange',
                xycoords='axes fraction',
                fontsize=fontsize
            )
    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.show()


def plot_skeletons(imdir, param_file, reference_imfile, df_queries=None, annotate_keys=None, sort_key=None,
                   max_project=True, outfile='./skel_examples.png'):
    """imdir should be a directory of binary segmentations. They will be skeletonized
    during plotting."""
    # Filter segmentations by given keys
    param_df = query_df(read_params(param_file), df_queries)
    if sort_key:
        param_df = param_df.sort_values(sort_key)

    true_seg = tifffile.imread(reference_imfile)

    rows = param_df.shape[0] + 1  # include ground-truth
    cols = 2
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(18,18))

    # MIP of ground-truth segmentation
    skel = skeletonize(true_seg)
    if max_project:
        true_seg = np.max(true_seg, axis=0)
        skel = np.max(skel, axis=0)
    axes[0][0].imshow(color.label2rgb(true_seg), aspect="auto")
    axes[0][0].set_title(f"ground truth")
    axes[0][1].imshow(color.label2rgb(skel), aspect="auto")
    for i in range(param_df.shape[0]):
        test_seg = tifffile.imread(os.path.join(imdir, param_df.index[i]))
        skel = skeletonize(test_seg)
        if max_project:
            test_seg = np.max(test_seg, axis=0)
            skel = np.max(skel, axis=0)
        axes[i + 1][0].imshow(color.label2rgb(test_seg), aspect="auto")
        axes[i + 1][1].imshow(color.label2rgb(skel), aspect="auto")
        # Annotate compression parameters and accuracy metrics
        filtered_keys = [k for k in annotate_keys if k in param_df]
        key_df = param_df[filtered_keys]
        # Try to evenly space parameter annotations on each image
        label_step = 1.0 / len(key_df.columns)
        fontsize = 10
        for j, column in enumerate(key_df):
            val = param_df[column].iloc[i]
            val = round(val, 4) if isinstance(val, numbers.Number) else val
            axes[i + 1][0].annotate(
                f"{column}={val}",
                xy=(0, j * label_step),
                color='orange',
                xycoords='axes fraction',
                fontsize=fontsize
            )
    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.show()


def main():
    image_dir = "./segmented"
    param_file = os.path.join(image_dir, "params.json")
    ground_truth_file = os.path.join(image_dir, 'true_seg.tif')
    # Queries to filter the params.json dataframe
    queries = ['compressor_name == "none"']
    with open(param_file, 'r') as f:
        params = json.load(f)
    print(next(iter(params.values())).keys())
    # Keys to paint on each image.
    # These depend on the compressor family chosen. Blosc does not have a 'rate' parameter, for example.
    # The relevant keys will get filtered out based on those in the parameter json file.
    annotate_keys = ['mse', 'ssim', 'psnr', 'rate', 'precision', 'tolerance', 'compressor_name', 'storage_ratio',
                     'adapted_rand_error', 'prec', 'rec', 'sigma', 'ridge_filter', 'level', 'trunc', 'shuffle']
    # plot_parameters(image_dir, param_file, ground_truth_file, queries, annotate_keys, max_project=True,
    #                 sort_key=None)
    plot_skeletons(image_dir, param_file, ground_truth_file, queries, annotate_keys, max_project=True,
                    sort_key=None)


if __name__ == "__main__":
    main()
