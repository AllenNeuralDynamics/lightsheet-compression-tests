import argparse
import logging
import os
import sys
from timeit import default_timer as timer

import cv2 as cv
import bm3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import skimage
import tifffile
import zarr
from scipy import fftpack
from skimage.exposure import rescale_intensity
from skimage.restoration import (
    estimate_sigma,
    denoise_nl_means,
    denoise_bilateral,
    denoise_tv_chambolle,
    denoise_wavelet
)

import compress_zarr

try:
    import imagej
    import scyjava
    # initialize Fiji with the external non-local means plugin
    IMAGEJ = imagej.init(['sc.fiji:fiji', 'com.github.thorstenwagner:ij-nl-means'], headless=False)
    IMAGEJ.ui().showUI()
except ImportError:
    IMAGEJ = None


def identity(data):
    return data


def run_ij_plugin(arr, plugin, args):
    """Running external ImageJ plugins does not work headlessly,
    for debug use only."""
    # Set the active image in the Window Manager
    IMAGEJ.ui().show(IMAGEJ.py.to_java(arr))
    # Run plugin on the active image
    IMAGEJ.py.run_plugin(plugin, args)
    # Get active image
    imp = IMAGEJ.py.active_image_plus()
    # Convert to numpy ndarray
    result = IMAGEJ.py.from_java(imp)
    # Hack to bypass the "Save image before closing?" prompt
    imp.changes = False
    imp.close()
    return result


# OpenCV Total variation denoising
# This only works with 8-bit images
def cv_tv(data):
    data = skimage.img_as_ubyte(data)
    result = np.zeros_like(data)
    if data.ndim == 2:
        cv.denoise_TVL1([data], result)
    elif data.ndim == 3:
        for i in range(data.shape[0]):
            cv.denoise_TVL1([data[i]], result[i])
    else:
        raise ValueError
    return result


# OpenCV Block-matching and 3D filtering
# Only available in non-free extension
# I was unable to compile it so this remains untested
def cv_bm3d(data):
    result = np.zeros_like(data)
    if data.ndim == 2:
        cv.xphoto.bm3dDenoising(data, result)
    elif data.ndim == 3:
        for i in range(data.shape[0]):
            cv.xphoto.bm3dDenoising(data[i], result[i])
    else:
        raise ValueError
    return result


# OpenCV non-local means denoising
# Much faster than the scikit-image version
def cv_nl_means(data, h):
    norm_type = cv.NORM_L1  # required for uint16
    if data.ndim == 2:
        return cv.fastNlMeansDenoising(data, h=[3], normType=norm_type)
    elif data.ndim == 3:
        slices = []
        for s in data:
            slices.append(cv.fastNlMeansDenoising(s, h=[3], normType=norm_type))
        return np.array(slices)
    else:
        raise ValueError


# ImageJ N-D median filter
RectangleShape = scyjava.jimport("net.imglib2.algorithm.neighborhood.RectangleShape")
def ij_median(data, shape):
    src = IMAGEJ.op().transform().flatIterableView(IMAGEJ.py.to_java(data))
    dst = IMAGEJ.op().run('create.img', src)
    return IMAGEJ.py.from_java(IMAGEJ.op().filter().median(dst, src, shape))


# ImageJ non-local means denoising
def ij_nl_means(data):
    """IJ.py.to_java() returns a Java view of the numpy array.
    It does not allocate new memory, so make a copy to avoid overwriting the input image."""
    data_copy = data.copy()
    plugin = "Non-local Means Denoising"
    args = {
        'sigma': 5,
        'smoothing_factor': 1,
        'auto_estimate_sigma': True
    }
    if data_copy.ndim == 3:
        slices = []
        for s in data_copy:
            result = run_ij_plugin(s, plugin, args)
            slices.append(result)
        return np.array(slices)
    elif data_copy.ndim == 2:
        return run_ij_plugin(data_copy, plugin, args)
    else:
        raise ValueError


# Scikit-image non-local means denoising
# Much slower than OpenCV and ImageJ implementations
def skimage_nl_means(data):
    """This is way too slow"""
    sigma = np.mean(estimate_sigma(data))
    denoised = denoise_nl_means(data, h=0.8 * sigma, sigma=sigma, preserve_range=True)
    # cast back to uint16
    return denoised


def denoise_fft(data):
    """This does not work well, ringing artifacts in the result.
    Taken from https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html"""
    im_fft = fftpack.fftn(data)
    # Define the fraction of coefficients (in each direction) we keep
    keep_fraction = 0.3
    # Set s, r and c to be the number of slices, rows and columns of the array.
    s, r, c = im_fft.shape
    # Set to zero all slices with indices between s*keep_fraction and
    # s*(1-keep_fraction):
    im_fft[int(s * keep_fraction):int(s * (1 - keep_fraction)), :, :] = 0
    # Similarly with the rows and columns:
    im_fft[:, int(r * keep_fraction):int(r * (1 - keep_fraction)), :] = 0
    im_fft[:, :, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    denoised = fftpack.ifftn(im_fft).real
    # not sure if this makes sense
    return denoised


# Python wrapper around Block-Matching and 3D filtering pypi project
# This is remarkably slow with the default parameters
# https://pypi.org/project/bm3d/
def denoise_bm3d(data, sigma_psd):
    """This is way too slow"""
    if data.ndim == 2:
        return bm3d.bm3d(data, sigma_psd)
    elif data.ndim == 3:
        slices = []
        for s in data:
            slices.append(bm3d.bm3d(s, sigma_psd))
        return np.array(slices)
    else:
        raise ValueError


# Scikit-image bilateral filter
def bilateral(data, sigma_spatial, sigma_color):
    if data.ndim == 2:
        return denoise_bilateral(data, sigma_spatial=sigma_spatial, sigma_color=sigma_color)
    elif data.ndim == 3:
        slices = []
        for s in data:
            slices.append(denoise_bilateral(s, sigma_spatial=sigma_spatial, sigma_color=sigma_color))
        return np.array(slices)
    else:
        raise ValueError


# I believe this version is designed for 2D or multichannel images
# def tv_bregman(data):
#     return denoise_tv_bregman(data)


# Scikit-image total variation denoising for N-D images
def tv_chambolle(data, weight):
    # Use a larger weight for more denoising
    return denoise_tv_chambolle(data, weight)


# Scikit-image wavelet denoising
def wavelet(data, sigma):
    if data.ndim == 2:
        return denoise_wavelet(data, sigma=sigma)
    elif data.ndim == 3:
        slices = []
        for s in data:
            slices.append(denoise_wavelet(s, sigma=sigma))
        return np.array(slices)
    else:
        raise ValueError


def rescale(data, dmin, dmax):
    return rescale_intensity(data, out_range=(dmin, dmax)).astype(np.uint16)


class FilterConfig:
    def __init__(self, name, func, args):
        self.name = name
        self.func = func
        self.args = args


def get_filters():
    filters = [
        FilterConfig('no filter', identity, dict()),
        FilterConfig('median-imagej', ij_median, dict(shape=RectangleShape(1, False))),
        FilterConfig('nl-means-cv', cv_nl_means, dict(h=[3])),
        FilterConfig('tv-chambolle', tv_chambolle, dict(weight=0.001)),
        FilterConfig('bilateral', bilateral, dict(sigma_spatial=1, sigma_color=None)),
        FilterConfig('wavelet', wavelet, dict(sigma=None)),
        # FilterConfig('bm3d', denoise_bm3d, dict(sigma_psd=1.0)),
    ]
    return filters


def test_filters(data, filters, outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    tifffile.imwrite(os.path.join(outdir, "noisy-input.tif"), data)
    for f in filters:
        denoised = f.func(data, **f.args)
        tifffile.imwrite(os.path.join(outdir, f"{f.name}.tif"), denoised)


def run(input_file, num_tiles, output_data_file, output_metrics_file):
    filters = get_filters()
    compressors = compress_zarr.build_compressors("blosc", [0], [0])
    all_metrics = []

    total_tests = num_tiles * len(filters) * len(compressors)

    for i in range(num_tiles):
        data, rslice, read_time = compress_zarr.read_random_chunk(input_file, 1)
        dmin = data.min()
        dmax = data.max()
        logging.info(f"loaded data with shape {data.shape}, size {data.nbytes / 2 ** 20} MiB, min {dmin}, max {dmax}")
        for f in filters:
            for c in compressors:
                compressor = c['compressor']

                tile_metrics = {
                    "compressor_name": c['name'],
                    "tile": rslice
                }
                tile_metrics.update(c['params'])

                logging.info(f"starting test {len(all_metrics) + 1}/{total_tests}")

                psutil.cpu_percent(interval=None)
                start = timer()
                # include filter step in compression time
                filtered = rescale(f.func(data, **f.args), dmin, dmax)
                ds = zarr.DirectoryStore(output_data_file)
                za = zarr.array(filtered, compressor=compressor, store=ds, overwrite=True)
                logging.info(str(za.info))
                cpu_utilization = psutil.cpu_percent(interval=None)
                end = timer()
                compress_dur = end - start

                storage_ratio = za.nbytes / za.nbytes_stored
                compress_bps = data.nbytes / compress_dur

                logging.info(f"compression time = {compress_dur}, "
                             f"bps = {compress_bps}, "
                             f"storage ratio = {storage_ratio}, "
                             f"cpu percent = {cpu_utilization}%")

                tile_metrics.update({
                    'name': c['name'],
                    'bytes_read': za.nbytes,
                    'compress_time': compress_dur,
                    'bytes_written': za.nbytes_stored,
                    'shape': data.shape,
                    'cpu_utilization': cpu_utilization,
                    'storage_ratio': storage_ratio,
                    'compress_bps': compress_bps,
                    'filter': f.name
                })

                all_metrics.append(tile_metrics)

    df = pd.DataFrame.from_records(all_metrics)
    df.to_csv(output_metrics_file, index_label='test_number')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str,
                        default=r"C:\Users\cameron.arshadi\Downloads\BrainSlice1_MMStack_Pos33_15_shift.tif")
    parser.add_argument('-n', "--num-tiles", type=int, default=1)
    parser.add_argument('-d', '--output-data-file', type=str, default='./denoise-test.zarr')
    parser.add_argument('-o', '--output-metrics-file', type=str, default='./denoise-metrics.csv')
    parser.add_argument("-l", "--log-level", type=str, default=logging.INFO)

    args = parser.parse_args(sys.argv[1:])
    print(args)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(args.log_level)

    run(args.input_file, args.num_tiles, args.output_data_file, args.output_metrics_file)

    plot_speed(args.output_metrics_file, compressor="blosc-zstd", shuffle=1)
    plt.legend()
    plt.show()


def plot_mips(data, filters, cols=3, outfile=None):
    """plot maximum intensity projections of each filter configuration"""
    total = len(filters)
    rows = total // cols
    rows += total % cols
    fig, ax = plt.subplots(rows, cols)
    for i, f in enumerate(filters):
        denoised = np.max(f.func(data, **f.args), axis=0)
        ax.ravel()[i].imshow(denoised, vmin=np.percentile(denoised, 0.1), vmax=np.percentile(denoised, 99.9))
        ax.ravel()[i].set_title(f.name)
        ax.ravel()[i].axis('off')
    if outfile is not None:
        plt.savefig(outfile, dpi=300)


def plot_speed(metrics_file, compressor, shuffle, outfile=None):
    df = pd.read_csv(metrics_file, index_col='test_number')
    df = df[df['compressor_name'] == compressor]
    df = df[df['shuffle'] == shuffle]

    fig, axes = plt.subplots(2, 1, sharex=True)
    bar_width = 0.2

    for i, f in enumerate(df['filter'].unique().tolist()):
        f_df = df[df['filter'] == f]
        axes[0].bar(f_df['level'].to_numpy() + (i * bar_width), f_df['storage_ratio'], bar_width, label=f)
        axes[1].bar(f_df['level'].to_numpy() + (i * bar_width), f_df['compress_bps'] / (2 ** 20), bar_width, label=f)

    axes[0].set_ylabel("storage ratio")
    axes[0].set_xlabel("compression level")
    axes[0].set_xticks(df['level'].unique().tolist())
    axes[0].grid(axis='y')

    axes[1].set_ylabel("compress speed (MiB/s)")
    axes[1].set_xlabel("compression level")
    axes[1].set_xticks(df['level'].unique().tolist())
    axes[1].grid(axis='y')

    plt.legend()
    plt.suptitle(compressor + f", shuffle {shuffle}")
    if outfile is not None:
        plt.savefig(outfile, dpi=300)


if __name__ == "__main__":
    test_chunk_file = r"C:\Users\cameron.arshadi\Desktop\repos\lightsheet-compression-tests\chunk.tif"
    test_chunk = tifffile.imread(test_chunk_file)
    plot_mips(test_chunk, get_filters())
    test_filters(test_chunk, get_filters(), "./denoised")
    main()
    if IMAGEJ is not None:
        IMAGEJ.getContext().dispose()
        scyjava.shutdown_jvm()
