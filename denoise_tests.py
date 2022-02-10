import logging
from timeit import default_timer as timer

import bm3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import tifffile
import zarr
from scipy import fftpack
from skimage.exposure import rescale_intensity
from skimage.filters.rank import median
from skimage.restoration import (
    estimate_sigma,
    denoise_nl_means,
    denoise_bilateral,
    denoise_tv_bregman,
    denoise_tv_chambolle,
    denoise_wavelet
)

import compress_zarr

try:
    import imagej
    import scyjava

    IJ = imagej.init(['sc.fiji:fiji', 'com.github.thorstenwagner:ij-nl-means'], headless=False)
    IJ.ui().showUI()
except ImportError:
    IJ = None


def identity(data):
    return data


def run_ij_plugin(arr, plugin, args):
    """Running external ImageJ plugins does not work headlessly,
    for debug use only."""
    # Set the active image in the Window Manager
    IJ.ui().show(IJ.py.to_java(arr))
    # Run plugin on the active image
    IJ.py.run_plugin(plugin, args)
    # Get active image
    imp = IJ.py.active_image_plus()
    # Convert to numpy ndarray
    result = IJ.py.from_java(imp)
    # Hack to bypass the "Save image before closing?" prompt
    imp.changes = False
    imp.close()
    return result


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
        return rescale(np.array(slices))
    else:
        return rescale(run_ij_plugin(data_copy, plugin, args))


def nl_means(data):
    """This is way too slow"""
    sigma = np.mean(estimate_sigma(data))
    denoised = denoise_nl_means(data, h=0.8 * sigma, sigma=sigma, preserve_range=True)
    # cast back to uint16
    return np.clip(denoised, 0, 2 ** 16 - 1).astype(np.uint16)


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
    return rescale_intensity(denoised, out_range=(0, 2 ** 16 - 1)).astype(np.uint16)


def denoise_bm3d(data):
    """This is way too slow"""
    psd = 1
    if data.ndim == 2:
        return rescale(bm3d.bm3d(data, psd))
    else:
        slices = []
        for s in data:
            slices.append(bm3d.bm3d(s, psd))
        return rescale(np.array(slices))


def bilateral(data):
    sigma_spatial = 0.8
    if data.ndim == 2:
        return rescale(denoise_bilateral(data, sigma_spatial=sigma_spatial))
    else:
        slices = []
        for s in data:
            slices.append(denoise_bilateral(s, sigma_spatial=sigma_spatial))
        return rescale(np.array(slices))


# def tv_bregman(data):
#     return rescale(denoise_tv_bregman(data))


def tv_chambolle(data):
    # Use a larger weight for more denoising
    weight = 0.001
    return rescale(denoise_tv_chambolle(data, weight))


def wavelet(data):
    if data.ndim == 2:
        return rescale(denoise_wavelet(data))
    else:
        slices = []
        for s in data:
            slices.append(denoise_wavelet(s))
        return rescale(np.array(slices))


def rescale(data):
    return rescale_intensity(data, out_range=(0, 2 ** 16 - 1)).astype(np.uint16)


def get_funcs():
    return {
        'identity': identity,
        'median': median,
        'nl_means': ij_nl_means,
        'bilateral': bilateral,
        # 'bm3d': denoise_bm3d,
        # 'fft': denoise_fft,
        # 'tv_bregman': tv_bregman,
        'tv_chambolle': tv_chambolle,
        'wavelet': wavelet
    }


def test_filter(data, func):
    denoised = func(data)
    tifffile.imwrite("./noisy-input.tif", data)
    tifffile.imwrite("./denoised.tif", denoised)


def plot(metrics_file, compressor, shuffle):
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

    plt.suptitle(compressor + f", shuffle {shuffle}")


def run(input_file, num_tiles, output_data_file, output_metrics_file):
    funcs = get_funcs()
    compressors = compress_zarr.build_compressors("blosc", [0], [0])
    all_metrics = []

    total_tests = num_tiles * len(funcs) * len(compressors)

    for i in range(num_tiles):
        data, rslice, read_time = compress_zarr.read_random_chunk(input_file, 1)
        # data = tifffile.imread("./chunk.tif")
        # rslice = 0
        # read_time = 1
        print(f"loaded data with shape {data.shape}, {data.nbytes / 2 ** 20} MiB")

        for f in funcs:
            for c in compressors:
                compressor = c['compressor']

                tile_metrics = {
                    "compressor_name": c['name'],
                    "tile": rslice
                }
                tile_metrics.update(c['params'])

                print(f"starting test {len(all_metrics) + 1}/{total_tests}")

                psutil.cpu_percent(interval=None)
                start = timer()
                # include filter step in compression time
                filtered = funcs[f](data)
                ds = zarr.DirectoryStore(output_data_file)
                za = zarr.array(filtered, compressor=compressor, store=ds, overwrite=True)
                logging.info(str(za.info))
                cpu_utilization = psutil.cpu_percent(interval=None)
                end = timer()
                compress_dur = end - start

                storage_ratio = za.nbytes / za.nbytes_stored
                compress_bps = data.nbytes / compress_dur

                print(f"compression time = {compress_dur}, "
                      f"bps = {compress_bps}, "
                      f"ratio = {storage_ratio}, "
                      f"cpu = {cpu_utilization}%")

                tile_metrics.update({
                    'name': c['name'],
                    'bytes_read': za.nbytes,
                    'compress_time': compress_dur,
                    'bytes_written': za.nbytes_stored,
                    'shape': data.shape,
                    'cpu_utilization': cpu_utilization,
                    'storage_ratio': storage_ratio,
                    'compress_bps': compress_bps,
                    'filter': f
                })

                all_metrics.append(tile_metrics)

    df = pd.DataFrame.from_records(all_metrics)
    df.to_csv(output_metrics_file, index_label='test_number')


def main():
    input_file = r"C:\Users\cameron.arshadi\Downloads\BrainSlice1_MMStack_Pos33_15_shift.tif"
    output_data_file = "./test_file.zarr"
    output_metrics_file = "./median-test-metrics.csv"
    num_tiles = 1

    run(input_file, num_tiles, output_data_file, output_metrics_file)

    plot(output_metrics_file, compressor="blosc-zstd", shuffle=1)
    plt.legend()
    plt.show()


def plot_filters(data):
    funcs = get_funcs()
    fig, ax = plt.subplots(2, 3)
    for i, name in enumerate(funcs):
        denoised = np.max(funcs[name](data), axis=0)
        ax.ravel()[i].imshow(denoised, vmin=np.percentile(denoised, 0.1), vmax=np.percentile(denoised, 99.9))
        ax.ravel()[i].set_title(name)
    plt.show()


if __name__ == "__main__":
    # test_chunk_file = r"C:\Users\cameron.arshadi\Desktop\repos\lightsheet-compression-tests\chunk.tif"
    # test_chunk = tifffile.imread(test_chunk_file)
    # plot_filters(test_chunk)
    # test_filter(test_chunk, ij_nl_means)
    main()
    if IJ is not None:
        scyjava.shutdown_jvm()
