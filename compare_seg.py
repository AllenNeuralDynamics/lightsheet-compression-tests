import argparse
import itertools
import json
import logging
import multiprocessing
import os
import shutil
import sys
from fractions import Fraction
import random
from itertools import repeat
from timeit import default_timer as timer

import h5py
import imagej
import numpy as np
import pandas as pd
import scyjava
import tifffile
import zarr
import scipy.ndimage as ndi
from skimage.filters.thresholding import threshold_mean, threshold_otsu
from skimage.measure import label
from skimage.metrics import adapted_rand_error, variation_of_information

import compress_zarr


class ImageJWrapper:
    def __init__(self, maven_endpoints, max_memory=None, headless=True):
        if max_memory is not None:
            scyjava.config.add_option(f"-Xmx{max_memory}m")
        self.ij = imagej.init(maven_endpoints, headless)
        self.sntutils_cls = scyjava.jimport("sc.fiji.snt.SNTUtils")
        self.frangi_cls = scyjava.jimport("sc.fiji.snt.filter.Frangi")
        self.tubeness_cls = scyjava.jimport("sc.fiji.snt.filter.Tubeness")
        self.floattype_cls = scyjava.jimport("net.imglib2.type.numeric.real.FloatType")
        self.runtime_cls = scyjava.jimport("java.lang.Runtime")


def encode_decode(im, filters, compressor):
    za = zarr.array(im, chunks=True, filters=filters, compressor=compressor)
    return za[:]


def filter_ij(im, op, ij_wrapper):
    # Convert numpy array to Java RandomAccessibleInterval
    # This automatically swaps axes from ZYX -> XYZ
    java_in = ij_wrapper.ij.py.to_java(im)
    java_out = ij_wrapper.ij.op().run("create.img", java_in, ij_wrapper.floattype_cls())
    op.compute(java_in, java_out)
    # Back to numpy and ZYX order
    return ij_wrapper.ij.py.from_java(java_out)


def compare_seg(true_seg, test_seg, metrics):
    m = dict()
    if "are" in metrics:
        m['adapted_rand_error'], m['prec'], m['rec'] = adapted_rand_error(true_seg, test_seg)
    if "voi" in metrics:
        m['splits'], m['merges'] = variation_of_information(true_seg, test_seg)
    return m


def threshold(im, func):
    return im > func(im)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-tiles", type=int, default=1)
    parser.add_argument("-i", "--input-file", type=str, default=r"C:\Users\cameron.arshadi\Downloads\BrainSlice1_MMStack_Pos33_15_shift.tif")
    parser.add_argument("-r", "--resolution", type=str, default="2")
    # Order should be XYZ
    parser.add_argument("-v", "--voxel-size", nargs="+", type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("-s", "--random-seed", type=int, default=42)
    parser.add_argument("-f", "--filter", type=str, default='sato')  # 'frangi' or 'sato'
    # Scales for ridge filters, these are multiplied by the Nyquist rate
    # i.e., half the average voxel spacing of the input image
    parser.add_argument("-g", "--scales", nargs="+", type=float, default=[2, 3, 4])
    # Max memory for JVM in megabytes, i.e., -Xmx{max_memory}m
    parser.add_argument("-x", "--max-memory", type=int, default=12000)
    parser.add_argument("-d", "--output-image-dir", type=str, default="./segmented")  # Optional, will save a lot of images
    parser.add_argument("-o", "--output-metrics-file", type=str, default="./segmentation_metrics.csv")
    parser.add_argument("-l", "--log-level", type=str, default=logging.INFO)
    parser.add_argument("-c", "--codecs", nargs="+", type=str, default=["none"])
    parser.add_argument("-t", "--trunc-bits", nargs="+", type=int, default=range(13))
    parser.add_argument("-m", "--metrics", nargs="+", type=str, default=['are', 'voi'])  # ['are', 'voi']

    args = parser.parse_args(sys.argv[1:])
    print(args)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(args.log_level)

    if args.output_image_dir is not None:
        if os.path.isdir(args.output_image_dir):
            # Wipe output directory on each run
            shutil.rmtree(args.output_image_dir)
        os.mkdir(args.output_image_dir)

    chunk_factor = [4]
    compressors = compress_zarr.build_compressors(args.codecs, args.trunc_bits, chunk_factor)

    maven_endpoints = ['sc.fiji:fiji', 'org.morphonets:SNT:4.0.8']
    ij_wrapper = ImageJWrapper(maven_endpoints, max_memory=args.max_memory, headless=True)
    # This really should be 4.0.8, but I get 4.0.3??
    logging.info("We are running SNT version " + str(ij_wrapper.sntutils_cls.VERSION))

    if args.random_seed is not None:
        random.seed(args.random_seed)

    run(
        args.num_tiles,
        args.resolution,
        args.input_file,
        args.voxel_size,
        ij_wrapper,
        args.filter,
        args.scales,
        compressors,
        args.output_image_dir,
        args.output_metrics_file,
        args.metrics
    )


def parse_tiff_metadata(f):
    xres = Fraction(*f.pages[0].tags['XResolution'].value)
    yres = Fraction(*f.pages[0].tags['YResolution'].value)
    spacing = float(f.imagej_metadata['spacing'])
    voxel_spacing = [1. / float(xres), 1. / float(yres), spacing]
    bytes_per_sample = int(f.pages[0].tags['BitsPerSample'].value) / 8
    return voxel_spacing, bytes_per_sample


def get_downsample_factors(input_file, tile, resolution):
    if input_file.endswith(".zarr"):
        z = zarr.open(input_file, 'r')
        return z[f"{tile}/resolutions"][resolution]
    elif input_file.endswith(".h5"):
        with h5py.File(input_file) as f:
            return f[f"{tile}/resolutions"][resolution]
    else:
        raise ValueError("Unsupported file format " + os.path.splitext(input_file)[1])


def lazy_blocks(arr_shape, block_shape, discard_singletons=False):
    """Partition an array with shape arr_shape into blocks with shape block_shape.
    If arr_shape is not wholly divisible by block_shape,
    some blocks will have the remainder as a dimension length."""
    assert all(block_shape[i] <= arr_shape[i] for i in range(len(arr_shape)))
    blocks = []
    for z in range(0, arr_shape[0], block_shape[0]):
        zint = [z, min(z + block_shape[0], arr_shape[0])]
        for y in range(0, arr_shape[1], block_shape[1]):
            yint = [y, min(y + block_shape[1], arr_shape[1])]
            for x in range(0, arr_shape[2], block_shape[2]):
                xint = [x, min(x + block_shape[2], arr_shape[2])]
                blocks.append(np.vstack([zint, yint, xint]))
    if discard_singletons:
        # make sure all chunks are truly 3D
        blocks = [c for c in blocks if np.all(c[:, 1] - c[:, 0] > 1)]
    return blocks


def split_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_best_chunks(chunks, tiff_path, nstd=1):
    """Find the chunks with summed intensity at least nstd standard
    deviations above the mean sum."""
    num_workers = multiprocessing.cpu_count()
    arrs = list(split_list(chunks, num_workers))
    args = list(zip(arrs, repeat(tiff_path), repeat(nstd)))
    start = timer()
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(itertools.chain(*pool.starmap(best_chunks_worker, args)))
    end = timer()
    logging.info(f"finding chunks took {end-start}s")
    return results


def best_chunks_worker(chunks, tiff_path, nstd):
    with tifffile.TiffFile(tiff_path) as f:
        z = zarr.open(f.aszarr(), 'r')
        sums = []
        for i, c in enumerate(chunks):
            # print(f"{i}/{len(chunks)}")
            data = z[c[0][0]:c[0][1], c[1][0]:c[1][1], c[2][0]:c[2][1]]
            sums.append(data.sum())
        avgs = np.mean(sums)
        std = np.std(sums)
        best_chunks = []
        for i, s in enumerate(sums):
            if s >= avgs + nstd * std:
                best_chunks.append(chunks[i])
    return best_chunks


def get_tiff_chunks(input_file, chunk_shape, discard_singletons=False):
    # we only need the array shape, not the data
    return lazy_blocks(get_tiff_shape(input_file), chunk_shape, discard_singletons)


def get_tiff_shape(input_file):
    # get the shape without loading any image data
    with tifffile.TiffFile(input_file) as f:
        za = zarr.open(f.aszarr(), 'r')
        return za.shape


def erode_border(a):
    # Set all 6 faces of a 3D array to zero
    a[0, :, :] = 0
    a[a.shape[0] - 1, :, :] = 0
    a[:, :, 0] = 0
    a[:, :, a.shape[2] - 1] = 0
    a[:, 0, :] = 0
    a[:, a.shape[1] - 1, :] = 0
    return a


def run(num_tiles, resolution, input_file, voxel_size, ij_wrapper, ridge_filter, scales, compressors, output_image_dir,
        output_metrics_file, metrics):

    total_tests = len(compressors) * num_tiles

    all_metrics = []

    # Dictionary mapping segmentation result image to its parameters
    seg_params = {}

    if input_file.endswith('.tif'):
        chunks = get_tiff_chunks(input_file, (64, 512, 512), discard_singletons=True)
        logging.info(f"# chunks = {len(chunks)}")
        chunks = get_best_chunks(chunks, input_file)
        logging.info(f"# chunks after filtering = {len(chunks)}")
        assert len(chunks) >= num_tiles

    for ti in range(num_tiles):
        if input_file.endswith('.h5') or input_file.endswith(".zarr"):
            data, rslice, read_time = compress_zarr.read_random_chunk(input_file, resolution)
            res_voxel_size = np.array(voxel_size) * get_downsample_factors(input_file, rslice, int(resolution))
        elif input_file.endswith('.tif'):
            c = random.choice(chunks)
            with tifffile.TiffFile(input_file) as f:
                za = zarr.open(f.aszarr(), 'r')
                # override user voxel size from tiff metadata
                voxel_size, _ = parse_tiff_metadata(f)
                data = za[c[0][0]:c[0][1], c[1][0]:c[1][1], c[2][0]:c[2][1]]
            logging.info(f"loaded data with shape {data.shape}")
            res_voxel_size = np.array(voxel_size)

        logging.info(f"voxel spacing={res_voxel_size}")

        if output_image_dir is not None:
            tifffile.imwrite(os.path.join(output_image_dir, 'input_data.tif'), data)

        # Nyquist
        scale_step = sum(res_voxel_size) / (2 * len(res_voxel_size))
        print(f"scale step: {scale_step}")
        # N scales takes N times as long
        sigmas = scale_step * np.array(scales)
        print("sigmas: " + str(sigmas))

        num_threads = ij_wrapper.runtime_cls.getRuntime().availableProcessors()

        if ridge_filter == 'frangi':
            op = ij_wrapper.frangi_cls(sigmas, res_voxel_size, np.max(data), num_threads)
        elif ridge_filter == 'sato':
            op = ij_wrapper.tubeness_cls(sigmas, res_voxel_size, num_threads)
        else:
            raise ValueError("Unknown filter: " + ridge_filter)

        response = filter_ij(data, op, ij_wrapper)
        binary = threshold(response, threshold_otsu)
        true_seg, _ = ndi.label(erode_border(binary), structure=np.ones(shape=(3,3,3), dtype=bool))

        if output_image_dir is not None:
            tifffile.imwrite(os.path.join(output_image_dir, 'true_seg.tif'), true_seg)

        for i, c in enumerate(compressors):
            compressor = c['compressor']
            filters = c['filters']

            seg_metrics = {
                'compressor_name': c['name'],
            }

            seg_metrics.update(c['params'])

            logging.info(f"starting test {len(all_metrics) + 1}/{total_tests}")
            logging.info(f"compressor: {c['name']} params: {c['params']}")

            start = timer()
            decoded = encode_decode(data, filters, compressor)
            if ridge_filter == 'frangi':
                op = ij_wrapper.frangi_cls(sigmas, res_voxel_size, np.max(decoded), num_threads)
            elif ridge_filter == 'sato':
                op = ij_wrapper.tubeness_cls(sigmas, res_voxel_size, num_threads)
            else:
                raise ValueError("Unknown filter: " + ridge_filter)
            response = filter_ij(decoded, op, ij_wrapper)
            binary = threshold(response, threshold_otsu)
            test_seg, _ = ndi.label(erode_border(binary), structure=np.ones(shape=(3,3,3), dtype=bool))
            end = timer()
            seg_dur = end - start
            logging.info(f"seg time ij = {seg_dur}")

            seg_metrics.update(compare_seg(true_seg, test_seg, metrics))

            if output_image_dir is not None:
                outfile = f"test_seg_{i}.tif"
                seg_params[outfile] = seg_metrics
                tifffile.imwrite(os.path.join(output_image_dir, outfile), test_seg)

            all_metrics.append(seg_metrics)

    output_metrics_file = output_metrics_file.replace('.csv', '_' + os.path.basename(input_file) + '.csv')

    df = pd.DataFrame.from_records(all_metrics)
    df.to_csv(output_metrics_file, index_label='test_number')

    if output_image_dir is not None:
        with open(os.path.join(output_image_dir, "params.json"), 'w') as f:
            json.dump(seg_params, f)

    scyjava.shutdown_jvm()


if __name__ == "__main__":
    main()
