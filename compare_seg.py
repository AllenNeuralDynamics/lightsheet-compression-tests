import argparse
import json
import logging
import os
import shutil
import sys
from fractions import Fraction
from timeit import default_timer as timer

import imagej
import numpy as np
import pandas as pd
import scyjava
import tifffile
import zarr
from skimage.filters.thresholding import threshold_mean
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
    parser.add_argument("-i", "--input-file", type=str, default=r"./chunk.tif")
    parser.add_argument("-f", "--filter", type=str, default='sato')  # 'frangi' or 'sato'
    # Scales for ridge filters, these are multiplied by the Nyquist rate
    # i.e., half the average voxel spacing of the input image
    parser.add_argument("-s", "--scales", nargs="+", type=float, default=[2])
    # Max memory for JVM in megabytes, i.e., -Xmx{max_memory}m
    parser.add_argument("-x", "--max-memory", type=int, default=8000)
    parser.add_argument("-d", "--output-image-dir", type=str, default="./segmented")  # Optional, will save a lot of images
    parser.add_argument("-o", "--output-metrics-file", type=str, default="./segmentation_metrics.csv")
    parser.add_argument("-l", "--log-level", type=str, default=logging.INFO)
    parser.add_argument("-c", "--codecs", nargs="+", type=str, default=["blosc"])
    parser.add_argument("-t", "--trunc-bits", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("-m", "--metrics", nargs="+", type=str, default=['are', 'voi'])  # ['are', 'voi']

    args = parser.parse_args(sys.argv[1:])
    print(args)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(args.log_level)

    if args.output_image_dir is not None:
        if os.path.isdir(args.output_image_dir):
            shutil.rmtree(args.output_image_dir)
        os.mkdir(args.output_image_dir)

    chunk_factor = [1]
    compressors = compress_zarr.build_compressors(args.codecs, args.trunc_bits, chunk_factor)

    maven_endpoints = ['sc.fiji:fiji', 'org.morphonets:SNT:4.0.8']
    ij_wrapper = ImageJWrapper(maven_endpoints, max_memory=args.max_memory, headless=True)
    # This really should be 4.0.8, but I get 4.0.3??
    logging.info("We are running SNT version " + str(ij_wrapper.sntutils_cls.VERSION))

    run(args.input_file, ij_wrapper, args.filter, args.scales, compressors, args.output_image_dir, args.output_metrics_file,
        args.metrics)


def parse_tiff_metadata(f):
    xres = Fraction(*f.pages[0].tags['XResolution'].value)
    yres = Fraction(*f.pages[0].tags['YResolution'].value)
    spacing = float(f.imagej_metadata['spacing'])
    voxel_spacing = [1. / float(xres), 1. / float(yres), spacing]
    bytes_per_sample = int(f.pages[0].tags['BitsPerSample'].value) / 8
    return voxel_spacing, bytes_per_sample


def run(input_file, ij_wrapper, ridge_filter, scales, compressors, output_image_dir, output_metrics_file, metrics):
    with tifffile.TiffFile(input_file) as f:
        voxel_spacing, bytes_per_sample = parse_tiff_metadata(f)
        data = f.asarray()
        logging.info(f"loading volume, shape {data.shape},"
                     f" size {(np.product(data.shape) * bytes_per_sample) / (1024. * 1024)} MiB,"
                     f" voxel spacing {voxel_spacing}")

    if output_image_dir is not None:
        tifffile.imwrite(os.path.join(output_image_dir, 'input_data.tif'), data)

    # Nyquist
    scale_step = sum(voxel_spacing) / (2 * len(voxel_spacing))
    print(f"scale step: {scale_step}")
    # N scales takes N times as long
    sigmas = scale_step * np.array(scales)
    print("sigmas: " + str(sigmas))

    num_threads = ij_wrapper.runtime_cls.getRuntime().availableProcessors()

    if ridge_filter == 'frangi':
        op = ij_wrapper.frangi_cls(sigmas, voxel_spacing, np.max(data), num_threads)
    elif ridge_filter == 'sato':
        op = ij_wrapper.tubeness_cls(sigmas, voxel_spacing, num_threads)
    else:
        raise ValueError("Unknown filter: " + ridge_filter)

    response = filter_ij(data, op, ij_wrapper)

    true_seg = threshold(response, threshold_mean)

    if output_image_dir is not None:
        tifffile.imwrite(os.path.join(output_image_dir, 'true_seg.tif'), true_seg)

    all_metrics = []

    total_tests = len(compressors)

    # Dictionary mapping segmentation result image to its parameters
    seg_params = {}

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
        response = filter_ij(decoded, ij_wrapper.frangi_cls(sigmas, voxel_spacing, np.max(decoded), num_threads), ij_wrapper)
        test_seg = threshold(response, threshold_mean)
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
