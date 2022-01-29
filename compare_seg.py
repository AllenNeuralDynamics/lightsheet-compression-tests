import argparse
import json
import logging
import os
import shutil
import sys
from fractions import Fraction
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import skimage
import skimage.filters
import tifffile
import zarr
from skimage.filters.thresholding import threshold_mean
from skimage.metrics import adapted_rand_error, variation_of_information

import compress_zarr

logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
logging.getLogger().setLevel(logging.INFO)

try:
    import imagej
    logging.info("Found pyimagej")
    HAS_IMAGEJ = True
except ImportError:
    logging.warning("pyimagej not found, using scikit-image instead")
    HAS_IMAGEJ = False

if HAS_IMAGEJ:
    import scyjava
    # In megabytes
    MAX_JVM_MEMORY = 12000
    # Increase JVM memory, 32-bit float images can be large...
    scyjava.config.add_option(f"-Xmx{MAX_JVM_MEMORY}m")
    # Start ImageJ with maven endpoints for Fiji and SNT.
    # Headless since we won't display any Java-based UI components
    ij = imagej.init(['sc.fiji:fiji', 'org.morphonets:SNT:4.0.8'], headless=True)
    # Java classes can only be imported once the JVM starts
    SNTUtils = scyjava.jimport("sc.fiji.snt.SNTUtils")
    # This should be 4.0.8, but I get 4.0.3??
    logging.info("We are running SNT version " + str(SNTUtils.VERSION))  # Java string
    # Frangi, et al., 1998
    Frangi = scyjava.jimport("sc.fiji.snt.filter.Frangi")
    # Sato, et al., 1998
    Tubeness = scyjava.jimport("sc.fiji.snt.filter.Tubeness")
    FloatType = scyjava.jimport("net.imglib2.type.numeric.real.FloatType")
    Runtime = scyjava.jimport("java.lang.Runtime")


def encode_decode(im, filters, compressor):
    za = zarr.array(im, chunks=True, filters=filters, compressor=compressor)
    return za[:]


def filter_ij(im, op):
    # Convert numpy array to Java RandomAccessibleInterval
    # This automatically swaps axes from ZYX -> XYZ
    java_in = ij.py.to_java(im)
    java_out = ij.op().run("create.img", java_in, FloatType())
    op.compute(java_in, java_out)
    # Back to numpy and ZYX order
    return ij.py.from_java(java_out)


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
    parser.add_argument("-d", "--output-image-dir", type=str, default="./images")  # Optional, will save a lot of images
    parser.add_argument("-o", "--output-metrics-file", type=str, default="./segmentation_metrics.csv")
    parser.add_argument("-l", "--log-level", type=str, default=logging.INFO)
    parser.add_argument("-c", "--codecs", nargs="+", type=str, default=["blosc"])
    parser.add_argument("-t", "--trunc-bits", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("-m", "--metrics", nargs="+", type=str, default=['are', 'voi'])  # ['are', 'voi']

    args = parser.parse_args(sys.argv[1:])
    print(args)

    logging.getLogger().setLevel(args.log_level)

    if args.output_image_dir is not None:
        if os.path.isdir(args.output_image_dir):
            shutil.rmtree(args.output_image_dir)
        os.mkdir(args.output_image_dir)

    chunk_factor = [1]
    compressors = compress_zarr.build_compressors(args.codecs, args.trunc_bits, chunk_factor)

    run(args.input_file, compressors, args.output_image_dir, args.output_metrics_file, args.metrics)


def parse_tiff_metadata(f):
    xres = Fraction(*f.pages[0].tags['XResolution'].value)
    yres = Fraction(*f.pages[0].tags['YResolution'].value)
    spacing = float(f.imagej_metadata['spacing'])
    voxel_spacing = [1. / float(xres), 1. / float(yres), spacing]
    bytes_per_sample = int(f.pages[0].tags['BitsPerSample'].value) / 8
    return voxel_spacing, bytes_per_sample


def run(input_file, compressors, output_image_dir, output_metrics_file, metrics):
    with tifffile.TiffFile(input_file) as f:
        voxel_spacing, bytes_per_sample = parse_tiff_metadata(f)
        data = f.asarray()
        logging.info(f"loading volume, shape {data.shape},"
                     f" size {(np.product(data.shape) * bytes_per_sample) / (1024. * 1024)} MiB,"
                     f" voxel spacing {voxel_spacing}")

    if output_image_dir is not None:
        tifffile.imwrite(os.path.join(output_image_dir, 'input_data.tif'), data)

    if HAS_IMAGEJ:
        # # Nyquist
        # scale_step = sum(voxel_spacing) / (2 * len(voxel_spacing))
        # print(f"scale step: {scale_step}")
        # # N scales takes N times as long
        # ij_sigmas = scale_step * np.arange(1, 5)
        # print("sigmas: " + str(ij_sigmas))
        ij_sigmas = [1.0]
        num_threads = Runtime.getRuntime().availableProcessors()
        response = filter_ij(data, Frangi(ij_sigmas, voxel_spacing, np.max(data), num_threads))
    else:
        # Voxel size is ignored in scikit-image filters, use pixel units
        py_sigmas = [2.0]  # 4x half of min-separation in pixels
        response = skimage.filters.frangi(data, py_sigmas, black_ridges=False)

    threshold_func = threshold_mean

    true_seg = threshold(response, threshold_func)

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

        if HAS_IMAGEJ:
            start = timer()
            decoded = encode_decode(data, filters, compressor)
            response = filter_ij(decoded, Frangi(ij_sigmas, voxel_spacing, np.max(decoded), num_threads))
            test_seg = threshold(response, threshold_func)
            end = timer()
            seg_dur = end - start
            logging.info(f"seg time ij = {seg_dur}")
        else:
            start = timer()
            response = skimage.filters.frangi(encode_decode(data, filters, compressor), sigmas=py_sigmas,
                                              black_ridges=False)
            test_seg = threshold(response, threshold_func)
            end = timer()
            seg_dur = end - start
            logging.info(f"seg time sklearn = {seg_dur}")

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
        with open(os.path.join(output_image_dir, "seg_params.json"), 'w') as f:
            json.dump(seg_params, f)

    if HAS_IMAGEJ:
        scyjava.shutdown_jvm()


if __name__ == "__main__":
    main()
