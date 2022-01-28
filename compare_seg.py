import json
import logging
import os
import shutil
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import skimage
import skimage.filters
import tifffile
import zarr
from skimage.filters.thresholding import threshold_mean
from skimage.metrics import adapted_rand_error

import compress_zarr

USE_IMAGEJ = True

if USE_IMAGEJ:
    import imagej
    import scyjava

    # Increase JVM memory, 32-bit float images can be large...
    scyjava.config.add_option("-Xmx12g")
    # Start ImageJ with maven endpoints for Fiji and SNT.
    # Headless since we won't display any Java-based UI components
    ij = imagej.init(['sc.fiji:fiji', 'org.morphonets:SNT'], headless=True)
    # Java classes can only be imported once the JVM starts
    SNTUtils = scyjava.jimport("sc.fiji.snt.SNTUtils")
    # This should be 4.0.8, but I get 4.0.3??
    print("We are running SNT version " + str(SNTUtils.VERSION))  # Java string
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


def compare_seg(true_seg, test_seg):
    metrics = dict()
    metrics['adapted_rand_error'], metrics['prec'], metrics['rec'] = adapted_rand_error(true_seg, test_seg)
    return metrics


def threshold(im, func):
    return im > func(im)


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(logging.INFO)

    input_file = r"C:\Users\cameron.arshadi\Downloads\BrainSlice1_MMStack_Pos33_15_shift.tif"

    # This will be created later
    ground_truth_file = './true_seg.tif'

    output_image_dir = "./images"
    if os.path.isdir(output_image_dir):
        shutil.rmtree(output_image_dir)
    os.mkdir(output_image_dir)

    seg_params_file = os.path.join(output_image_dir, "seg_params.json")

    output_metrics_file = "./segmentation_metrics.csv"

    # Voxel spacing from image metadata
    voxel_spacing = [0.255, 0.255, 0.999]

    # Coordinates of bounding cuboid ROI
    min_z = 175
    max_z = 207
    min_y = 1435
    max_y = 1730
    min_x = 2325
    max_x = 2513
    # Bigger block, filtering takes ~7X longer
    # min_z = 115
    # max_z = 232
    # min_y = 1265
    # max_y = 1872
    # min_x = 2188
    # max_x = 2817
    xbounds = (min_x, max_x)
    ybounds = (min_y, max_y)
    zbounds = (min_z, max_z)

    run(input_file, xbounds, ybounds, zbounds, voxel_spacing, ground_truth_file, output_image_dir, seg_params_file,
        output_metrics_file)


def run(input_file, xbounds, ybounds, zbounds, voxel_spacing, ground_truth_outfile, output_image_dir, seg_params_file,
        output_metrics_file):
    with tifffile.TiffFile(input_file) as f:
        z = zarr.open(f.aszarr(), 'r')
        data = z[zbounds[0]:zbounds[1], ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
        logging.info(f"loading volume, shape {data.shape}, size {(np.product(data.shape) * 2) / (1024. * 1024)} MiB")

    trunc_bits = [0, 2, 4, 8]
    chunk_factor = [1]
    compressors = compress_zarr.build_compressors("blosc", trunc_bits)
    print(len(compressors))

    tifffile.imwrite(os.path.join(output_image_dir, 'input_data.tif'), data)

    if USE_IMAGEJ:
        # Nyquist
        scale_step = sum(voxel_spacing) / (2 * len(voxel_spacing))
        print(f"scale step: {scale_step}")
        # N scales takes N times as long
        ij_sigmas = scale_step * np.arange(1, 5)
        print("sigmas: " + str(ij_sigmas))
        ij_sigmas = [1.0]
        num_threads = Runtime.getRuntime().availableProcessors()
        response = filter_ij(data, Frangi(ij_sigmas, voxel_spacing, np.max(data), num_threads))
    else:
        # Voxel size is ignored in scikit-image filters, use pixel units
        py_sigmas = [2.0]
        response = skimage.filters.frangi(data, py_sigmas, black_ridges=False)

    threshold_func = threshold_mean

    true_seg = threshold(response, threshold_func)
    tifffile.imwrite(ground_truth_outfile, true_seg)

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

        if USE_IMAGEJ:
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

        outfile = f"test_seg_{i}.tif"
        tifffile.imwrite(os.path.join(output_image_dir, outfile), test_seg)

        seg_metrics.update(compare_seg(true_seg, test_seg))
        seg_params[outfile] = seg_metrics
        all_metrics.append(seg_metrics)

    output_metrics_file = output_metrics_file.replace('.csv', '_' + os.path.basename(input_file) + '.csv')

    df = pd.DataFrame.from_records(all_metrics)
    df.to_csv(output_metrics_file, index_label='test_number')

    with open(seg_params_file, 'w') as f:
        json.dump(seg_params, f)

    if USE_IMAGEJ:
        scyjava.shutdown_jvm()


if __name__ == "__main__":
    main()
