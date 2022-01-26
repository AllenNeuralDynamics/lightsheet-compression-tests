import os
import skimage
import zarr
import tifffile
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from skimage.metrics import adapted_rand_error
import skimage.filters
from skimage.filters.thresholding import threshold_otsu, threshold_mean
import compress_zarr


USE_IMAGEJ = True

if USE_IMAGEJ:
    import imagej
    import scyjava
    # Increase JVM memory, 32-bit float images can be large...
    scyjava.config.add_option("-Xmx12g")
    # Start ImageJ
    ij = imagej.init(['sc.fiji:fiji', 'org.morphonets:SNT'], headless=True)
    # Java classes can only be imported once the JVM starts
    SNTUtils = scyjava.jimport("sc.fiji.snt.SNTUtils")
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
    metrics['adapted_rand_error'], metrics['precision'], metrics['recall'] = adapted_rand_error(true_seg, test_seg)
    return metrics


def threshold(im, func):
    return im > func(im)


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(logging.INFO)

    input_file = r"C:\Users\cameron.arshadi\Downloads\BrainSlice1_MMStack_Pos33_15_shift.tif"
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

    with tifffile.TiffFile(input_file) as f:
        z = zarr.open(f.aszarr(), 'r')
        data = z[min_z:max_z, min_y:max_y, min_x:max_x]
        logging.info(f"loading volume, size {(np.product(data.shape) * 2) / (1024. * 1024)} MiB")

    trunc_bits = [0, 2, 4]
    chunk_factor = [1]
    compressors = compress_zarr.build_compressors("blosc", trunc_bits)
    print(len(compressors))

    # fig, axes = plt.subplots(nrows=len(compressors), ncols=2)

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
        py_op = skimage.filters.frangi
        response = py_op(data, py_sigmas, black_ridges=False)

    threshold_func = threshold_mean

    true_seg = threshold(response, threshold_func)

    all_metrics = []

    total_tests = len(compressors)

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
            response = filter_ij(decoded, Frangi(ij_sigmas, voxel_spacing, np.max(decoded)))
            test_seg = threshold(response, threshold_func)
            end = timer()
            seg_dur = end - start
            logging.info(f"seg time ij = {seg_dur}")
        else:
            start = timer()
            response = skimage.filters.frangi(encode_decode(data, filters, compressor), sigmas=py_sigmas, black_ridges=False)
            test_seg = threshold(response, threshold_func)
            end = timer()
            seg_dur = end - start
            logging.info(f"seg time sklearn = {seg_dur}")

        seg_metrics.update(compare_seg(true_seg, test_seg))
        all_metrics.append(seg_metrics)

        # axes[i, 0].imshow(np.max(true_seg, axis=0), cmap=plt.cm.gray, vmin=0, vmax=1)
        # axes[i, 0].set_title('True segmentation')
        # axes[i, 1].imshow(np.max(test_seg, axis=0), cmap=plt.cm.gray, vmin=0, vmax=1)
        # axes[i, 1].set_title(f'Rand Error={metrics["Adapted Rand Error"]}')

    output_metrics_file = output_metrics_file.replace('.csv', '_' + os.path.basename(input_file) + '.csv')

    df = pd.DataFrame.from_records(all_metrics)
    df.to_csv(output_metrics_file, index_label='test_number')

    if USE_IMAGEJ:
        scyjava.shutdown_jvm()

    # plt.show()


if __name__ == "__main__":
    main()
