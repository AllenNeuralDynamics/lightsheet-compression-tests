import zarr
import tifffile
import numcodecs
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import adapted_rand_error
from skimage.filters import sato, frangi, hessian
from skimage.filters.thresholding import threshold_otsu, threshold_mean
import compress_zarr

from timeit import default_timer as timer


def do_comparison(im, filters, compressor, post_filter_func, threshold_func, **kwargs):
    za = zarr.array(im, chunks=True, filters=filters, compressor=compressor)
    decoded = za[:]

    data_filtered = post_filter_func(im, **kwargs)
    true_seg = data_filtered > threshold_func(data_filtered)

    decoded_filtered = post_filter_func(decoded, **kwargs)
    test_seg = decoded_filtered > threshold_func(decoded_filtered)

    # tifffile.imwrite("./test_seg.tif", true_seg)
    # tifffile.imwrite("./true_seg.tif", test_seg)

    are, prec, rec = adapted_rand_error(true_seg, test_seg)
    print(f"adapted rand error: {are}")
    print(f"precision: {prec}")
    print(f"recall: {rec}")

    return true_seg, test_seg, are, prec, rec


def main():

    filepath = r"C:\Users\cameron.arshadi\Downloads\BrainSlice1_MMStack_Pos33_15_shift.tif"
    # Voxel spacing from image metadata
    xy_res = 0.255
    z_res = 0.999
    # Coordinates of bounding cuboid ROI
    min_z = 115
    max_z = 232
    min_y = 1265
    max_y = 1872
    min_x = 2188
    max_x = 2817

    with tifffile.TiffFile(filepath) as f:
        z = zarr.open(f.aszarr(), 'r')
        data = z[min_z:max_z, min_y:max_y, min_x:max_x]

    trunc_bits = [0]
    chunk_factor = [1]
    # I can only get through two of these before running out of memory
    compressors = compress_zarr.build_compressors("lossy", trunc_bits, chunk_factor)

    fig, axes = plt.subplots(nrows=len(compressors), ncols=2)

    for i, c in enumerate(compressors):
        compressor = c['compressor']
        filters = c['filters']

        # Half the average voxel spacing
        #step = (2 * pixsize + spacing) / 6
        # N scales takes N times as long
        #sigmas = step * np.arange(1, 5)
        sigmas = [2.0]

        filter_func = frangi
        threshold_func = threshold_mean
        true_seg, test_seg, are, prec, rec = do_comparison(data, filters, compressor, filter_func, threshold_func,
                                                           sigmas=sigmas, black_ridges=False)

        axes[i, 0].imshow(np.max(true_seg, axis=0), cmap=plt.cm.gray, aspect="auto", vmin=0, vmax=1)
        axes[i, 0].set_title('True segmentation')
        axes[i, 1].imshow(np.max(test_seg, axis=0), cmap=plt.cm.gray, aspect="auto", vmin=0, vmax=1)
        axes[i, 1].set_title(f'Rand Error={are}')

    plt.show()


if __name__ == "__main__":
    main()
