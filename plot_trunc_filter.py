import matplotlib.pyplot as plt
import numcodecs
import zarr
import tifffile
import numpy as np
compress_zarr = __import__("compress-zarr")


filepath = "/allen/scratch/aindtemp/data/anatomy/2020-12-01-training-data/2020-12-01-stack-15/images/BrainSlice1_MMStack_Pos33_15_shift.tif"
slice = 438
# bounding box coordinates for ROI
minx = 2088
maxx = 2194
miny = 1505
maxy = 1598
with tifffile.TiffFile(filepath) as f:
    z = zarr.open(f.aszarr(), 'r')
    # Extract ROI
    data = z[slice][miny:maxy, minx:maxx]

    row = 2
    col = 3
    fig, ax = plt.subplots(row, col, figsize=(16, 16))
    i = j = 0
    for trunc_bits in [0, 2, 4, 6, 8, 10]:
        filter = numcodecs.fixedscaleoffset.FixedScaleOffset(offset=0, scale=1.0 / (2 ** trunc_bits), dtype=np.uint16)
        filtered = np.reshape(filter.decode(filter.encode(data)), data.shape)
        metrics = compress_zarr.eval_quality(data, filtered, ['mse', 'ssim', 'psnr'])
        ax[i, j].imshow(filtered, cmap='gray')
        ax[i, j].set_title(f"trunc_bits={trunc_bits}, mse={round(metrics['mse'], 2)}, ssim={round(metrics['ssim'], 4)}")
        j += 1
        if j % col == 0:
            j = 0
            i += 1

fig.savefig('./trunc_filters.png', dpi=500)
plt.show()
