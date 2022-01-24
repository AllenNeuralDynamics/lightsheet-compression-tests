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
    data = z[slice, miny:maxy, minx:maxx]
    trunc_bits = [0, 2, 4, 6, 8, 10]
    row = 2
    col = 3
    fig, ax = plt.subplots(row, col, figsize=(16, 16))
    for i, tb in enumerate(trunc_bits):
        filt = numcodecs.fixedscaleoffset.FixedScaleOffset(offset=0, scale=1.0 / (2 ** tb), dtype=np.uint16)
        filtered = np.reshape(filt.decode(filt.encode(data)), data.shape)
        metrics = compress_zarr.eval_quality(data, filtered, ['mse', 'ssim', 'psnr'])
        idx = np.unravel_index(i, (row, col))
        ax[idx].imshow(filtered, cmap='gray')
        ax[idx].set_title(f"trunc_bits={tb}, mse={round(metrics['mse'], 2)}, ssim={round(metrics['ssim'], 4)}")

fig.savefig('./trunc_filters.png', dpi=500)
plt.show()
