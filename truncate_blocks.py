import os

import numcodecs
import numpy as np
import tifffile
import h5py


indir = r"C:\Users\cameron.arshadi\Desktop\repos\lightsheet-compression-tests\data\validation"
dirs = [os.path.join(indir, f) for f in os.listdir(indir) if os.path.isdir(os.path.join(indir, f)) and f.startswith("block_")]
bits = [0,2,4]
for b in bits:
    outdir = r"C:\Users\cameron.arshadi\Desktop\repos\lightsheet-compression-tests\data\validation\raw_data\trunc_{}".format(b)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filter = numcodecs.fixedscaleoffset.FixedScaleOffset(offset=0, scale=1.0 / (2 ** b), dtype=np.uint16)
    for dir in dirs:
        imfile = os.path.join(dir, "input.tif")
        if not os.path.isfile(imfile):
            continue
        data = tifffile.imread(imfile)
        filtered = np.reshape(filter.decode(filter.encode(data)), data.shape)
        block_name = dir.split('\\')[-1]
        outfile = os.path.join(outdir, block_name + f"_trunc_{b}.h5")
        print("Writing " + outfile)
        with h5py.File(outfile, 'w') as f:
            ds = f.create_dataset("data", shape=(1, *data.shape, 1), dtype=np.uint16)
            ds[...] = filtered[np.newaxis, :, :, :, np.newaxis]