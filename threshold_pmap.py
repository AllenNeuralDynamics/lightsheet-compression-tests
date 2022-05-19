import os
import h5py
import numpy as np
import scipy.ndimage as ndi
import tifffile
import pandas as pd

import segment


def collect_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    return sorted(files)


gt_dir = r"C:\Users\cameron.arshadi\Desktop\repos\lightsheet-compression-tests\data\validation\gt"
gt_files = collect_files(gt_dir)

trunc = [0, 2, 4]
all_metrics = []

pmap_threshold = 0.95

for i, gt in enumerate(gt_files):
    gt_seg = tifffile.imread(gt)
    for tb in trunc:
        pmap_dir = rf'C:\Users\cameron.arshadi\Desktop\repos\lightsheet-compression-tests\data\validation\pmap\trunc_{tb}'
        pmap_files = collect_files(pmap_dir)
        pmap = pmap_files[i]
        with h5py.File(os.path.join(pmap)) as h5:
            ds = h5['exported_data']
            # Ilastik HDF5 files have axes TZYXC
            # extract foreground channel
            foreground = ds[0, :, :, :, 0]
        binary = foreground > pmap_threshold
        test_seg, _ = ndi.label(binary, structure=None, output=np.uint32)
        metrics = ['are', 'voi']
        results = segment.compare_seg(gt_seg, test_seg, metrics)
        results['trunc'] = tb
        all_metrics.append(results)

df = pd.DataFrame.from_records(all_metrics)
df.to_csv("./trunc-segmentation-metrics.csv", index_label='test_number')
