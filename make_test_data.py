import numpy as np
import h5py
import zarr


def create_group_structure(f):
    # Simulate expected structure of data
    ds = f.create_group("t00000")
    slice = ds.create_group("s00")
    resolution = slice.create_group("1")
    cells = resolution.create_dataset("cells", shape=(4918,64,768), dtype=np.uint16)
    cells[:] = np.floor(np.random.rand(4918, 64, 768) * (2**16 - 1)).astype(np.uint16)

with zarr.open(r"C:\Users\cameron.arshadi\Desktop\test_data.zarr", mode='w') as f:
    create_group_structure(f)

with h5py.File(r"C:\Users\cameron.arshadi\Desktop\test_data.h5", mode='w') as f:
    create_group_structure(f)



