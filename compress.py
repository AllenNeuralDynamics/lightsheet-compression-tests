import h5py
import random
import zarr
import numcodecs
import argparse
import sys

from timeit import default_timer as timer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--num-tiles", type=int, default=1)
    parser.add_argument("-r","--resolution", type=str, default="1")
    parser.add_argument("-s","--random-seed", type=int, default=None)
    parser.add_argument("-i","--input-file", type=str, default="/allen/scratch/aindtemp/data/anatomy/exm-hemi-brain/data.h5")
    parser.add_argument("-o","--output-file", type=str, default="./test_file.zarr")

    args = parser.parse_args(sys.argv[1:])

    run(**vars(args))

def run(num_tiles, resolution, random_seed, input_file, output_file):

    if random_seed is not None:
        random.seed(random_seed)

    total_bytes_read = 0
    total_read_time = 0
    total_compress_time = 0
    total_write_time = 0
    total_bytes_stored = 0

    compressor = numcodecs.Blosc(cname='zstd', clevel=1, shuffle=numcodecs.Blosc.SHUFFLE)

    with h5py.File(input_file) as f:
        ds = f["t00000"]

        for ti in range(num_tiles):
            rslice = random.choice(list(ds.keys()))

            print(f"loading {rslice}")

            start = timer()
            data = ds[rslice][resolution]["cells"][()]
            end = timer()
            read_dur = end - start
            print(f"loaded {data.shape} {data.nbytes} bytes, time {read_dur}s")


            start = timer()
            za = zarr.array(data, chunks=True, compressor=compressor)
            end = timer()
            compress_dur = end - start

            print(f"compression time = {compress_dur}, bps = {data.nbytes / compress_dur}, ratio = {za.nbytes/za.nbytes_stored}")

            start = timer()

            start = timer()
            zarr.save(output_file, za)
            end = timer()
            write_dur = end - start

            print(f"write time = {write_dur}s")
            
            
            total_read_time += read_dur
            total_compress_time += compress_dur
            total_write_time += write_dur

            total_bytes_read += data.nbytes
            total_bytes_stored += za.nbytes_stored
            
        print(f"compress bps {total_bytes_read / total_compress_time}")
        print(f"compression ratio = {total_bytes_read/total_bytes_stored}")
        print(f"read bps {total_bytes_read / total_read_time}")
        print(f"write bps {total_bytes_stored / total_write_time}")
        
    

if __name__ == "__main__": main()


