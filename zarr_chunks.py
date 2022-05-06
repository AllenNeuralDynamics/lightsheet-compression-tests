import multiprocessing
import numpy as np
import numcodecs
import tifffile
import zarr
from timeit import default_timer as timer
import zarr_parallel_test
import random
import math
import scipy.stats
import sys
import argparse
import pandas as pd
import logging
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="/allen/scratch/aindtemp/cameron.arshadi/test-file-res0-12stack.zarr")
    parser.add_argument("--output-file", type=str, default="/allen/programs/aind/workgroups/msma/test-file.zarr")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--credentials", type=str, default=None, help="AWS or GCS credentials file")
    parser.add_argument("--metrics-file", type=str, default="/allen/scratch/aindtemp/cameron.arshadi/chunksize-metrics.csv")

    args = parser.parse_args(sys.argv[1:])

    input_file = args.input_file
    output_file = args.output_file

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(logging.INFO)

    f = zarr.open(input_file, 'r')
    data = f[:]
    logging.info(f"Loaded data with shape: {data.shape}, size {data.nbytes / 2**20} MiB")

    credentials_fpath = args.credentials

    threads = args.threads
    logging.info(f"using {threads} threads")

    metrics_fpath = args.metrics_file
    
    #compressor = numcodecs.Blosc(cname='zstd', clevel=1, shuffle=numcodecs.Blosc.SHUFFLE)
    compressor = None
    
    bits = 2
    scale = 1.0 / (2 ** bits)
    #filters = [ numcodecs.fixedscaleoffset.FixedScaleOffset(offset=0, scale=scale, dtype=np.uint16) ]
    filters = []
    
    chunk_factors = [1, 2, 4, 8, 16, 32, 64, 128]
    
    all_metrics = []
    for cf in chunk_factors:
        chunks_metrics = {}
        # chunk along Z
        chunks = (int(math.ceil(data.shape[0] / (threads * cf))), data.shape[1], data.shape[2])
        chunks_metrics['chunk_shape'] = chunks
        logging.info(f"chunk shape: {chunks}")

        chunk_size = np.product(chunks) * data.itemsize
        chunks_metrics['chunk_size'] = chunk_size
        logging.info(f"chunk size: {chunk_size}")

        blocks = np.array(chunks)
        # block shape (thread FOV) must be a multiple of chunk shape
        blocks[0] *= cf
        chunks_metrics['block_shape'] = blocks
        logging.info(f"block shape: {blocks}")

        blocklist = zarr_parallel_test.make_blocks(data.shape, blocks)
        logging.info(f"num blocks: {len(blocklist)}")

        num_trials = 5
        for i in range(num_trials):
            # Write the output Zarr
            start = timer()
            zarr_parallel_test.write_threading(
                data,
                output_file,
                chunks,
                blocklist,
                compressor,
                filters=filters,
                num_workers=threads,
                credentials_file=credentials_fpath
            )
            end = timer()
            write_time = end - start
            write_speed = data.nbytes / write_time
            chunks_metrics['write_time'] = write_time
            chunks_metrics['write_bps'] = write_speed
            logging.info(f"write time: {write_time}, compress MiB/s {write_speed / 2**20}")

            # Read the output Zarr
            start = timer()
            zarr_parallel_test.read_threading(output_file, blocks, threads)
            end = timer()
            read_time = end-start
            chunks_metrics['read_time'] = read_time
            read_speed = data.nbytes / read_time
            chunks_metrics['read_bps'] = read_speed
            logging.info(f"read time: {read_time}, read MiB/s {read_speed / 2**20}")

            all_metrics.append(chunks_metrics)

    df = pd.DataFrame.from_records(all_metrics)
    df.to_csv(metrics_fpath, index_label='test_number')
