import argparse
import ast
import itertools
import logging
import math
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer

import numpy as np
import zarr
from distributed import Client
from numcodecs import blosc


def make_intervals(arr_shape, chunk_shape):
    """Partition an array with shape arr_shape into chunks with shape chunk_shape.
    If arr_shape is not wholly divisible by chunk_shape,
    some chunks will have the remainder as a dimension length."""
    assert (np.array(chunk_shape) <= np.array(arr_shape)).all()
    chunks = []
    for z in range(0, arr_shape[0], chunk_shape[0]):
        zint = [z, min(z + chunk_shape[0], arr_shape[0])]
        for y in range(0, arr_shape[1], chunk_shape[1]):
            yint = [y, min(y + chunk_shape[1], arr_shape[1])]
            for x in range(0, arr_shape[2], chunk_shape[2]):
                xint = [x, min(x + chunk_shape[2], arr_shape[2])]
                chunks.append(np.vstack([zint, yint, xint]))
    return chunks


def guess_chunk_shape(data_shape, num_workers):
    """Find a chunk shape that results in roughly 2X as many chunks as worker threads.
    Chunks are generally close to some power-of-2 fraction of each axis, slightly
    favoring bigger values for the last index.
    """
    data_shape = np.array(data_shape)
    chunk_shape = data_shape.copy()
    ndims = len(data_shape)
    num_chunks = np.product(data_shape / chunk_shape)
    idx = 0
    # we want more chunks than workers
    while num_chunks <= num_workers:
        chunk_shape[idx % ndims] = math.ceil(chunk_shape[idx % ndims] / 2.0)
        idx += 1
        num_chunks = np.product(data_shape / chunk_shape)
    return tuple(int(c) for c in chunk_shape)


def _worker(input_zarr_path, input_key, output_zarr_path, block):
    iz = zarr.open(input_zarr_path, mode='r')
    # Read relevant interval from input zarr
    data = iz["t00000" + "/" + input_key][block[0][0]:block[0][1], block[1][0]:block[1][1], block[2][0]:block[2][1]]
    oz = zarr.open(output_zarr_path, mode='r+', synchronizer=None)
    oz[block[0][0]:block[0][1], block[1][0]:block[1][1], block[2][0]:block[2][1]] = data


def _thread_worker(data, output_zarr_path, block):
    oz = zarr.open(output_zarr_path, mode='r+', synchronizer=None)
    oz[block[0][0]:block[0][1], block[1][0]:block[1][1], block[2][0]:block[2][1]] = \
        data[block[0][0]:block[0][1], block[1][0]:block[1][1], block[2][0]:block[2][1]]


def write_zarr_dask(input_zarr_path, input_key, output_zarr_path, full_shape, chunk_shape, block_list, compressor,
                    filters, client):
    """Write a zarr array in parallel over a Dask cluster. Data reads only occur within workers to minimize
    data movement.
    args:
        input_zarr_path  - path to the input file
        input_key        - the zarr dataset path, e.g. "s00/0/cells"
        output_zarr_path - the path to write the output zarr file
        full_shape       - the shape of the input array
        chunk_shape      - the chunk shape
        block_list       - list of min-max intervals used to access chunk data from the input zarr file
        compressor       - the numcodecs compressor instance
        filters          - list of numcodecs filters
        client           - the Dask client instance
    """
    blosc.use_threads = False

    # initialize output array
    z = zarr.open(
        output_zarr_path,
        mode='w',
        compressor=compressor,
        filters=filters,
        chunks=chunk_shape,
        shape=full_shape,
        dtype=np.uint16,
        synchronizer=None
    )

    argslist = list(
        zip(itertools.repeat(input_zarr_path),
            itertools.repeat(input_key),
            itertools.repeat(output_zarr_path),
            block_list)
    )
    futures = []
    for args in argslist:
        futures.append(client.submit(_worker, *args))
    client.gather(futures)

    return z


def write_zarr_multiprocessing(input_zarr_path, input_key, output_zarr_path, full_shape, chunk_shape, block_list, compressor,
                    filters, num_workers=1):
    """Write a zarr array in parallel with Python multiprocessing. Data reads only occurs within workers to minimize
    data movement.
    args:
        input_zarr_path  - path to the input file
        input_key        - the zarr dataset path, e.g. "s00/0/cells"
        output_zarr_path - the path to write the output zarr file
        full_shape       - the shape of the input array
        chunk_shape      - the chunk shape
        block_list       - list of min-max intervals used to access chunk data from the input zarr file
        compressor       - the numcodecs compressor instance
        filters          - list of numcodecs filters
        num_workers      - number of processes to split the computation
    """
    blosc.use_threads = False

    # initialize output array
    z = zarr.open(
        output_zarr_path,
        mode='w',
        compressor=compressor,
        filters=filters,
        chunks=chunk_shape,
        shape=full_shape,
        dtype=np.uint16,
        synchronizer=None
    )

    argslist = list(
        zip(itertools.repeat(input_zarr_path),
            itertools.repeat(input_key),
            itertools.repeat(output_zarr_path),
            block_list)
    )

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(_worker, argslist)

    return z


def write_zarr_threading(data, output_zarr_path, full_shape, chunk_shape, block_list, compressor, filters, num_workers=1):
    blosc.use_threads = False

    # initialize output array
    z = zarr.open(
        output_zarr_path,
        mode='w',
        compressor=compressor,
        filters=filters,
        chunks=chunk_shape,
        shape=full_shape,
        dtype=np.uint16,
        synchronizer=None
    )

    argslist = list(
        zip(itertools.repeat(data),
            itertools.repeat(output_zarr_path),
            block_list)
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_thread_worker, *args) for args in argslist]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                logging.error(exc)

    return z


def write_zarr(data, zarr_path, compressor, filters, chunk_shape, num_workers):
    blosc.use_threads = True
    blosc.set_nthreads(num_workers)
    ds = zarr.DirectoryStore(zarr_path)
    z = zarr.array(data, chunks=chunk_shape, filters=filters, compressor=compressor, store=ds, overwrite=True)
    return z


if __name__ == "__main__":
    # synchronizer = zarr.sync.ProcessSynchronizer("foo.sync")

    usage_text = ("Usage:" + "  zarr_parallel_test.py" + " [options]")
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("-i", "--input-file", type=str, default="/allen/scratch/aindtemp/data/anatomy/exm-hemi-brain.zarr")
    parser.add_argument("-o", "--output-dir", type=str, default="/allen/scratch/aindtemp/cameron.arshadi")
    parser.add_argument("-r", "--resolution", type=int, default=0)
    parser.add_argument("-s", "--random-seed", type=int, default=None)
    parser.add_argument("-l", "--log-level", type=str, default=logging.INFO)
    parser.add_argument("--multiprocessing", default=False, action="store_true", help="use Python multiprocessing")
    parser.add_argument("--multithreading", default=False, action="store_true", help="use Python multithreading")
    parser.add_argument("--slurm", default=False, action="store_true", help="use SLURM cluster")
    parser.add_argument("-c", "--cores", type=int, default=8)
    parser.add_argument("-p", "--processes", type=int, default=1)
    parser.add_argument("-m", "--mem", type=int, default=16)
    parser.add_argument("--chunk-shape", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])

    logging.info(args)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(args.log_level)

    output_zarr_file1 = os.path.join(args.output_dir, "test-file1.zarr")
    output_zarr_file2 = os.path.join(args.output_dir, "test-file2.zarr")
    output_zarr_file3 = os.path.join(args.output_dir, "test-file3.zarr")
    output_zarr_file4 = os.path.join(args.output_dir, "test-file4.zarr")

    if os.path.exists(output_zarr_file1):
        shutil.rmtree(output_zarr_file1)
    if os.path.exists(output_zarr_file2):
        shutil.rmtree(output_zarr_file2)
    if os.path.exists(output_zarr_file3):
        shutil.rmtree(output_zarr_file3)
    if os.path.exists(output_zarr_file4):
        shutil.rmtree(output_zarr_file4)

    if args.slurm:
        from dask_jobqueue import SLURMCluster
        logging.info(f"Using SLURM cluster with {args.cores} cores and {args.processes} processes")
        # 1 process with many cores seems to work better than processes ~= cores
        cluster = SLURMCluster(cores=args.cores, processes=args.processes, memory=f"{args.mem}GB", queue="aind")
        # cluster.adapt(1, args.cores)
        cluster.scale(args.cores)  # ??
        client = Client(cluster)
    else:
        logging.info("Using local cluster")
        client = Client()

    logging.info(client)

    import random
    random.seed(args.random_seed)

    z = zarr.open(args.input_file, 'r')
    ds = z['t00000']

    resolution = args.resolution
    tile = random.choice(list(ds.keys()))
    key = f"{tile}/{resolution}/cells"
    logging.info(f"loading tile {key}")

    data = ds[key][()]

    # The chunk shape determines the number of blocks.
    # In my testing, roughly 1.5-2X as many blocks as workers gives good performance.
    # A 1:1 ratio did not work as well for some reason.
    if args.chunk_shape is None:
        chunk_shape = guess_chunk_shape(data.shape, args.cores)
    else:
        chunk_shape = tuple(ast.literal_eval(args.chunk_shape))
    logging.info(f"data shape {data.shape}")
    logging.info(f"chunk shape {chunk_shape}")

    compressor = blosc.Blosc(cname='zstd', clevel=1, shuffle=blosc.Blosc.SHUFFLE)

    interval_list = make_intervals(data.shape, chunk_shape)
    logging.info(f"num blocks {len(interval_list)}")

    start = timer()
    dask_result = write_zarr_dask(args.input_file, key, output_zarr_file1, data.shape, chunk_shape, interval_list,
                                  compressor, filters=None, client=client)
    end = timer()
    logging.info(f"dask write time: {end - start}, compress MiB/s {dask_result.nbytes / 2**20 / (end-start)}")

    start = timer()
    default_result = write_zarr(data, output_zarr_file2, compressor, None, chunk_shape, args.cores)
    end = timer()
    logging.info(f"default write time: {end - start}, compress MiB/s {default_result.nbytes / 2**20 / (end-start)}")

    if args.multiprocessing:
        start = timer()
        multiprocessing_result = write_zarr_multiprocessing(args.input_file, key, output_zarr_file3, data.shape,
                                                            chunk_shape, interval_list, compressor, filters=None,
                                                            num_workers=args.cores)
        end = timer()
        logging.info(f"multiprocessing write time: {end - start}, compress MiB/s "
                     f"{multiprocessing_result.nbytes / 2**20 / (end-start)}")

    if args.multithreading:
        start = timer()
        multithreading_result = write_zarr_threading(data, output_zarr_file4, data.shape,
                                                     chunk_shape, interval_list, compressor, filters=None,
                                                     num_workers=args.cores)
        end = timer()
        logging.info(f"multithreading write time: {end - start}, compress MiB/s "
                     f"{multithreading_result.nbytes / 2**20 / (end-start)}")

    all_equal = np.array_equal(dask_result, default_result)
    if args.multiprocessing:
        all_equal &= np.array_equal(default_result, multiprocessing_result)
    if args.multithreading:
        all_equal &= np.array_equal(default_result, multithreading_result)

    logging.info(f"All equal: {all_equal}")
    logging.info(dask_result.info)
