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
import dask.array as da


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
    """Find a chunk shape that results in roughly as many chunks as worker threads.
    Chunks are generally close to some power-of-2 fraction of each axis, slightly
    favoring bigger values for the last index.
    """
    data_shape = np.array(data_shape)
    chunk_shape = data_shape.copy()
    ndims = len(data_shape)
    num_chunks = 1
    idx = 0
    while num_chunks < num_workers:
        chunk_shape[idx % ndims] = math.ceil(chunk_shape[idx % ndims] / 2.0)
        idx += 1
        num_chunks = math.ceil(np.product(data_shape / chunk_shape))
    return tuple(int(c) for c in chunk_shape)


def _worker(input_zarr_path, output_zarr_path, block, storage_options):
    iz = zarr.open(input_zarr_path, mode='r')
    # Read relevant interval from input zarr
    data = iz[block[0][0]:block[0][1], block[1][0]:block[1][1], block[2][0]:block[2][1]]
    oz = zarr.open(output_zarr_path, mode='r+', storage_options=storage_options)
    oz[block[0][0]:block[0][1], block[1][0]:block[1][1], block[2][0]:block[2][1]] = data


def _thread_worker(data, output_zarr_array, block):
    output_zarr_array[block[0][0]:block[0][1], block[1][0]:block[1][1], block[2][0]:block[2][1]] = \
        data[block[0][0]:block[0][1], block[1][0]:block[1][1], block[2][0]:block[2][1]]



def write_chunked_dask(input_zarr_path, output_zarr_path, full_shape, chunk_shape, block_list, compressor,
                       filters, client, credentials_file=None):
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
        credentials_file - the file containing AWS or GCS credentials, .json for GCS, .csv for AWS
    """
    blosc.use_threads = False

    store, storage_options = _init_storage(output_zarr_path, credentials_file)

    z = zarr.create(
        store=store,
        compressor=compressor,
        filters=filters,
        chunks=chunk_shape,
        shape=full_shape,
        dtype=np.uint16,
        overwrite=True
    )

    argslist = list(
        zip(itertools.repeat(input_zarr_path),
            itertools.repeat(output_zarr_path),
            block_list,
            itertools.repeat(storage_options))
    )
    futures = []
    for args in argslist:
        futures.append(client.submit(_worker, *args))
    client.gather(futures)

    return z


def write_dask(data, output_zarr_path, compressor, filters, chunk_shape, client, credentials_file=None):
    # FIXME: this does not work with s3 or gcs, get permissions errors
    blosc.use_threads = False
    store, storage_options = _init_storage(output_zarr_path, credentials_file)
    darray = da.from_array(data, chunks=chunk_shape)
    delayed_result = darray.to_zarr(store, compressor=compressor, filters=filters, compute=False,
                                    overwrite=True, storage_options=storage_options)
    _ = client.compute(delayed_result).result()
    return zarr.open(output_zarr_path, 'r')


def write_multiprocessing(input_zarr_path, output_zarr_path, full_shape, chunk_shape, block_list, compressor,
                          filters, num_workers=1, credentials_file=None):
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
        credentials_file - the file containing AWS or GCS credentials, .json for GCS, .csv for AWS
    """
    blosc.use_threads = False

    store, storage_options = _init_storage(output_zarr_path, credentials_file)

    z = zarr.create(
        store=store,
        compressor=compressor,
        filters=filters,
        chunks=chunk_shape,
        shape=full_shape,
        dtype=np.uint16,
        overwrite=True
    )

    argslist = list(
        zip(itertools.repeat(input_zarr_path),
            itertools.repeat(output_zarr_path),
            block_list,
            itertools.repeat(storage_options))
    )

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(_worker, argslist)

    return z


def write_threading(data, output_zarr_path, chunk_shape, block_list=None, compressor=None, filters=None, num_workers=1,
                    credentials_file=None):
    """Write a zarr array in parallel with Python threading.
    args:
        data             - the input array
        output_zarr_path - the path to write the output zarr file
        chunk_shape      - the chunk shape
        block_list       - list of min-max intervals used to access chunk data from the input zarr file
        compressor       - the numcodecs compressor instance
        filters          - list of numcodecs filters
        num_workers      - number of threads to split the computation
        credentials_file - the file containing AWS or GCS credentials, .json for GCS, .csv for AWS
    """
    blosc.use_threads = False

    store, _ = _init_storage(output_zarr_path, credentials_file)

    z = zarr.create(
        store=store,
        compressor=compressor,
        filters=filters,
        chunks=chunk_shape,
        shape=data.shape,
        dtype=np.uint16,
        overwrite=True
    )

    argslist = list(
        zip(itertools.repeat(data),
            itertools.repeat(z),
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


def write_default(data, output_zarr_path, compressor, filters, chunk_shape, num_workers, credentials_file):
    blosc.use_threads = True
    blosc.set_nthreads(num_workers)
    store, _ = _init_storage(output_zarr_path, credentials_file=credentials_file)
    z = zarr.array(data, chunks=chunk_shape, filters=filters, compressor=compressor, store=store, overwrite=True)
    return z


def _init_storage(output_path, credentials_file=None):
    """Credentials are required to both create the initial store from the parent process,
    and to open the store for writing from child processes. We pass credentials parsed from credentials_file
    in the storage_options dict to each worker."""
    storage_options = {}
    if output_path.startswith("s3://"):
        import s3fs
        import csv
        with open(credentials_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # ignore header
            key, secret = next(reader)
            storage_options['key'] = key
            storage_options['secret'] = secret
        s3 = s3fs.S3FileSystem(
            anon=False,
            key=key,
            secret=secret,
            client_kwargs=dict(region_name='us-west-2')
        )
        store = s3fs.S3Map(root=output_path, s3=s3, check=False)
    elif output_path.startswith("gs://"):
        import gcsfs
        storage_options['token'] = credentials_file
        gcs = gcsfs.GCSFileSystem(project='allen-nd-goog', token=storage_options['token'], default_location='US-WEST1')
        store = gcsfs.GCSMap(output_path, gcs=gcs, check=False)
    elif output_path.endswith(".n5"):
        store = zarr.N5Store(output_path)
    else:
        store = zarr.DirectoryStore(output_path)

    return store, storage_options


def main():
    # synchronizer = zarr.sync.ProcessSynchronizer("foo.sync")

    usage_text = ("Usage:" + "  zarr_parallel_test.py" + " [options]")
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("-i", "--input-file", type=str, default="/allen/scratch/aindtemp/cameron.arshadi/test-file-res0-12stack.zarr")
    parser.add_argument("-o", "--output-dir", type=str, default="/net/172.20.102.30/aind")
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
    parser.add_argument("--nchunks", type=int, default=None)
    parser.add_argument("--credentials", type=str, default=None, help="path to AWS or GCS credentials file")
    parser.add_argument("--monitor", default=False, action="store_true")

    args = parser.parse_args(sys.argv[1:])

    credentials_file = args.credentials

    logging.info(args)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(args.log_level)

    # force compatible path syntax with s3 and gcs
    output_zarr_file1 = args.output_dir + "/test-file1.zarr"
    output_zarr_file2 = args.output_dir + "/test-file2.zarr"
    output_zarr_file3 = args.output_dir + "/test-file3.zarr"
    output_zarr_file4 = args.output_dir + "/test-file4.zarr"
    output_zarr_file5 = args.output_dir + "/test-file5.zarr"

    # if os.path.exists(output_zarr_file1):
    #     shutil.rmtree(output_zarr_file1)
    # if os.path.exists(output_zarr_file2):
    #     shutil.rmtree(output_zarr_file2)
    # if os.path.exists(output_zarr_file3):
    #     shutil.rmtree(output_zarr_file3)
    # if os.path.exists(output_zarr_file4):
    #     shutil.rmtree(output_zarr_file4)
    # if os.path.exists(output_zarr_file5):
    #     shutil.rmtree(output_zarr_file5)

    logging.info(f"num cpus: {multiprocessing.cpu_count()}")

    my_slurm_kwargs = {}

    if args.monitor:
        dashboard_address = ":8787"
        my_slurm_kwargs['scheduler_options'] = {"dashboard_address": dashboard_address}

    cluster = None
    if args.slurm:
        from dask_jobqueue import SLURMCluster
        logging.info(f"Using SLURM cluster with {args.processes} workers and {args.cores} threads per worker")
        cluster = SLURMCluster(cores=args.cores, memory=f"{args.mem}GB", queue="aind", walltime="02:00:00", **my_slurm_kwargs)
        cluster.scale(args.processes)
        logging.info(cluster.job_script())
        client = Client(cluster)
    else:
        logging.info("Using local cluster")
        client = Client(n_workers=args.processes, threads_per_worker=args.cores)

    logging.info(client)

    import random
    random.seed(args.random_seed)

    z = zarr.open(args.input_file, 'r')
    data = z

    # The chunk shape determines the number of blocks.
    # In my testing, roughly 1.5-2X as many blocks as workers gives good performance.
    # A 1:1 ratio did not work as well for some reason.
    if args.chunk_shape is None:
        if args.nchunks is None:
            nchunks = args.processes * 2
        else:
            nchunks = args.nchunks
        chunk_shape = guess_chunk_shape(data.shape, nchunks)
    else:
        chunk_shape = tuple(ast.literal_eval(args.chunk_shape))
    logging.info(f"data shape {data.shape}")
    logging.info(f"chunk shape {chunk_shape}")

    compressor = blosc.Blosc(cname='zstd', clevel=1, shuffle=blosc.Blosc.SHUFFLE)

    interval_list = make_intervals(data.shape, chunk_shape)
    logging.info(f"num blocks {len(interval_list)}")

    start = timer()
    dask_chunked_result = write_chunked_dask(args.input_file, output_zarr_file1, data.shape, chunk_shape, interval_list,
                                             compressor, filters=None, client=client, credentials_file=credentials_file)
    end = timer()
    logging.info(f"dask chunked write time: {end - start}, compress MiB/s {dask_chunked_result.nbytes / 2 ** 20 / (end - start)}")

    # start = timer()
    # dask_full_result = write_dask(data, output_zarr_file2, compressor, None, chunk_shape, client, credentials_file=credentials_file)
    # end = timer()
    # logging.info(f"dask full write time: {end - start}, compress MiB/s {dask_full_result.nbytes / 2**20 / (end-start)}")

    if cluster is not None:
        cluster.close()
    client.close()

    start = timer()
    default_result = write_default(data, output_zarr_file3, compressor, None, chunk_shape, args.cores,
                                   credentials_file=credentials_file)
    end = timer()
    logging.info(f"default write time: {end - start}, compress MiB/s {default_result.nbytes / 2**20 / (end-start)}")

    if args.multiprocessing:
        start = timer()
        multiprocessing_result = write_multiprocessing(args.input_file, output_zarr_file4, data.shape,
                                                       chunk_shape, interval_list, compressor, filters=None,
                                                       num_workers=args.cores, credentials_file=credentials_file)
        end = timer()
        logging.info(f"multiprocessing write time: {end - start}, compress MiB/s "
                     f"{multiprocessing_result.nbytes / 2**20 / (end-start)}")

    if args.multithreading:
        start = timer()
        multithreading_result = write_threading(data, output_zarr_file5, chunk_shape, interval_list, compressor,
                                                filters=None, num_workers=args.cores, credentials_file=credentials_file)
        end = timer()
        logging.info(f"threading write time: {end - start}, compress MiB/s "
                     f"{multithreading_result.nbytes / 2**20 / (end-start)}")

    all_equal = np.array_equal(dask_chunked_result[:], default_result[:])
    # all_equal &= np.array_equal(dask_full_result[:], default_result[:])
    if args.multiprocessing:
        all_equal &= np.array_equal(default_result[:], multiprocessing_result[:])
    if args.multithreading:
        all_equal &= np.array_equal(default_result[:], multithreading_result[:])

    logging.info(f"All equal: {all_equal}")
    logging.info(dask_chunked_result.info)


if __name__ == "__main__":
    main()
