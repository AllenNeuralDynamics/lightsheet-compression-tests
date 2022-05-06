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


def make_blocks(arr_shape, block_shape):
    """Partition an array with shape arr_shape into chunks with shape chunk_shape.
    If arr_shape is not wholly divisible by chunk_shape,
    some chunks will have the remainder as a dimension length."""
    assert (np.array(block_shape) <= np.array(arr_shape)).all()
    blocks = []
    for z in range(0, arr_shape[0], block_shape[0]):
        zint = [z, min(z + block_shape[0], arr_shape[0])]
        for y in range(0, arr_shape[1], block_shape[1]):
            yint = [y, min(y + block_shape[1], arr_shape[1])]
            for x in range(0, arr_shape[2], block_shape[2]):
                xint = [x, min(x + block_shape[2], arr_shape[2])]
                blocks.append(np.vstack([zint, yint, xint]))
    return blocks


def guess_blocks(data_shape, chunk_shape, num_workers, mode="z"):
    block_shape = np.array(chunk_shape)
    num_blocks = math.ceil(np.product(data_shape / block_shape))
    if mode == "z":
        while num_blocks > 2 * num_workers:
            block_shape[0] *= 2
            num_blocks = math.ceil(np.product(data_shape / block_shape))
    elif mode == "cycle":
        ndims = len(data_shape)
        idx = 0
        while num_blocks > 2 * num_workers:
            block_shape[idx % ndims] *= 2
            idx += 1
            num_blocks = math.ceil(np.product(data_shape / block_shape))
    else:
        raise ValueError(f"Invalid mode {mode}")

    # convert numpy int64 to Python int or zarr will complain
    return tuple(int(d) for d in block_shape)


def guess_chunks(data_shape, target_size, bytes_per_pixel, mode="z"):
    chunk_shape = np.array(data_shape)
    if mode == "z":
        while np.product(chunk_shape) * bytes_per_pixel > target_size:
            chunk_shape[0] = int(math.ceil(chunk_shape[0] / 2.0))
    elif mode == "cycle":
        ndims = len(data_shape)
        idx = 0
        while np.product(chunk_shape) * bytes_per_pixel > target_size:
            chunk_shape[idx % ndims] = int(math.ceil(chunk_shape[idx % ndims] / 2.0))
            idx += 1
    else:
        raise ValueError(f"Invalid mode {mode}")

    # convert numpy int64 to Python int or zarr will complain
    return tuple(int(d) for d in chunk_shape)


def _worker(input_zarr_path, output_zarr_path, block, storage_options):
    iz = zarr.open(input_zarr_path, mode='r')
    # Read relevant interval from input zarr
    data = iz[block[0,0]:block[0,1], block[1,0]:block[1,1], block[2,0]:block[2,1]]
    oz = zarr.open(output_zarr_path, mode='r+', storage_options=storage_options)
    oz[block[0,0]:block[0,1], block[1,0]:block[1,1], block[2,0]:block[2,1]] = data


def _thread_worker(data, output_zarr_array, block):
    output_zarr_array[block[0,0]:block[0,1], block[1,0]:block[1,1], block[2,0]:block[2,1]] = \
        data[block[0,0]:block[0,1], block[1,0]:block[1,1], block[2,0]:block[2,1]]


def _thread_reader_worker(inzarr, out_array, block):
    out_array[block[0,0]:block[0,1], block[1,0]:block[1,1], block[2,0]:block[2,1]] = \
        inzarr[block[0,0]:block[0,1], block[1,0]:block[1,1], block[2,0]:block[2,1]]


def read_threading(input_zarr_path, block_shape=None, num_workers=1):
    blosc.use_threads = False
    
    inzarr = zarr.open(input_zarr_path, 'r')
    if block_shape is None:
        block_shape = inzarr.chunks
    blocks = make_blocks(inzarr.shape, block_shape)
    data = np.empty(shape=inzarr.shape, dtype=inzarr.dtype)

    argslist = list(
        zip(
            itertools.repeat(inzarr),
            itertools.repeat(data),
            blocks
        )
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_thread_reader_worker, *args) for args in argslist]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                logging.error(exc)

    return data


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
    """Write a Zarr array without parallelization. Blosc compression may use multithreading."""
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

    usage_text = ("Usage:" + "  zarr_io_test.py" + " [options]")
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("-i", "--input-file", type=str, default="/allen/scratch/aindtemp/cameron.arshadi/test-file-res1-12stack.zarr")
    parser.add_argument("-o", "--output-dir", type=str, default="/allen/programs/aind/workgroups/msma/test-file.zarr")
    parser.add_argument("-l", "--log-level", type=str, default=logging.INFO)
    parser.add_argument("--multiprocessing", default=False, action="store_true", help="use Python multiprocessing")
    parser.add_argument("--multithreading", default=False, action="store_true", help="use Python multithreading")
    parser.add_argument("--dask", default=False, action="store_true", help="use Dask")
    parser.add_argument("--slurm", default=False, action="store_true", help="use SLURM cluster")
    parser.add_argument("-c", "--cores-per-worker", type=int, default=1, help="number of threads per worker. Only applies for --dask and --slurm.")
    parser.add_argument("-p", "--workers", type=int, default=4, help="number of workers. Also applies to --multithreading.")
    parser.add_argument("-m", "--mem", type=str, default="4000M", help="memory limit per worker. Only applies for --dask and --slurm.")
    parser.add_argument("--chunk-shape", type=str, default=None, help="zarr array chunk shape. Do not use with --chunk-size.")
    parser.add_argument("--chunk-size", type=int, default=100E6, help="target chunk size, in bytes. Do not use with --chunk-shape.")
    parser.add_argument("--chunk-mode", type=str, default="cycle", help="chunking mode. Options are 'z' and 'cycle'. Do not use with --chunk-shape.")
    parser.add_argument("--block-shape", type=str, default=None, help="shape of thread fov. If None, will compute from chunk size.")
    parser.add_argument("--credentials", type=str, default=None, help="path to AWS or GCS credentials file")
    parser.add_argument("--monitor", default=False, action="store_true", help="start the Dask dashboard at :8787. Only applies for --dask and --slurm")
    parser.add_argument("--walltime", type=str, default="02:00:00", help="SLURM cluster wall time (HH:MM:SS)")

    args = parser.parse_args(sys.argv[1:])

    credentials_file = args.credentials

    logging.info(args)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(args.log_level)

    # force compatible path syntax with s3 and gcs
    # I'm not re-using the same file here so that we can
    # check if the arrays are all equal later.
    output_zarr_file1 = args.output_dir + "/test-file1.zarr"
    output_zarr_file2 = args.output_dir + "/test-file2.zarr"
    output_zarr_file3 = args.output_dir + "/test-file3.zarr"
    output_zarr_file4 = args.output_dir + "/test-file4.zarr"
    output_zarr_file5 = args.output_dir + "/test-file5.zarr"

    logging.info(f"num cpus: {multiprocessing.cpu_count()}")

    my_slurm_kwargs = {}

    if args.monitor:
        dashboard_address = ":8787"
        my_slurm_kwargs['scheduler_options'] = {"dashboard_address": dashboard_address}

    client = None
    if args.slurm:
        from dask_jobqueue import SLURMCluster
        logging.info(f"Using SLURM cluster with {args.workers} workers and {args.cores_per_worker} threads per worker")
        cluster = SLURMCluster(cores=args.cores_per_worker, memory=args.mem, queue="aind", walltime=args.walltime, **my_slurm_kwargs)
        cluster.scale(args.workers)
        logging.info(cluster.job_script())
        client = Client(cluster)
    elif args.dask:
        logging.info("Using local cluster")
        client = Client(n_workers=args.workers, threads_per_worker=args.cores_per_worker, memory_limit=args.mem)

    logging.info(client)

    z = zarr.open(args.input_file, 'r')
    data = z[:]
    logging.info(f"data shape {data.shape}, data size {data.nbytes / 2**20} MiB")

    # Zarr array chunk shape
    if args.chunk_shape is None:
        target_size = args.chunk_size
        chunk_shape = guess_chunks(data.shape, target_size, data.itemsize, args.chunk_mode)
    else:
        chunk_shape = tuple(ast.literal_eval(args.chunk_shape))
    logging.info(f"chunk shape {chunk_shape}, chunk_size {np.product(chunk_shape) * data.itemsize / 2**20} MiB")

    # Worker region shape
    if args.block_shape is None:
        block_shape = guess_blocks(data.shape, chunk_shape, args.workers, args.chunk_mode)
    else:
        block_shape = tuple(ast.literal_eval(args.block_shape))
    logging.info(f"block shape {block_shape}, block_size {np.product(block_shape) * data.itemsize / 2**20} MiB")

    compressor = blosc.Blosc(cname='zstd', clevel=1, shuffle=blosc.Blosc.SHUFFLE)

    # Create bboxes for workers
    block_list = make_blocks(data.shape, block_shape)
    logging.info(f"num blocks {len(block_list)}")

    # Start write tests
    start = timer()
    default_result = write_default(data, output_zarr_file3, compressor, None, chunk_shape, args.workers,
                                   credentials_file=credentials_file)
    end = timer()
    logging.info(f"default write time: {end - start}, compress MiB/s {default_result.nbytes / 2**20 / (end-start)}")

    if args.dask or args.slurm:
        start = timer()
        dask_chunked_result = write_chunked_dask(args.input_file, output_zarr_file1, data.shape, chunk_shape, block_list,
                                                 compressor, filters=None, client=client, credentials_file=credentials_file)
        end = timer()
        logging.info(f"dask chunked write time: {end - start}, compress MiB/s {dask_chunked_result.nbytes / 2 ** 20 / (end - start)}")

        start = timer()
        dask_full_result = write_dask(data, output_zarr_file2, compressor, None, chunk_shape, client, credentials_file=credentials_file)
        end = timer()
        logging.info(f"dask full write time: {end - start}, compress MiB/s {dask_full_result.nbytes / 2**20 / (end-start)}")

        client.close()

    if args.multiprocessing:
        start = timer()
        multiprocessing_result = write_multiprocessing(args.input_file, output_zarr_file4, data.shape,
                                                       chunk_shape, block_list, compressor, filters=None,
                                                       num_workers=args.workers, credentials_file=credentials_file)
        end = timer()
        logging.info(f"multiprocessing write time: {end - start}, compress MiB/s "
                     f"{multiprocessing_result.nbytes / 2**20 / (end-start)}")

    if args.multithreading:
        start = timer()
        multithreading_result = write_threading(data, output_zarr_file5, chunk_shape, block_list, compressor,
                                                filters=None, num_workers=args.workers, credentials_file=credentials_file)
        end = timer()
        logging.info(f"threading write time: {end - start}, compress MiB/s "
                     f"{multithreading_result.nbytes / 2**20 / (end-start)}")

    # all_equal = np.array_equal(dask_chunked_result[:], default_result[:])
    # # all_equal &= np.array_equal(dask_full_result[:], default_result[:])
    # if args.multiprocessing:
    #     all_equal &= np.array_equal(default_result[:], multiprocessing_result[:])
    # if args.multithreading:
    #     all_equal &= np.array_equal(default_result[:], multithreading_result[:])
    #logging.info(f"All equal: {all_equal}")

    logging.info(default_result.info)


if __name__ == "__main__":
    main()
