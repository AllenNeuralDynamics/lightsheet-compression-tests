import argparse
import dask
import dask.array as da
import h5py
import hdf5plugin
import itertools
import logging
import math
import numcodecs
import numpy as np
import os
import pandas as pd
import psutil
import random
import sys
import tifffile
import zarr
from aind_data_transfer.util.io_utils import DataReaderFactory
from distributed import Client, LocalCluster, wait
from timeit import default_timer as timer
from numcodecs import blosc
blosc.use_threads = False

os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"


def trunc_filter(bits):
    scale = 1.0 / (2 ** bits)
    return [] if bits == 0 else [numcodecs.fixedscaleoffset.FixedScaleOffset(offset=0, scale=scale, dtype=np.uint16)]


def blosc_compressor_lib(trunc_bits):
    cnames = ['zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib']  # , 'snappy' ]
    shuffles = [numcodecs.Blosc.SHUFFLE, numcodecs.Blosc.NOSHUFFLE]
    clevels = [1, 3, 5, 7, 9]

    opts = []
    for cname, clevel, shuffle, tb in itertools.product(cnames, clevels, shuffles, trunc_bits):
        opts.append(
            {
                'name': f'blosc-{cname}',
                'compressor': numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle),
                'filters': trunc_filter(tb),
                'params': {
                    'shuffle': shuffle,
                    'level': clevel,
                    'trunc': tb,
                }
            }
        )

    return opts


def lossless_compressor_lib(trunc_bits):
    clevels = [1, 3, 5, 7, 9]

    opts = []
    for clevel, tb in itertools.product(clevels, trunc_bits):
        opts.append(
            {
                'name': 'lossless-zlib',
                'compressor': numcodecs.zlib.Zlib(level=clevel),
                'filters': trunc_filter(tb),
                'params': {
                    'level': clevel,
                    'trunc': tb,
                }
            }
        )

        opts.append(
            {
                'name': 'lossless-gzip',
                'compressor': numcodecs.gzip.GZip(level=clevel),
                'filters': trunc_filter(tb),
                'params': {
                    'level': clevel,
                    'trunc': tb,
                }
            }
        )

        opts.append(
            {
                'name': 'lossless-bz2',
                'compressor': numcodecs.bz2.BZ2(level=clevel),
                'filters': trunc_filter(tb),
                'params': {
                    'level': clevel,
                    'trunc': tb,
                }
            }
        )

        opts.append(
            {
                'name': 'lossless-lzma',
                'compressor': numcodecs.lzma.LZMA(preset=clevel),
                'filters': trunc_filter(tb),
                'params': {
                    'level': clevel,
                    'trunc': tb,
                }
            }
        )

    return opts


def lossy_compressor_lib(trunc_bits):
    import zfpy
    tols = [0, 2 ** 0, 2 ** 1, 2 ** 2, 2 ** 4, 2 ** 8, 2 ** 16]
    rates = [4, 6, 8, 10, 12, 14, 16]  # maxbits / 4^d
    precisions = [16, 14, 12]  # number of bit planes encoded for transform coefficients

    cast_filter = [numcodecs.astype.AsType(encode_dtype=np.float32, decode_dtype=np.uint16)]

    compressors = []

    compressors += [{
        'name': 'zfpy-fixed-accuracy',
        'compressor': numcodecs.zfpy.ZFPY(mode=zfpy.mode_fixed_accuracy, tolerance=t),
        'filters': trunc_filter(tb) + cast_filter,
        'params': {
            'tolerance': t,
            'rate': None,
            'precision': None,
            'trunc': tb,
            'level': 0,
        }
    } for t, tb in itertools.product(tols, trunc_bits)]

    compressors += [{
        'name': 'zfpy-fixed-rate',
        'compressor': numcodecs.zfpy.ZFPY(mode=zfpy.mode_fixed_rate, rate=r),
        'filters': trunc_filter(tb) + cast_filter,
        'params': {
            'tolerance': None,
            'rate': r,
            'precision': None,
            'trunc': tb,
            'level': 0,
        }
    } for r, tb in itertools.product(rates, trunc_bits)]

    compressors += [{
        'name': 'zfpy-fixed-precision',
        'compressor': numcodecs.zfpy.ZFPY(mode=zfpy.mode_fixed_precision, precision=p),
        'filters': trunc_filter(tb) + cast_filter,
        'params': {
            'tolerance': None,
            'rate': None,
            'precision': p,
            'trunc': tb,
            'level': 0,
        }
    } for p, tb in itertools.product(precisions, trunc_bits)]

    return compressors


def build_compressors(codecs, trunc_bits):
    compressors = []
    if 'other-lossless' in codecs:
        compressors += lossless_compressor_lib(trunc_bits)
    if 'blosc' in codecs:
        compressors += blosc_compressor_lib(trunc_bits)
    if 'lossy' in codecs:
        compressors += lossy_compressor_lib(trunc_bits)

    return compressors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-tiles", type=int, default=20)
    parser.add_argument("-r", "--resolution", type=str, default="0")
    parser.add_argument("-s", "--random-seed", type=int, default=0)
    parser.add_argument(
        "-i", "--input-file", type=str,
        default="/mnt/vast/aind/exaSPIM/exaSPIM_609281_2022-11-03_13-49-18/exaSPIM/tile_x_0015_y_0000_z_0000_ch_488.ims"
    )
    parser.add_argument(
        "-d", "--output-data-file", type=str, default="gs://aind-data-dev/cameron.arshadi/test-file.zarr"
    )
    parser.add_argument("-o", "--output-metrics-file", type=str, default="./compression_metrics.csv")
    parser.add_argument("-l", "--log-level", type=str, default=logging.INFO)
    parser.add_argument("-c", "--codecs", nargs="+", type=str, default=["blosc"])
    parser.add_argument("-t", "--trunc-bits", nargs="+", type=int, default=[0])
    parser.add_argument("-m", "--metrics", nargs="+", type=str, default=[])  # [mse, ssim, psnr]
    parser.add_argument("-p", "--parallel", default=True, action="store_true")
    parser.add_argument("--threads", type=int, default=1)

    args = parser.parse_args(sys.argv[1:])

    logging.info(args)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(args.log_level)

    client = Client(LocalCluster(processes=False))

    compressors = build_compressors(args.codecs, args.trunc_bits)
    logging.info(compressors)

    run(
        compressors=compressors,
        num_tiles=args.num_tiles,
        resolution=args.resolution,
        random_seed=args.random_seed,
        input_file=args.input_file,
        output_data_file=args.output_data_file,
        quality_metrics=args.metrics,
        output_metrics_file=args.output_metrics_file,
        use_parallel=args.parallel,
        threads=args.threads
    )


def get_block(ds, origin=None):
    shape = np.array([s for s in ds.shape], dtype=int)
    chunks = np.array([c for c in ds.chunksize], dtype=int)
    blocks = np.array([chunks[0] * 2, chunks[1] * 4, chunks[2] * 4], dtype=int)
    if origin is None:
        origin = np.zeros(3, dtype=int)
        # keep blocks aligned with chunks
        # don't allow truncated dimensions
        origin[0] = random.choice([i for i in range(0, shape[0], blocks[0])][:-1])
        origin[1] = random.choice([i for i in range(0, shape[1], blocks[1])][:-1])
        origin[2] = random.choice([i for i in range(0, shape[2], blocks[2])][:-1])
    block = ds[
            origin[0]:origin[0] + blocks[0],
            origin[1]:origin[1] + blocks[1],
            origin[2]:origin[2] + blocks[2]
            ]
    return block, origin


def read_block(input, origin=None):
    reader = DataReaderFactory().create(input)
    logging.info(f"original chunks: {reader.get_chunks()}")
    chunks = [512, 256, 256]
    logging.info(f"new chunks: {chunks}")
    block, o = get_block(reader.as_dask_array(chunks=chunks), origin)
    logging.info(f"origin: {o}")
    t0 = timer()
    block = block.compute()
    read_dur = timer() - t0
    return dask.array.from_array(block, chunks=chunks), o, read_dur


def get_size(shape, bytes_per_pixel):
    """Array size in MiB"""
    return (np.product(shape) * bytes_per_pixel) / (1024. * 1024)


def compress_write(
        data, compressor, filters, quality_metrics, output_path, use_parallel=False, threads=1
):
    psutil.cpu_percent(interval=None)
    start = timer()
    if use_parallel:
        data.to_zarr(
            url=output_path,
            overwrite=True,
            compressor=compressor,
            return_stored=False,
            compute=True,
            dimension_separator='/'
        )
    else:
        blosc.use_threads = True
        blosc.set_nthreads(threads)
        za = zarr.open(
            output_path,
            'w',
            shape=data.shape,
            chunks=data.chunks,
            dtype=data.dtype,
            compressor=compressor,
            dimension_separator="/"
        )
        za[:] = data

    end = timer()
    cpu_utilization = psutil.cpu_percent(interval=None)
    compress_dur = end - start
    za = zarr.open(output_path, 'r')
    logging.info(za)
    logging.info(
        f"compression time = {compress_dur}, "
        f"bps = {data.nbytes / compress_dur / (1024 ** 2)}, "
        f"ratio = {za.nbytes / za.nbytes_stored}, "
        f"cpu = {cpu_utilization}%"
    )

    out = {
        'bytes_read': za.nbytes,
        'compress_time': compress_dur,
        'bytes_written': za.nbytes_stored,
        'shape': data.shape,
        'chunk_shape': data.chunksize,
        'chunk_size': get_size(data.chunksize, 2),
    }

    if quality_metrics:
        metrics = eval_quality(data, za[:], quality_metrics)
        out.update(metrics)

    return out


def eval_quality(input_data, decoded_data, quality_metrics):
    import skimage.metrics as metrics
    qa = {}
    if 'mse' in quality_metrics:
        qa['mse'] = metrics.mean_squared_error(input_data, decoded_data)
    if 'ssim' in quality_metrics:
        qa['ssim'] = metrics.structural_similarity(
            input_data, decoded_data,
            data_range=decoded_data.max() - decoded_data.min()
        )
    if 'psnr' in quality_metrics:
        qa['psnr'] = metrics.peak_signal_noise_ratio(
            input_data, decoded_data,
            data_range=decoded_data.max() - decoded_data.min()
        )
    return qa


def run(
        compressors, num_tiles, resolution, random_seed, input_file, output_data_file, quality_metrics,
        output_metrics_file, use_parallel=False, threads=1
):
    if random_seed is not None:
        random.seed(random_seed)

    all_metrics = []

    total_tests = num_tiles * len(compressors)

    for i in range(num_tiles):
        # only randomize the location of the first read per tile
        # we still want to read the same tile multiple times to measure I/O
        origin = None
        for c in compressors:
            data, origin, read_time = read_block(input_file, origin=origin)
            logging.info(f"read time: {read_time}, bps={data.nbytes / read_time / (1024 ** 2)}")

            compressor = c['compressor']
            filters = c['filters']

            tile_metrics = {
                'compressor_name': c['name'],
            }

            tile_metrics.update(c['params'])

            logging.info(f"starting test {len(all_metrics) + 1}/{total_tests}")
            logging.info(f"compressor: {c['name']} params: {c['params']}")

            metrics = compress_write(
                data, compressor, filters, quality_metrics, output_data_file, use_parallel, threads
            )

            tile_metrics.update(metrics)
            tile_metrics['read_time'] = read_time
            tile_metrics['read_bps'] = metrics['bytes_read'] / read_time
            tile_metrics['compress_bps'] = metrics['bytes_read'] / metrics['compress_time']
            tile_metrics['storage_ratio'] = metrics['bytes_read'] / metrics['bytes_written']

            all_metrics.append(tile_metrics)

            df = pd.DataFrame.from_records(all_metrics)
            df.to_csv(output_metrics_file, index_label='test_number')


if __name__ == "__main__":
    main()
