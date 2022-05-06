import h5py
import random
import math
import zarr
import numcodecs
import argparse
import os
import sys
import pandas as pd
import itertools
import numpy as np
import logging
import psutil
import tifffile

from timeit import default_timer as timer

import zarr_io_test


def trunc_filter(bits):
    scale = 1.0 / (2 ** bits)
    return [] if bits == 0 else [ numcodecs.fixedscaleoffset.FixedScaleOffset(offset=0, scale=scale, dtype=np.uint16) ]

def blosc_compressor_lib(trunc_bits, chunk_factor):
    cnames = [ 'zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib' ]#, 'snappy' ]
    shuffles = [ numcodecs.Blosc.SHUFFLE, numcodecs.Blosc.NOSHUFFLE ]
    clevels = [ 1, 3, 5, 7, 9 ]

    opts = []
    for cname, clevel, shuffle, tb, cf in itertools.product(cnames, clevels, shuffles, trunc_bits, chunk_factor):
        opts.append({
            'name': f'blosc-{cname}',
            'compressor': numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle),
            'filters': trunc_filter(tb),            
            'params': {
                'shuffle': shuffle,
                'level': clevel,
                'trunc': tb,
                "chunk_factor": cf
            }
        })

    return opts

def lossless_compressor_lib(trunc_bits, chunk_factor):
    clevels = [ 1, 3, 5, 7, 9 ]

    opts = []
    for clevel, tb, cf in itertools.product(clevels, trunc_bits, chunk_factor):
        opts.append({
            'name': 'lossless-zlib',
            'compressor': numcodecs.zlib.Zlib(level=clevel),
            'filters': trunc_filter(tb),
            'params': {
                'level': clevel,
                'trunc': tb,
                'chunk_factor': cf
            }
        })

        opts.append({
            'name': 'lossless-gzip',
            'compressor': numcodecs.gzip.GZip(level=clevel),
            'filters': trunc_filter(tb),
            'params': {
                'level': clevel,
                'trunc': tb,
                'chunk_factor': cf
            }
        })

        opts.append({
            'name': 'lossless-bz2',
            'compressor': numcodecs.bz2.BZ2(level=clevel),
            'filters': trunc_filter(tb),
            'params': {
                'level': clevel,
                'trunc': tb,
                'chunk_factor': cf
            }
        })

        opts.append({
            'name': 'lossless-lzma',
            'compressor': numcodecs.lzma.LZMA(preset=clevel),
            'filters': trunc_filter(tb),
            'params': {
                'level': clevel,
                'trunc': tb,
                'chunk_factor': cf
            }
        })

    return opts

def lossy_compressor_lib(trunc_bits, chunk_factor):
    import zfpy
    tols = [ 0, 2**0, 2**1, 2**2, 2**4, 2**8, 2**16 ]
    rates = [ 4, 6, 8, 10, 12, 14, 16 ] # maxbits / 4^d
    precisions = [ 16, 14, 12 ] # number of bit planes encoded for transform coefficients

    cast_filter = [ numcodecs.astype.AsType(encode_dtype=np.float32, decode_dtype=np.uint16) ]

    compressors = []

    compressors += [{ 
        'name': 'zfpy-fixed-accuracy',
        'compressor': numcodecs.zfpy.ZFPY(mode=zfpy.mode_fixed_accuracy, tolerance=t), 
        'filters': trunc_filter(tb)+cast_filter,        
        'params': {
            'tolerance': t,
            'rate': None,
            'precision': None,
            'trunc': tb,
            'level': 0,
            'chunk_factor': cf
        }
    } for t, tb, cf in itertools.product(tols, trunc_bits, chunk_factor)]

    compressors += [{
        'name': 'zfpy-fixed-rate',
        'compressor': numcodecs.zfpy.ZFPY(mode=zfpy.mode_fixed_rate, rate=r),
        'filters': trunc_filter(tb)+cast_filter,        
        'params': {
            'tolerance': None,
            'rate': r,
            'precision': None,
            'trunc': tb,
            'level': 0,
            'chunk_factor': cf
        }
    } for r, tb, cf in itertools.product(rates, trunc_bits, chunk_factor)]

    compressors += [{
        'name': 'zfpy-fixed-precision',
        'compressor': numcodecs.zfpy.ZFPY(mode=zfpy.mode_fixed_precision, precision=p),
        'filters': trunc_filter(tb)+cast_filter,        
        'params': {
            'tolerance': None,
            'rate': None,
            'precision': p,
            'trunc': tb,
            'level': 0,
            'chunk_factor': cf
        }
    } for p, tb, cf in itertools.product(precisions, trunc_bits, chunk_factor)]

    return compressors

def build_compressors(codecs, trunc_bits, chunk_factor):
    compressors = []
    if 'other-lossless' in codecs:
        compressors += lossless_compressor_lib(trunc_bits, chunk_factor)
    if 'blosc' in codecs:
        compressors += blosc_compressor_lib(trunc_bits, chunk_factor)
    if 'lossy' in codecs:
        compressors += lossy_compressor_lib(trunc_bits, chunk_factor)

    return compressors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--num-tiles", type=int, default=1)
    parser.add_argument("-r","--resolution", type=str, default="1")
    parser.add_argument("-s","--random-seed", type=int, default=None)
    parser.add_argument("-i","--input-file", type=str, default="/allen/scratch/aindtemp/data/anatomy/exm-hemi-brain.zarr")
    parser.add_argument("-d","--output-data-file", type=str, default="/allen/scratch/aindtemp/cameron.arshadi/test_file.zarr")
    parser.add_argument("-o","--output-metrics-file", type=str, default="./compression_metrics.csv")
    parser.add_argument("-l","--log-level", type=str, default=logging.INFO)
    parser.add_argument("-c","--codecs", nargs="+", type=str, default=["blosc"])
    parser.add_argument("-t","--trunc-bits", nargs="+", type=int, default=[0,2,4])
    parser.add_argument("-b", "--block-scale-factor", nargs="+", type=int, default=[1])
    parser.add_argument("-m", "--metrics", nargs="+", type=str, default=[])  # [mse, ssim, psnr]
    parser.add_argument("-p", "--parallel", default=True, action="store_true")
    parser.add_argument("--threads", type=int, default=8)

    args = parser.parse_args(sys.argv[1:])

    print(args)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(args.log_level)
    
    compressors = build_compressors(args.codecs, args.trunc_bits, args.block_scale_factor)

    run(compressors=compressors,
        num_tiles=args.num_tiles,
        resolution=args.resolution,
        random_seed=args.random_seed,
        input_file=args.input_file,
        output_data_file=args.output_data_file,
        quality_metrics=args.metrics,
        output_metrics_file=args.output_metrics_file,
        use_parallel=args.parallel,
        threads=args.threads)

def read_dataset_chunk(dataset, key):
    logging.info(f"loading {key}")

    start = timer()
    data = dataset[key][()]
    end = timer()
    read_dur = end - start
    logging.info(f"loaded {data.shape}, {data.nbytes} bytes, time {read_dur}s")

    return data, read_dur

def make_random_key(dataset, resolution):
    rslice = random.choice(list(dataset.keys()))
    key = f"{rslice}/{resolution}/cells"

    return key, rslice

def read_random_chunk(input_file, resolution):
    _, file_format = os.path.splitext(input_file)

    if file_format == '.h5':
        with h5py.File(input_file, mode='r') as f:
            ds = f["t00000"]
            key, rslice = make_random_key(ds, resolution)
            data, read_dur = read_dataset_chunk(ds, key)

    elif file_format == '.zarr':
        f = zarr.open(input_file, mode='r')
        ds = f["t00000"]
        key, rslice = make_random_key(ds, resolution)
        data, read_dur = read_dataset_chunk(ds, key)

    elif file_format == ".tif":
        with tifffile.TiffFile(input_file) as f:
            # This works with the 4 or so Tiffs that I tested
            z = zarr.open(f.aszarr(), 'r')
            rslice = random.randrange(z.shape[0])  # Axis order ZYX
            logging.info(f"loading {rslice}")
            start = timer()
            data = z[rslice][()]
            end = timer()
            read_dur = end - start
            logging.info(f"loaded {data.shape}, {data.nbytes} bytes, time {read_dur}s")
    else:
        raise ValueError("Unsupported input file format: " + file_format)

    return data, rslice, read_dur

def guess_chunk_shape(data, bytes_per_pixel, scale_factor, min_side=8):
    """Use the zarr chunk size heuristic as a starting point, then
    scale each dimension by scale_factor, clamping if necessary.
    Result shape will range between:
    [min_side, min_side, min_side] <= chunk <= [data.shape[0], data.shape[1], data.shape[2]]"""
    from zarr.util import guess_chunks
    chunk_shape = [math.floor(c * scale_factor) for c in guess_chunks(data.shape, bytes_per_pixel)]
    for i in range(len(chunk_shape)):
        if chunk_shape[i] < min_side:
            chunk_shape[i] = min_side
        if chunk_shape[i] > data.shape[i]:
            chunk_shape[i] = data.shape[i]
    chunk_size = estimate_size(chunk_shape, bytes_per_pixel)
    return chunk_shape, chunk_size

def estimate_size(shape, bytes_per_pixel):
    """Array size in MiB"""
    return (np.product(shape) * bytes_per_pixel) / (1024. * 1024)

def compress_write(data, compressor, filters, block_multiplier, quality_metrics, output_path, use_parallel=False, threads=1):
    psutil.cpu_percent(interval=None)
    start = timer()
    if use_parallel:
        #chunk_shape = zarr_parallel_test.guess_chunk_shape(data.shape, threads)
        chunk_shape = (int(math.ceil(data.shape[0] / threads)), data.shape[1], data.shape[2])
        chunk_size = estimate_size(chunk_shape, data.itemsize)
        block_list = zarr_io_test.make_blocks(data.shape, chunk_shape)
        za = zarr_io_test.write_threading(data, output_path, chunk_shape, block_list, compressor, filters, threads)
    else:
        chunk_shape, chunk_size = guess_chunk_shape(data, bytes_per_pixel=data.itemsize, scale_factor=block_multiplier)
        za = zarr_io_test.write_default(data, output_path, compressor, filters, chunk_shape, threads)
    end = timer()
    logging.info(str(za.info))
    cpu_utilization = psutil.cpu_percent(interval=None)
    compress_dur = end - start
    logging.info(f"compression time = {compress_dur}, bps = {data.nbytes / compress_dur}, ratio = {za.nbytes/za.nbytes_stored}, cpu = {cpu_utilization}%")

    #start = timer()
    #zarr.copy_store(za.store, zarr.DirectoryStore(output_path), if_exists='replace')
    #end = timer()
    #write_dur = end - start
    #logging.info(f"write time = {write_dur}, bps = {za.nbytes_stored/write_dur}")

    out = {
        'bytes_read': za.nbytes,
        'compress_time': compress_dur,
        'bytes_written': za.nbytes_stored,
        'shape': data.shape,
        'chunk_shape': chunk_shape,
        'chunk_size' : chunk_size,
        'cpu_utilization': cpu_utilization
        #'write_time': write_dur
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
        qa['ssim'] = metrics.structural_similarity(input_data, decoded_data,
                                                   data_range=decoded_data.max() - decoded_data.min())
    if 'psnr' in quality_metrics:
        qa['psnr'] = metrics.peak_signal_noise_ratio(input_data, decoded_data,
                                                     data_range=decoded_data.max() - decoded_data.min())
    return qa

def run(compressors, num_tiles, resolution, random_seed, input_file, output_data_file, quality_metrics,
        output_metrics_file, use_parallel=False, threads=1):
    if random_seed is not None:
        random.seed(random_seed)

    all_metrics = []

    total_tests = num_tiles * len(compressors)

    for ti in range(num_tiles):
        data, rslice, read_time = read_random_chunk(input_file, resolution)

        for c in compressors:
            compressor = c['compressor']
            filters = c['filters']
            chunk_factor = c['params']['chunk_factor']

            tile_metrics = {
                'compressor_name': c['name'],
                'tile': rslice
            }

            tile_metrics.update(c['params'])

            logging.info(f"starting test {len(all_metrics)+1}/{total_tests}")
            logging.info(f"compressor: {c['name']} params: {c['params']}")

            metrics = compress_write(data, compressor, filters, chunk_factor, quality_metrics, output_data_file, use_parallel, threads)

            tile_metrics.update(metrics)
            tile_metrics['read_time'] = read_time
            tile_metrics['read_bps'] = metrics['bytes_read'] / read_time
            tile_metrics['compress_bps'] = metrics['bytes_read'] / metrics['compress_time']
            tile_metrics['storage_ratio'] = metrics['bytes_read'] / metrics['bytes_written']
            # tile_metrics['write_time'] = data['write_time']
            # tile_metrics['write_bps'] = data['write_time'] / data['bytes_written']

            all_metrics.append(tile_metrics)

    output_metrics_file = output_metrics_file.replace('.csv', '_' + os.path.basename(input_file) + '.csv')

    df = pd.DataFrame.from_records(all_metrics)
    df.to_csv(output_metrics_file, index_label='test_number')

if __name__ == "__main__": 
    main()


