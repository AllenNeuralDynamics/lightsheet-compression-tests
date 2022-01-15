import h5py
import random
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

def trunc_filter(bits):
    scale = 1.0 / (2 ** bits)
    return [] if bits == 0 else [ numcodecs.fixedscaleoffset.FixedScaleOffset(offset=0, scale=scale, dtype=np.uint16) ]

def blosc_compressor_lib(trunc_bits):
    cnames = [ 'zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib' ]#, 'snappy' ]
    shuffles = [ numcodecs.Blosc.SHUFFLE, numcodecs.Blosc.NOSHUFFLE ]
    clevels = [ 1, 3, 5, 9 ]

    opts = []
    for cname, clevel, shuffle, tb in itertools.product(cnames, clevels, shuffles, trunc_bits):
        opts.append({
            'name': f'blosc-{cname}',
            'compressor': numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle),
            'filters': trunc_filter(tb),            
            'params': {
                'shuffle': shuffle,
                'level': clevel,
                'trunc': tb,
            }
        })

    return opts

def lossless_compressor_lib(trunc_bits):
    clevels = [ 1, 3, 5, 9 ]

    opts = []
    for clevel,tb in itertools.product(clevels, trunc_bits):
        opts.append({
            'name': 'zlib',
            'compressor': numcodecs.zlib.Zlib(level=clevel),
            'filters': trunc_filter(tb),
            'params': {
                'level': clevel,
                'trunc': tb
            }
        })

        opts.append({
            'name': 'gzip',
            'compressor': numcodecs.gzip.GZip(level=clevel),
            'filters': trunc_filter(tb),
            'params': {
                'level': clevel,
                'trunc': tb
            }
        })

        opts.append({
            'name': 'bz2',
            'compressor': numcodecs.bz2.BZ2(level=clevel),
            'filters': trunc_filter(tb),
            'params': {
                'level': clevel,
                'trunc': tb
            }
        })

        opts.append({
            'name': 'lzma',
            'compressor': numcodecs.lzma.LZMA(preset=clevel),
            'filters': trunc_filter(tb),
            'params': {
                'level': clevel,
                'trunc': tb
            }
        })

    return opts

def lossy_compressor_lib(trunc_bits):
    import zfpy
    tols = [ 0, 2**4, 2**8, 2**16  ]
    rates = [ 1.0, 0.8, 0.5 ] # maxbits / 4^d
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
            'level': 0            
        }
    } for t,tb in itertools.product(tols,trunc_bits)]

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
        }
    } for r,tb in itertools.product(rates, trunc_bits)]

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
        }
    } for p,tb in itertools.product(precisions, trunc_bits)]

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
    parser.add_argument("-n","--num-tiles", type=int, default=1)
    parser.add_argument("-r","--resolution", type=str, default="1")
    parser.add_argument("-s","--random-seed", type=int, default=None)
    parser.add_argument("-i","--input-file", type=str, default="/allen/scratch/aindtemp/data/anatomy/exm-hemi-brain/data.h5")
    parser.add_argument("-d","--output-data-file", type=str, default="/allen/scratch/aindtemp/david.feng/test_file.zarr")
    parser.add_argument("-o","--output-metrics-file", type=str, default="./compression_metrics.csv")
    parser.add_argument("-l","--log-level", type=str, default=logging.INFO)
    parser.add_argument("-c","--codecs", nargs="+", type=str, default=["blosc"])
    parser.add_argument("-t","--trunc-bits", nargs="+", type=int, default=[0,2,4])

    args = parser.parse_args(sys.argv[1:])

    print(args)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(args.log_level)
    
    compressors = build_compressors(args.codecs, args.trunc_bits)

    _, file_extension = os.path.splitext(args.input_file)

    if file_extension == ".h5" or file_extension == ".zarr":
        run(compressors=compressors,
            num_tiles=args.num_tiles,
            resolution=args.resolution,
            random_seed=args.random_seed,
            input_file=args.input_file,
            output_data_file=args.output_data_file,
            output_metrics_file=args.output_metrics_file,
            file_format=file_extension)
    elif file_extension == '.tif':
        # Assume full resolution for now
        run_tiff(compressors=compressors,
                 num_tiles=args.num_tiles,
                 random_seed=args.random_seed,
                 input_file=args.input_file,
                 output_data_file=args.output_data_file,
                 output_metrics_file=args.output_metrics_file)
    else:
        raise ValueError("Unrecognized input file extension: " + file_extension)

def read_compress_write(dataset, key, compressor, filters, output_path):
    logging.info(f"loading {key}")
    
    start = timer()
    data = dataset[key][()]
    end = timer()
    read_dur = end - start
    logging.info(f"loaded {data.shape}, {data.nbytes} bytes, time {read_dur}s")

    out = compress_write(data, compressor, filters, output_path)
    out['read_time'] = read_dur
    return out

def read_compress_write_tiff(tiff_path, key, compressor, filters, output_path):
    logging.info(f"loading slice {key}")

    start = timer()
    # FIXME: Do we only want to compress one slice at a time?
    data = tifffile.imread(tiff_path, key=key)
    end = timer()
    read_dur = end - start
    logging.info(f"loaded {data.shape}, {data.nbytes} bytes, time {read_dur}s")

    out = compress_write(data, compressor, filters, output_path)
    out['read_time'] = read_dur
    return out

def compress_write(data, compressor, filters, output_path):
    psutil.cpu_percent(interval=None)
    start = timer()
    ds = zarr.DirectoryStore(output_path)
    za = zarr.array(data, chunks=True, filters=filters, compressor=compressor, store=ds, overwrite=True)
    logging.info(str(za.info))
    cpu_utilization = psutil.cpu_percent(interval=None)
    end = timer()
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
        'cpu_utilization': cpu_utilization
        #'write_time': write_dur
    }

    return out


def run(compressors, num_tiles, resolution, random_seed, input_file, output_data_file, output_metrics_file, file_format):

    if random_seed is not None:
        random.seed(random_seed)

    all_metrics = []

    if file_format == '.h5':
        f = h5py.File(input_file, 'r')
    elif file_format == '.zarr':
        f = zarr.open(input_file, 'r')
    else:
        raise ValueError("Unsupported input file format: " + file_format)

    ds = f["t00000"]

    total_tests = num_tiles * len(compressors)

    for ti in range(num_tiles):
        rslice = random.choice(list(ds.keys()))

        for c in compressors:
            compressor = c['compressor']
            filters = c['filters']

            tile_metrics = {
                'compressor_name': c['name'],
                'tile': rslice
            }

            tile_metrics.update(c['params'])

            logging.info(f"starting test {len(all_metrics)+1}/{total_tests}")
            logging.info(f"compressor: {c['name']} params: {c['params']}")

            key = f"{rslice}/{resolution}/cells"
            data = read_compress_write(ds, key, compressor, filters, output_data_file)

            tile_metrics['read_time'] = data['read_time']
            tile_metrics['bytes_read'] = data['bytes_read']
            tile_metrics['shape'] = data['shape']
            tile_metrics['read_bps'] = data['bytes_read'] / data['read_time']
            tile_metrics['compress_time'] = data['compress_time']
            tile_metrics['bytes_written'] = data['bytes_written']
            tile_metrics['compress_bps'] = data['bytes_written'] / data['compress_time']
            tile_metrics['storage_ratio'] = data['bytes_read'] / data['bytes_written']
            tile_metrics['cpu_utilization'] = data['cpu_utilization']
            #tile_metrics['write_time'] = data['write_time']
            #tile_metrics['write_bps'] = data['write_time'] / data['bytes_written']

            all_metrics.append(tile_metrics)

    df = pd.DataFrame.from_records(all_metrics)
    output_metrics_file = output_metrics_file.replace('.csv', "_" + file_format.replace('.', '') + ".csv")
    df.to_csv(output_metrics_file, index_label='test_number')

    if file_format == '.h5':
        # zarr datasets need not be closed?
        f.close()

def run_tiff(compressors, num_tiles, random_seed, input_file, output_data_file, output_metrics_file):
    if random_seed is not None:
        random.seed(random_seed)

    all_metrics = []

    total_tests = num_tiles * len(compressors)

    # Just read metadata, don't load image into memory
    tiff = tifffile.TiffFile(input_file)
    num_slices = len(tiff.pages)
    tifffile.TiffFile.close(tiff)  # FIXME: is this correct?

    for ti in range(num_tiles):
        rslice = random.randrange(num_slices)

        for c in compressors:
            compressor = c['compressor']
            filters = c['filters']

            tile_metrics = {
                'compressor_name': c['name'],
                'tile': rslice
            }

            tile_metrics.update(c['params'])

            logging.info(f"starting test {len(all_metrics)+1}/{total_tests}")
            logging.info(f"compressor: {c['name']} params: {c['params']}")

            data = read_compress_write_tiff(input_file, rslice, compressor, filters, output_data_file)

            tile_metrics['read_time'] = data['read_time']
            tile_metrics['bytes_read'] = data['bytes_read']
            tile_metrics['shape'] = data['shape']
            tile_metrics['read_bps'] = data['bytes_read'] / data['read_time']
            tile_metrics['compress_time'] = data['compress_time']
            tile_metrics['bytes_written'] = data['bytes_written']
            tile_metrics['compress_bps'] = data['bytes_written'] / data['compress_time']
            tile_metrics['storage_ratio'] = data['bytes_read'] / data['bytes_written']
            tile_metrics['cpu_utilization'] = data['cpu_utilization']
            # tile_metrics['write_time'] = data['write_time']
            # tile_metrics['write_bps'] = data['write_time'] / data['bytes_written']

            all_metrics.append(tile_metrics)

    df = pd.DataFrame.from_records(all_metrics)
    output_metrics_file = output_metrics_file.replace('.csv', "_tiff.csv")
    df.to_csv(output_metrics_file, index_label='test_number')

if __name__ == "__main__": 
    main()

