import h5py
import random
import zarr
import numcodecs
import argparse
import sys
import pandas as pd
import itertools
import numpy as np
import logging

from timeit import default_timer as timer

def compressor_lib():    
    cnames = [ 'zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib' ]#, 'snappy' ]
    shuffles = [ numcodecs.Blosc.SHUFFLE, numcodecs.Blosc.NOSHUFFLE ]
    clevels = [ 0,1,2,5,9 ]
    qs = [ 0, 2, 4 ]

    opts = []
    for cname, clevel, shuffle, q in itertools.product(cnames, clevels, shuffles, qs):
        opts.append({
            'name': f'blosc-{cname}',
            'level': clevel,
            'shuffle': shuffle,
            'quant':q,
            'compressor': numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle),
            'filters': None if q == 0 else [ numcodecs.fixedscaleoffset.FixedScaleOffset(offset=0, scale=1/10**q, dtype=np.uint16) ]
        })

    return opts
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--num-tiles", type=int, default=1)
    parser.add_argument("-r","--resolution", type=str, default="1")
    parser.add_argument("-s","--random-seed", type=int, default=None)
    parser.add_argument("-i","--input-file", type=str, default="/allen/scratch/aindtemp/data/anatomy/exm-hemi-brain/data.h5")
    parser.add_argument("-d","--output-data-file", type=str, default="/allen/scratch/aindtemp/david.feng/test_file.zarr")
    parser.add_argument("-o","--output-metrics-file", type=str, default="./compression_metrics.csv")
    parser.add_argument("-l","--log-level", type=str, default=logging.INFO)

    args = parser.parse_args(sys.argv[1:])

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(args.log_level)

    run(num_tiles=args.num_tiles, 
        resolution=args.resolution, 
        random_seed=args.random_seed, 
        input_file=args.input_file, 
        output_data_file=args.output_data_file, 
        output_metrics_file=args.output_metrics_file)

def read_compress_write(dataset, key, compressor, filters, store):
    logging.info(f"loading {key}")
    
    start = timer()
    data = dataset[key][()]
    end = timer()
    read_dur = end - start
    logging.info(f"loaded {data.shape}, {data.nbytes} bytes, time {read_dur}s")

    start = timer()
    za = zarr.array(data, chunks=True, filters=filters, compressor=compressor, store=store, overwrite=True)
    end = timer()
    compress_dur = end - start
    logging.info(f"compression time = {compress_dur}, bps = {data.nbytes / compress_dur}, ratio = {za.nbytes/za.nbytes_stored}")

    return {
        'read_time': read_dur,
        'bytes_read': za.nbytes,
        'compress_time': compress_dur,
        'bytes_written': za.nbytes_stored,
        'shape': data.shape
    }


def run(num_tiles, resolution, random_seed, input_file, output_data_file, output_metrics_file):

    if random_seed is not None:
        random.seed(random_seed)


    all_metrics = []

    compressors = compressor_lib()

    with h5py.File(input_file) as f:
        ds = f["t00000"]
        
        total_tests = num_tiles * len(compressors)

        for ti in range(num_tiles):
            rslice = random.choice(list(ds.keys()))
            
            for c in compressors:
                compressor = c['compressor']
                filters = c['filters']

                tile_metrics = {
                    'compressor': c['name'],
                    'level': c['level'],
                    'shuffle': c['shuffle'],
                    'quant': c['quant'],
                    'tile': rslice
                }
                
                logging.info(f"starting test {len(all_metrics)+1}/{total_tests}")
                logging.info(f"compressor: {c['name']} level={c['level']} shuffle={c['shuffle']} quant={c['quant']}")

                key = f"{rslice}/{resolution}/cells"
                store = zarr.storage.DirectoryStore(output_data_file)
                data = read_compress_write(ds, key, compressor, filters, store)

                tile_metrics['read_time'] = data['read_time']
                tile_metrics['bytes_read'] = data['bytes_read']
                tile_metrics['shape'] = data['shape']
                tile_metrics['read_bps'] = data['bytes_read'] / data['read_time']
                tile_metrics['compress_time'] = data['compress_time']
                tile_metrics['bytes_written'] = data['bytes_written']
                tile_metrics['compress_bps'] = data['bytes_written'] / data['compress_time']
                tile_metrics['storage_ratio'] = data['bytes_read'] / data['bytes_written']

                all_metrics.append(tile_metrics)

        df = pd.DataFrame.from_records(all_metrics)
        df.to_csv(output_metrics_file, index_label='test_number')

if __name__ == "__main__": 
    main()


