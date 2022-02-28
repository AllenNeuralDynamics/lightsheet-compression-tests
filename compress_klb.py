import argparse
import itertools
import logging
import os
import random
import sys
from timeit import default_timer as timer

import pandas as pd
import psutil
import pyklb

import compress_zarr


def build_compressors(threads):
    # We only have 2 choices
    codecs = ["bzip2", "zlib"]
    opts = []
    for c, t in itertools.product(codecs, threads):
        opts.append({
            "name": c,
            "threads": t
        })
    return opts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-tiles", type=int, default=1)
    parser.add_argument("-r", "--resolution", type=str, default="1")
    parser.add_argument("-s", "--random-seed", type=int, default=None)
    parser.add_argument("-i","--input-file", type=str, default="/allen/scratch/aindtemp/data/anatomy/2020-12-01-training-data/2020-12-01-stack-15/images/BrainSlice1_MMStack_Pos33_15_shift.tif")
    parser.add_argument("-d","--output-data-file", type=str, default="/allen/scratch/aindtemp/cameron.arshadi/test_file.klb")
    parser.add_argument("-o", "--output-metrics-file", type=str, default="/allen/scratch/aindtemp/cameron.arshadi/klb-compression-metrics.csv")
    parser.add_argument("-l", "--log-level", type=str, default=logging.INFO)
    parser.add_argument("-t", "--threads", type=int, nargs="+", default=[1])

    args = parser.parse_args(sys.argv[1:])

    print(args)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")
    logging.getLogger().setLevel(args.log_level)

    compressors = build_compressors(args.threads)

    run(compressors=compressors,
        input_file=args.input_file,
        num_tiles=args.num_tiles,
        resolution=args.resolution,
        random_seed=args.random_seed,
        output_data_file=args.output_data_file,
        output_metrics_file=args.output_metrics_file)


def run(compressors, input_file, num_tiles, resolution, random_seed, output_data_file, output_metrics_file):
    if random_seed is not None:
        random.seed(random_seed)

    total_tests = num_tiles * len(compressors)

    all_metrics = []

    for ti in range(num_tiles):
        data, rslice, read_dur = compress_zarr.read_random_chunk(input_file, resolution)

        for c in compressors:
            logging.info(f"starting test {len(all_metrics) + 1}/{total_tests}")
            logging.info(f"compressor: {c['name']}")

            psutil.cpu_percent(interval=None)
            start = timer()
            pyklb.writefull(data, output_data_file, compression=c['name'], numthreads=c['threads'])
            cpu_utilization = psutil.cpu_percent(interval=None)
            end = timer()
            compress_dur = end - start
            # TODO: check if this makes sense
            bytes_written = os.path.getsize(output_data_file)

            tile_metrics = {
                'compressor_name': c['name'],
                'tile': rslice,
                'threads': c['threads'],
                'bytes_read': data.nbytes,
                'read_time': read_dur,
                'read_bps': data.nbytes / read_dur,
                'compress_bps': bytes_written / compress_dur,
                'compress_time': compress_dur,
                'bytes_written': bytes_written,
                'shape': data.shape,
                'cpu_utilization': cpu_utilization,
                'storage_ratio': data.nbytes / bytes_written
            }

            all_metrics.append(tile_metrics)

    output_metrics_file = output_metrics_file.replace('.csv', '_' + os.path.basename(input_file) + '.csv')

    df = pd.DataFrame.from_records(all_metrics)
    df.to_csv(output_metrics_file, index_label='test_number')


if __name__ == "__main__":
    main()
