import argparse
import itertools
import logging
import os
import os.path
import random
import sys
from timeit import default_timer as timer

import glymur
import pandas as pd
import psutil

import compress_zarr


def build_compressors(threads):
    cratios = range(1, 21)
    opts = []
    for r, t in itertools.product(cratios, threads):
        opts.append({
            'storage_ratio': r,
            'threads': t
        })
    return opts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-tiles", type=int, default=10)
    parser.add_argument("-r", "--resolution", type=str, default="1")
    parser.add_argument("-s", "--random-seed", type=int, default=42)
    parser.add_argument("-i", "--input-file", type=str,
                        default=r"C:\Users\cameron.arshadi\Downloads\BrainSlice1_MMStack_Pos33_15_shift.tif")
    parser.add_argument("-d", "--output-data-file", type=str, default="./test_file_jp2.jp2")
    parser.add_argument("-o", "--output-metrics-file", type=str, default="./jp2-compression-metrics.csv")
    parser.add_argument("-l", "--log-level", type=str, default=logging.INFO)
    parser.add_argument("-m", "--metrics", nargs="+", type=str, default=['psnr'])  # [mse, ssim, psnr]
    parser.add_argument("-x", "--threads", nargs="+", type=int, default=[8])

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
        output_metrics_file=args.output_metrics_file,
        metrics=args.metrics)


def run(compressors, input_file, num_tiles, resolution, random_seed, output_data_file, output_metrics_file,
        metrics):
    if random_seed is not None:
        random.seed(random_seed)

    total_tests = num_tiles * len(compressors)

    all_metrics = []

    for ti in range(num_tiles):
        data, rslice, read_dur = compress_zarr.read_random_chunk(input_file, resolution)
        print(data.shape)

        for c in compressors:
            logging.info(f"starting test {len(all_metrics) + 1}/{total_tests}")
            # logging.info(f"compressor: {c['name']}")

            psutil.cpu_percent(interval=None)
            start = timer()
            glymur.set_option('lib.num_threads', c['threads'])
            jp2 = glymur.Jp2k(output_data_file, data, cratios=[c['storage_ratio']], numres=1)
            cpu_utilization = psutil.cpu_percent(interval=None)
            end = timer()
            compress_dur = end - start
            # TODO: check if this makes sense
            bytes_written = os.path.getsize(output_data_file)

            tile_metrics = {
                'compressor_name': 'JPEG2000',
                'tile': rslice,
                'bytes_read': data.nbytes,
                'read_time': read_dur,
                'read_bps': data.nbytes / read_dur,
                'compress_bps': bytes_written / compress_dur,
                'compress_time': compress_dur,
                'bytes_written': bytes_written,
                'shape': data.shape,
                'cpu_utilization': cpu_utilization,
                'storage_ratio': c['storage_ratio'],
                'threads': c['threads']
            }

            quality_metrics = compress_zarr.eval_quality(data, jp2[:], metrics)
            tile_metrics.update(quality_metrics)

            all_metrics.append(tile_metrics)

    output_metrics_file = output_metrics_file.replace('.csv', '_' + os.path.basename(input_file) + '.csv')

    df = pd.DataFrame.from_records(all_metrics)
    df.to_csv(output_metrics_file, index_label='test_number')


if __name__ == "__main__":
    main()
