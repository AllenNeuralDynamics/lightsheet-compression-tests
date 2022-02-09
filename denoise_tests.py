import logging
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import skimage
import zarr

import compress_zarr


def identity(data):
    return data


def get_funcs():
    return {
        'identity': identity,
        'median': skimage.filters.rank.median
    }


def plot(metrics_file, compressor, shuffle):
    df = pd.read_csv(metrics_file, index_col='test_number')
    df = df[df['compressor_name'] == compressor]
    df = df[df['shuffle'] == shuffle]

    base_df = df[df['filter'] == 'identity']
    filtered_df = df[df['filter'] == 'median']

    fig, axes = plt.subplots(2, 1, sharex=True)
    bar_width = 0.2

    axes[0].bar(base_df['level'].to_numpy() - bar_width, base_df['storage_ratio'], bar_width, label="no filter")
    axes[0].bar(filtered_df['level'].to_numpy() + bar_width, filtered_df['storage_ratio'], bar_width, label="median")
    axes[0].set_ylabel("storage ratio")
    axes[0].set_xlabel("compression level")
    axes[0].set_xticks(base_df['level'].unique().tolist())
    axes[0].grid(axis='y')

    axes[1].bar(base_df['level'].to_numpy() - bar_width, base_df['compress_bps'] / (2 ** 20), bar_width, label="no filter")
    axes[1].bar(filtered_df['level'].to_numpy() + bar_width, filtered_df['compress_bps'] / (2 ** 20), bar_width, label="median")
    axes[1].set_ylabel("compress speed (MiB/s)")
    axes[1].set_xlabel("compression level")
    axes[1].set_xticks(base_df['level'].unique().tolist())
    axes[1].grid(axis='y')

    plt.suptitle(compressor + f", shuffle {shuffle}")


def run(input_file, num_tiles, output_data_file, output_metrics_file):
    funcs = get_funcs()
    compressors = compress_zarr.build_compressors("blosc", [0], [0])
    all_metrics = []

    for i in range(num_tiles):
        data, rslice, read_time = compress_zarr.read_random_chunk(input_file, 1)
        print(f"loaded data with shape {data.shape}, {data.nbytes / 2 ** 20} MiB")

        for f in funcs:
            for c in compressors:
                compressor = c['compressor']

                tile_metrics = {
                    "compressor_name": c['name'],
                    "tile": rslice
                }
                tile_metrics.update(c['params'])

                psutil.cpu_percent(interval=None)
                start = timer()
                # include filter step in compression time
                filtered = funcs[f](data)
                ds = zarr.DirectoryStore(output_data_file)
                za = zarr.array(filtered, compressor=compressor, store=ds, overwrite=True)
                logging.info(str(za.info))
                cpu_utilization = psutil.cpu_percent(interval=None)
                end = timer()
                compress_dur = end - start

                storage_ratio = za.nbytes / za.nbytes_stored
                compress_bps = data.nbytes / compress_dur

                print(f"compression time = {compress_dur}, "
                      f"bps = {compress_bps}, "
                      f"ratio = {storage_ratio}, "
                      f"cpu = {cpu_utilization}%")

                tile_metrics.update({
                    'name': c['name'],
                    'bytes_read': za.nbytes,
                    'compress_time': compress_dur,
                    'bytes_written': za.nbytes_stored,
                    'shape': data.shape,
                    'cpu_utilization': cpu_utilization,
                    'storage_ratio': storage_ratio,
                    'compress_bps': compress_bps,
                    'filter': f
                })

                all_metrics.append(tile_metrics)

    df = pd.DataFrame.from_records(all_metrics)
    df.to_csv(output_metrics_file, index_label='test_number')


def main():
    input_file = r"C:\Users\cameron.arshadi\Downloads\BrainSlice1_MMStack_Pos33_15_shift.tif"
    output_data_file = "./test_file.zarr"
    output_metrics_file = "./median-test-metrics.csv"
    num_tiles = 10

    run(input_file, num_tiles, output_data_file, output_metrics_file)

    plot(output_metrics_file, compressor="blosc-zstd", shuffle=1)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
