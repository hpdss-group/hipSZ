import sys
import time

import numpy as np

from pyhipSZ import (DIM_1D, MODE_FIXED, MODE_OUTLIER, MODE_PLAIN, TYPE_DOUBLE,
                      TYPE_FLOAT, DeviceBuffer, Stream, hipSZ)


def read_dataset(filename, dtype):
    return np.fromfile(filename, dtype=dtype)


def main():
    if len(sys.argv) < 9:
        print("Usage: python example-hpc.py <dataset_file> <data_type: f32|f64> <mode: fixed|plain|outlier> <dim: 1|2|3> <dim_z> <dim_y> <dim_x> <error_bound>")
        print("       if dim=1, dim_z, dim_y, and dim_x can be any value (e.g., 0).")
        print("       dim_x is the fastest changing dimension.")
        print("Example: python example-hpc.py data.f32 f32 plain 3 100 100 100 1e-3")
        sys.exit(1)

    filename = sys.argv[1]
    dtype_str = sys.argv[2]
    mode_str = sys.argv[3]
    dim = int(sys.argv[4])
    dim_z, dim_y, dim_x = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    error_bound = float(sys.argv[8])

    dims = (dim_x, dim_y, dim_z)

    if dtype_str == 'f32':
        data = read_dataset(filename, np.float32)
        data_type = TYPE_FLOAT
    elif dtype_str == 'f64':
        data = read_dataset(filename, np.float64)
        data_type = TYPE_DOUBLE
    else:
        print("Invalid data type.")
        sys.exit(1)

    modes = {'plain': MODE_PLAIN, 'outlier': MODE_OUTLIER, 'fixed': MODE_FIXED}
    if mode_str not in modes:
        print("Invalid mode.")
        sys.exit(1)
    mode = modes[mode_str]

    compressor = hipSZ()

    with Stream() as stream, \
         DeviceBuffer(data.nbytes) as d_oriData, \
         DeviceBuffer(data.nbytes + 16384) as d_cmpBytes, \
         DeviceBuffer(data.nbytes) as d_decData:
        d_oriData.copy_from_host(data)

        start_c = time.perf_counter()
        cmp_size = compressor.compress(d_oriData, d_cmpBytes, data.size, error_bound,
                                       dim=dim, dims=dims, data_type=data_type,
                                       mode=mode, stream=stream)
        stream.synchronize()
        end_c = time.perf_counter()

        start_d = time.perf_counter()
        compressor.decompress(d_decData, d_cmpBytes, data.size, cmp_size, error_bound,
                              dim=dim, dims=dims, data_type=data_type, mode=mode,
                              stream=stream)
        stream.synchronize()
        end_d = time.perf_counter()

        throughput_comp = (data.nbytes / (end_c - start_c)) / 1e9
        throughput_decomp = (data.nbytes / (end_d - start_d)) / 1e9

        print(f"\n===== hipSZ GPU Compression Test (HIP/DCU) =====")
        print(f"Data file:         {filename}")
        print(f"Type:              {dtype_str}")
        print(f"Mode:              {mode_str}")
        print(f"Dim:               {dim}D ({dim_z}, {dim_y}, {dim_x})")
        print(f"Error bound:       {error_bound}")
        print(f"Original size:     {data.nbytes / 1e6:.2f} MB")
        print(f"Compressed size:   {cmp_size / 1e6:.2f} MB")
        print(f"Compression ratio: {data.nbytes / cmp_size:.2f}x")
        print(f"Compression speed:   {throughput_comp:.2f} GB/s (10^9 B/s)")
        print(f"Decompression speed: {throughput_decomp:.2f} GB/s (10^9 B/s)")

        decompressed = np.empty_like(data)
        d_decData.copy_to_host(decompressed)
        err = np.abs(data.astype(np.float64) - decompressed.astype(np.float64))
        bad = int(np.sum(~(err <= error_bound * 1.1)))
        print(f"Within bound: {bad == 0} (max_err={err.max():.3g}, bad={bad})")
        sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
