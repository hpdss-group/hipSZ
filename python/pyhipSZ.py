"""ctypes bindings for libhipSZ on Hygon/AMD HIP (DTK).

Notes inherited from the C API:
  * ``errorBound`` is a C ``float`` in the unified interface, so double
    precision data also gets a single-precision error bound.
  * Compressed-output capacity contract: allocate at least
    ``nbEle * elem_size`` bytes, plus headroom for tiny inputs (the rate
    array alone is padded to >= 1024 B for single-CTA 1D grids).
"""
import ctypes
import os

import numpy as np

_HIP_RT_CANDIDATES = ("libgalaxyhip.so", "libamdhip64.so")
_HIPSZ_CANDIDATES = (
    "../install/lib/libhipSZ.so",
    "../build/install/lib/libhipSZ.so",
    "libhipSZ.so",
)


def _load_first(names, env_var=None):
    if env_var and os.environ.get(env_var):
        return ctypes.CDLL(os.environ[env_var])
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    raise OSError(f"none of {names} loadable (set {env_var} to the full path)")


hiprt = _load_first(_HIP_RT_CANDIDATES, env_var="HIP_RT_LIB")
lib = _load_first(_HIPSZ_CANDIDATES, env_var="HIPSZ_LIB")


class HipError(RuntimeError):
    pass


def _hip_check(res, what):
    if res != 0:
        raise HipError(f"{what} failed with HIP error {res}")


# --- HIP runtime surface (signatures matter: restype defaults to c_int) ----
for _f, _args, _res in [
    ("hipMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
    ("hipFree", [ctypes.c_void_p], ctypes.c_int),
    ("hipMemcpy", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
    ("hipMemset", [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t], ctypes.c_int),
    ("hipStreamCreate", [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
    ("hipStreamDestroy", [ctypes.c_void_p], ctypes.c_int),
    ("hipStreamSynchronize", [ctypes.c_void_p], ctypes.c_int),
    ("hipDeviceSynchronize", [], ctypes.c_int),
    ("hipGetLastError", [], ctypes.c_int),
]:
    _fn = getattr(hiprt, _f)
    _fn.argtypes = _args
    _fn.restype = _res

HIP_MEMCPY_H2D, HIP_MEMCPY_D2H = 1, 2


class Uint3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]


# --- hipSZ C API -----------------------------------------------------------
_DIM, _TYPE, _MODE = ctypes.c_int, ctypes.c_int, ctypes.c_int
lib.hipSZ_compress_ex.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t), ctypes.c_float,
    _DIM, Uint3, _TYPE, _MODE, ctypes.c_void_p,
]
lib.hipSZ_compress_ex.restype = None
lib.hipSZ_decompress_ex.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_float,
    _DIM, Uint3, _TYPE, _MODE, ctypes.c_void_p,
]
lib.hipSZ_decompress_ex.restype = None

DIM_1D, DIM_2D, DIM_3D = 1, 2, 3
TYPE_FLOAT, TYPE_DOUBLE = 0, 1
MODE_PLAIN, MODE_OUTLIER, MODE_FIXED = 0, 1, 2


class DeviceBuffer:
    """Owns one device allocation; free explicitly or use as a context."""

    def __init__(self, nbytes):
        self.nbytes = nbytes
        self.ptr = ctypes.c_void_p()
        _hip_check(hiprt.hipMalloc(ctypes.byref(self.ptr), nbytes), "hipMalloc")

    def copy_from_host(self, arr):
        assert arr.nbytes <= self.nbytes, "host array larger than device buffer"
        _hip_check(hiprt.hipMemcpy(self.ptr, arr.ctypes.data_as(ctypes.c_void_p),
                                   arr.nbytes, HIP_MEMCPY_H2D), "hipMemcpy H2D")

    def copy_to_host(self, arr):
        assert arr.nbytes <= self.nbytes, "host array larger than device buffer"
        _hip_check(hiprt.hipMemcpy(arr.ctypes.data_as(ctypes.c_void_p), self.ptr,
                                   arr.nbytes, HIP_MEMCPY_D2H), "hipMemcpy D2H")

    def fill_zero(self):
        _hip_check(hiprt.hipMemset(self.ptr, 0, self.nbytes), "hipMemset")

    def free(self):
        if self.ptr:
            _hip_check(hiprt.hipFree(self.ptr), "hipFree")
            self.ptr = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free()


class Stream:
    def __init__(self):
        self.ptr = ctypes.c_void_p()
        _hip_check(hiprt.hipStreamCreate(ctypes.byref(self.ptr)), "hipStreamCreate")

    def synchronize(self):
        _hip_check(hiprt.hipStreamSynchronize(self.ptr), "hipStreamSynchronize")

    def destroy(self):
        if self.ptr:
            _hip_check(hiprt.hipStreamDestroy(self.ptr), "hipStreamDestroy")
            self.ptr = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()


class hipSZ:
    def compress(self, d_oriData, d_cmpBytes, num_elements, error_bound=0.01,
                 dim=DIM_1D, dims=(0, 0, 0), data_type=TYPE_FLOAT, mode=MODE_PLAIN,
                 stream=None):
        compressed_size = ctypes.c_size_t(0)
        lib.hipSZ_compress_ex(
            _as_ptr(d_oriData), _as_ptr(d_cmpBytes),
            ctypes.c_size_t(num_elements), ctypes.byref(compressed_size),
            ctypes.c_float(error_bound),
            _DIM(dim), Uint3(dims[0], dims[1], dims[2]),
            _TYPE(data_type), _MODE(mode),
            _stream_ptr(stream),
        )
        _check_last_hip_error()
        return compressed_size.value

    def decompress(self, d_decData, d_cmpBytes, num_elements, compressed_size,
                   error_bound=0.01, dim=DIM_1D, dims=(0, 0, 0),
                   data_type=TYPE_FLOAT, mode=MODE_PLAIN, stream=None):
        lib.hipSZ_decompress_ex(
            _as_ptr(d_decData), _as_ptr(d_cmpBytes),
            ctypes.c_size_t(num_elements), ctypes.c_size_t(compressed_size),
            ctypes.c_float(error_bound),
            _DIM(dim), Uint3(dims[0], dims[1], dims[2]),
            _TYPE(data_type), _MODE(mode),
            _stream_ptr(stream),
        )
        _check_last_hip_error()


def _as_ptr(x):
    if isinstance(x, DeviceBuffer):
        return x.ptr
    return ctypes.c_void_p(x)


def _stream_ptr(s):
    if s is None:
        return ctypes.c_void_p(0)
    return s.ptr if isinstance(s, Stream) else ctypes.c_void_p(s)


def _check_last_hip_error():
    _hip_check(hiprt.hipGetLastError(), "asynchronous HIP work")
