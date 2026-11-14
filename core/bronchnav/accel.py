"""Optional CUDA acceleration through CuPy.

Everything in bronchnav works on the CPU without a GPU. When CuPy and an NVIDIA GPU are present the heavy array
steps (Gaussian filters, Hessians, labeling, morphology) run on the GPU instead. Set the environment variable
BRONCHNAV_GPU=0 to force the CPU path (used by the parity tests), or BRONCHNAV_GPU=1 to require the GPU.

Usage:  xp, ndx = backend()   # numpy + scipy.ndimage, or cupy + cupyx.scipy.ndimage
        a = to_device(host_array); ...; host = to_host(a)
"""

from __future__ import annotations

import os

import numpy as np
import scipy.ndimage as _cpu_ndi

_state: dict = {}


def gpu_available() -> bool:
    """True when CuPy imports, a CUDA device exists and a tiny kernel runs. Cached; honours BRONCHNAV_GPU."""
    flag = os.environ.get("BRONCHNAV_GPU", "auto").strip().lower()
    if flag in ("0", "off", "false", "cpu"):
        return False
    if "ok" not in _state:
        try:
            import cupy as cp

            ok = cp.cuda.runtime.getDeviceCount() > 0 and int(cp.asarray([1, 2, 3]).sum()) == 6
            _state["name"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() if ok else ""
        except Exception as e:  # no CuPy, no driver, wrong CUDA, ...
            ok, _state["why"], _state["name"] = False, f"{type(e).__name__}: {e}", ""
        _state["ok"] = ok
    if flag in ("1", "on", "true", "gpu") and not _state["ok"]:
        raise RuntimeError(f"BRONCHNAV_GPU=1 but no usable CUDA GPU: {_state.get('why', 'unknown')}")
    return bool(_state["ok"])


def is_gpu_error(exc: BaseException) -> bool:
    """True for CUDA/CuPy failures (out of memory, driver or kernel errors), which a CPU retry can get past."""
    name = type(exc).__name__
    mod = type(exc).__module__ or ""
    return mod.startswith(("cupy", "cupyx")) or name in ("OutOfMemoryError", "CUDARuntimeError", "CUDADriverError") or "CUDA" in str(exc)[:200]


def disable_gpu(why: str) -> None:
    """Stop using the GPU for the rest of this process (after a GPU failure)."""
    _state["ok"], _state["why"] = False, why
    free_gpu_memory_quietly()


def free_gpu_memory_quietly() -> None:
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def gpu_name() -> str:
    gpu_available()
    return _state.get("name", "")


def backend():
    """(array module, ndimage module): CuPy's when a GPU is usable, else NumPy's."""
    if gpu_available():
        import cupy as cp
        import cupyx.scipy.ndimage as cndi

        return cp, cndi
    return np, _cpu_ndi


def is_gpu_array(a) -> bool:
    return type(a).__module__.split(".")[0] == "cupy"


def to_device(a, dtype=None):
    """Move a host array to the backend's device (a no-op on the CPU path)."""
    xp, _ = backend()
    out = xp.asarray(a) if dtype is None else xp.asarray(a, dtype=dtype)
    return out


def to_host(a) -> np.ndarray:
    return a.get() if is_gpu_array(a) else np.asarray(a)


def free_gpu_memory() -> None:
    """Return cached GPU blocks to the driver (a long-lived server should call this after a job)."""
    if gpu_available():
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()


def dilate(ndx, mask, structure, iterations: int = 1):
    """Binary dilation that also works on the GPU for several iterations (CuPy needs brute_force for that)."""
    if is_gpu_array(mask):
        import cupy as cp

        return ndx.binary_dilation(mask, cp.asarray(structure), iterations=iterations, brute_force=True)
    return ndx.binary_dilation(mask, structure, iterations=iterations)
