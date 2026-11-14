"""GPU backend: it must agree with the CPU and never be the reason a run fails. GPU-only checks skip without CuPy."""

import numpy as np
import pytest
import scipy.ndimage as ndi

from bronchnav import accel
from bronchnav.segmentation import airway
from bronchnav.segmentation.small_airways import extend_into_small_airways, tube_measure

SP = 1.0


def _use(monkeypatch, mode):
    monkeypatch.setenv("BRONCHNAV_GPU", mode)
    accel._state.clear()


def _require_gpu(monkeypatch):
    _use(monkeypatch, "auto")
    if not accel.gpu_available():
        pytest.skip("no usable CUDA GPU / CuPy")


def _volume():
    """Lung with an open-topped trunk and a thinner side branch that has a brighter wall."""
    rng = np.random.default_rng(1)
    n = 72
    z, y, x = np.mgrid[:n, :n, :n].astype(float)
    hu = -850.0 + rng.normal(0, 8.0, (n, n, n))
    trunk = ((y - 36) ** 2 + (x - 20) ** 2) <= 5.0**2
    rr = (z - 36) ** 2 + (y - 36) ** 2
    along = (x >= 20) & (x < 55)
    hu[(rr <= 3.4**2) & along] = -300.0
    hu[(rr <= 2.0**2) & along] = -900.0
    hu[trunk] = -1000.0
    hu[:5] = -1000.0  # the trunk opens at the top of the scan (a trachea-like seed)
    return ndi.gaussian_filter(hu, 0.5).astype(np.float32), trunk


def test_cpu_backend_is_numpy_when_forced(monkeypatch):
    _use(monkeypatch, "0")
    xp, ndx = accel.backend()
    assert xp is np and ndx is ndi and not accel.gpu_available()


def test_requiring_a_gpu_that_is_missing_is_an_error(monkeypatch):
    _use(monkeypatch, "1")
    if accel.gpu_available():
        pytest.skip("this machine has a GPU")
    with pytest.raises(RuntimeError):
        accel.gpu_available()


def test_tube_measure_and_extension_match_between_cpu_and_gpu(monkeypatch):
    hu, trunk = _volume()
    _use(monkeypatch, "0")
    t_cpu = tube_measure(hu, SP)
    out_cpu, info_cpu = extend_into_small_airways(hu, SP, trunk, reach_mm=30.0)
    _require_gpu(monkeypatch)
    t_gpu = tube_measure(hu, SP)
    out_gpu, info_gpu = extend_into_small_airways(hu, SP, trunk, reach_mm=30.0)
    # float rounding can flip a voxel that sits exactly on a cut-off of the tube test, so compare statistically
    assert np.mean(np.abs(t_cpu - t_gpu) > 0.5) < 1e-3 and np.abs(t_cpu - t_gpu).mean() < 0.01
    assert (out_cpu == out_gpu).all() and info_cpu == info_gpu
    assert info_cpu["added_ml"] > 0  # the test actually exercises growth


def test_threshold_ladder_matches_between_cpu_and_gpu(monkeypatch):
    hu, _ = _volume()
    seed = (30, 36, 20)
    _use(monkeypatch, "0")
    cpu = airway.segment_airway(hu, SP, seed_zyx=seed, small_airways=True)
    _require_gpu(monkeypatch)
    gpu = airway.segment_airway(hu, SP, seed_zyx=seed, small_airways=True)
    assert (cpu.mask == gpu.mask).all()
    assert cpu.threshold_hu == gpu.threshold_hu and cpu.volumes_ml == gpu.volumes_ml


def test_a_gpu_failure_repeats_the_step_on_the_cpu(monkeypatch):
    hu, _ = _volume()
    calls, disabled = [], []

    class OutOfMemoryError(Exception):
        pass

    real = airway._segment_airway

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise OutOfMemoryError("out of memory allocating 2,000,000,000 bytes")
        return real(*a, **k)

    _use(monkeypatch, "0")  # the retry itself runs on the CPU
    monkeypatch.setattr(airway, "_segment_airway", flaky)
    monkeypatch.setattr(accel, "gpu_available", lambda: True)
    monkeypatch.setattr(accel, "disable_gpu", lambda why: disabled.append(why))
    seg = airway.segment_airway(hu, SP, seed_zyx=(30, 36, 20))
    assert len(calls) == 2 and len(disabled) == 1 and "out of memory" in disabled[0]
    assert seg.mask.any() and any("CPU" in n for n in seg.notes)


def test_other_errors_are_not_swallowed_by_the_gpu_fallback(monkeypatch):
    hu, _ = _volume()
    monkeypatch.setattr(airway, "_segment_airway", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad seed")))
    monkeypatch.setattr(accel, "gpu_available", lambda: True)
    with pytest.raises(ValueError, match="bad seed"):
        airway.segment_airway(hu, SP, seed_zyx=(30, 36, 20))
