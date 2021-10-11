import multiprocessing.shared_memory as sm

import numpy as np
import pytest

from src.gamelib.sharedmem import ArraySpec
from src.gamelib import sharedmem


class TestModule:
    def test_allocate_creates_a_shared_memory_file(self):
        spec = sharedmem.ArraySpec("arr1", int, 100)

        sharedmem.allocate([spec])

        assert sm.SharedMemory(sharedmem._SHM_NAME)

    def test_connection_before_allocation_fails(self):
        spec = sharedmem.ArraySpec("arr1", int, 100)

        with pytest.raises(FileNotFoundError):
            sharedmem.connect(spec)

    def test_after_allocation_connect_returns_array(self):
        spec = sharedmem.ArraySpec("arr1", int, 100)
        sharedmem.allocate([spec])

        arr = sharedmem.connect(spec)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (100,)
        assert arr.dtype == int

    def test_multiple_connections_share_the_same_array(self):
        spec = sharedmem.ArraySpec("arr1", int, 100)
        sharedmem.allocate([spec])

        arr1 = sharedmem.connect(spec)
        arr2 = sharedmem.connect(spec)

        arr1 += 5
        assert np.all(arr1 == arr2)

    def test_unlink_closes_the_shm_file(self):
        spec = sharedmem.ArraySpec("arr1", int, 100)
        sharedmem.allocate([spec])

        sharedmem.unlink()

        with pytest.raises(FileNotFoundError):
            sm.SharedMemory(sharedmem._SHM_NAME)

    def test_with_many_different_specs(self):
        dtypes = [int, bool, float]
        lengths = [100, 200, 50, 133]
        specs = [
            ArraySpec(f"arr{i}", dtypes[i % 3], lengths[i % 4]) for i in range(100)
        ]
        sharedmem.allocate(specs)

        for spec in specs:
            arr = sharedmem.connect(spec)
            assert spec.dtype == arr.dtype
            assert spec.length == len(arr)

            # ensure no buffer overlap
            assert np.all(arr == 0)
            if arr.dtype == bool:
                arr[:] = True
            else:
                arr += 1

    def test_readonly_optional_flag(self):
        spec = sharedmem.ArraySpec("arr1", int, 100)
        sharedmem.allocate([spec])

        readonly = sharedmem.connect(spec, readonly=True)

        with pytest.raises(ValueError):
            readonly += 1
