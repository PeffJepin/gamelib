import multiprocessing.shared_memory as sm
from contextlib import contextmanager

import numpy as np
import pytest

from src.gamelib.sharedmem import DoubleBufferedArray, ArraySpec
from src.gamelib import sharedmem


class TestDoubleBufferedArray:
    def test_not_open_upon_init(self):
        dbl = DoubleBufferedArray("my name", float, 8)
        assert not dbl.is_open

    def test_connecting_before_allocation_happens_errors(self):
        dbl = DoubleBufferedArray("my name", int, 16)

        with pytest.raises(FileNotFoundError):
            dbl.connect()

    def test_is_open_after_connecting_to_shared_block(self):
        dbl = DoubleBufferedArray("my name", int, 16)
        sharedmem.allocate(dbl.specs)
        dbl.connect()

        assert dbl.is_open

    def test_making_changes_to_a_view_of_the_array(self):
        data_in = [1, 3, 3, 3, 5]
        data_out = [1, 10, 10, 10, 5]

        with self.mutating_operations_tester(data_in, data_out) as dbl:
            dbl[np.where(dbl == 3)] = 10

    def test_numpy_function(self):
        with self.dbl_buffered([1, 2, 3, 4, 5]) as dbl:
            arr2 = np.array([5, 4, 3, 2, 1])
            actual = np.add(dbl, arr2)
            expected = np.array([6, 6, 6, 6, 6])
            assert np.all(expected == actual)

    def test_add(self):
        with self.dbl_buffered([1, 2, 3]) as dbl:
            assert np.all(np.array([3, 4, 5]) == dbl + 2)

    def test_sub(self):
        with self.dbl_buffered([1, 2, 3]) as dbl:
            assert np.all(np.array([0, 1, 2]) == dbl - 1)

    def test_mul(self):
        with self.dbl_buffered([1, 2, 3]) as dbl:
            assert np.all(np.array([2, 4, 6]) == dbl * 2)

    def test_div(self):
        with self.dbl_buffered([2, 4, 8]) as dbl:
            assert np.all(np.array([1, 2, 4]) == dbl / 2)

    def test_iadd(self):
        data_in = [1, 2, 3]
        data_out = [6, 7, 8]

        with self.mutating_operations_tester(data_in, data_out) as dbl:
            dbl += 5

    def test_isub(self):
        data_in = [1, 2, 3]
        data_out = [0, 1, 2]

        with self.mutating_operations_tester(data_in, data_out) as dbl:
            dbl -= 1

    def test_imul(self):
        data_in = [1, 2, 3]
        data_out = [5, 10, 15]

        with self.mutating_operations_tester(data_in, data_out) as dbl:
            dbl *= 5

    def test_itruediv(self):
        data_in = [6.0, 4.0, 2.0]
        data_out = [3.0, 2.0, 1.0]

        with self.mutating_operations_tester(data_in, data_out) as dbl:
            dbl /= 2.0

    def test_ifloordiv(self):
        data_in = [6, 4, 2]
        data_out = [3, 2, 1]

        with self.mutating_operations_tester(data_in, data_out) as dbl:
            dbl //= 2

    def test_indexing(self):
        data_in = [1, 2, 3]
        data_out = [1, 11, 3]

        with self.mutating_operations_tester(data_in, data_out) as dbl:
            dbl[1] = 11

    def test_slicing(self):
        data_in = [1, 2, 3]
        data_out = [1, 123, 123]

        with self.mutating_operations_tester(data_in, data_out) as dbl:
            dbl[1:] = 123

    def test_multiple_operations_one_swap(self):
        in_ = [1, 2, 3, 4, 5]
        out = [5, 15, 25, 35, 45]

        with self.mutating_operations_tester(in_, out) as dbl:
            dbl *= 10
            dbl -= 5

    def test_flip_effects_all_instances(self):
        dbls = [DoubleBufferedArray("id", int, 10) for _ in range(3)]
        specs = [spec for dbl in dbls for spec in dbl.specs]
        sharedmem.allocate(specs)

        for dbl in dbls:
            dbl.connect()

        dbls[0][:] = 123

        for arr in dbls:
            assert all(arr == 0)

        dbls[0].flip()

        for arr in dbls:
            assert all(arr == 123)

    @contextmanager
    def dbl_buffered(self, initial_data):
        dbl = DoubleBufferedArray("id", type(initial_data[0]), len(initial_data))
        sharedmem.allocate(dbl.specs)
        dbl.connect()

        for arr in (dbl._read_arr, dbl._write_arr):
            arr[:] = initial_data[:]

        yield dbl

    @contextmanager
    def mutating_operations_tester(self, initial_data, expected_data):
        dbl = DoubleBufferedArray("id", type(initial_data[0]), len(initial_data))
        sharedmem.allocate(dbl.specs)
        dbl.connect()

        for arr in (dbl._read_arr, dbl._write_arr):
            arr[:] = initial_data[:]

        yield dbl
        assert isinstance(dbl, DoubleBufferedArray)
        assert all(dbl[:] == initial_data[:])
        dbl.flip()
        assert all(dbl[:] == expected_data[:])


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
