from contextlib import contextmanager

import numpy as np
import pytest

from src.gamelib.sharedmem import SharedArray, DoubleBufferedArray


class TestSharedArray:
    def test_must_be_created_before_use(self):
        with pytest.raises(FileNotFoundError):
            arr = SharedArray("id", (10,), np.uint8)

        arr = SharedArray.create("id", (10,), np.uint8)
        arr.close()

    def test_closes_only_a_single_connection(self):
        initial = np.array([1, 2, 3, 4, 5])
        arr1 = SharedArray.create("id", array=initial)
        arr2 = SharedArray("id", initial.shape, initial.dtype)
        try:
            arr1.close()
            assert all(arr2[:] == initial[:])
        finally:
            arr2.close()

    def test_denies_connection_after_all_connections_closed(self):
        initial = np.array([1, 2, 3])
        arr1 = SharedArray.create("id", array=initial)
        arr1.close()

        with pytest.raises(FileNotFoundError):
            SharedArray("id", initial.shape, initial.dtype)

    def test_eq(self):
        with self.shared_numpy([1, 2, 3]) as arr:
            assert np.all(np.array([1, 2, 3]) == arr)

    def test_add(self):
        with self.shared_numpy([3, 2, 1]) as arr:
            assert np.all(np.array([8, 7, 6]) == arr + 5)

    def test_sub(self):
        with self.shared_numpy([3, 2, 1]) as arr:
            assert np.all(np.array([2, 1, 0]) == arr - 1)

    def test_mul(self):
        with self.shared_numpy([3, 2, 1]) as arr:
            assert np.all(np.array([30, 20, 10]) == arr * 10)

    def test_div(self):
        with self.shared_numpy([100, 50, 10]) as arr:
            assert np.all(np.array([10, 5, 1]) == arr / 10)

    def test_iadd(self):
        with self.shared_numpy([3, 2, 1]) as arr:
            arr += 5
            assert np.all(np.array([8, 7, 6]) == arr)
            assert isinstance(arr, SharedArray)

    def test_isub(self):
        with self.shared_numpy([3, 2, 1]) as arr:
            arr -= 1
            assert np.all(np.array([2, 1, 0]) == arr)
            assert isinstance(arr, SharedArray)

    def test_imul(self):
        with self.shared_numpy([3, 2, 1]) as arr:
            arr *= 10
            assert np.all(np.array([30, 20, 10]) == arr)
            assert isinstance(arr, SharedArray)

    def test_itruediv(self):
        with self.shared_numpy([100.0, 50.0, 10.0]) as arr:
            arr /= 10
            assert np.all(np.array([10.0, 5.0, 1.0]) == arr)
            assert isinstance(arr, SharedArray)

    def test_ifloordiv(self):
        with self.shared_numpy([100, 50, 10]) as arr:
            arr //= 10
            assert np.all(np.array([10, 5, 1]) == arr)
            assert isinstance(arr, SharedArray)

    def test_indexing(self):
        with self.shared_numpy([1, 2, 3]) as arr:
            assert np.all(arr[1] == 2)
            arr[1] = 5
            assert np.all(np.array([1, 5, 3]) == arr)

    def test_slicing(self):
        with self.shared_numpy([1, 2, 3]) as arr:
            assert all(arr[:] == [1, 2, 3][:])
            arr[1:] = 10
            assert np.all(np.array([1, 10, 10]) == arr)

    def test_numpy_func(self):
        with self.shared_numpy([1, 2, 3, 4, 5]) as arr:
            arr2 = np.array([5, 4, 3, 2, 1])
            actual = np.add(arr, arr2)
            expected = np.array([6, 6, 6, 6, 6])
            assert np.all(expected == actual)

    @contextmanager
    def shared_numpy(self, initial_data):
        arr = SharedArray.create("id", array=np.array(initial_data))
        try:
            yield arr
        finally:
            arr.close()


class TestDoubleBufferedArray:
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

    def test_access_from_multiple_objects(self):
        dbl1 = DoubleBufferedArray.create("1", shape=(10,), dtype=np.uint8)

        dbl1[:] = 200
        dbl1.swap()

        dbl2 = DoubleBufferedArray("1", (10,), np.uint8)
        try:
            assert np.all(dbl2 == 200)
        finally:
            dbl1.close()
            dbl2.close()

    def test_swap_effects_all_instances(self):
        dbl1 = DoubleBufferedArray.create("id", shape=(10,), dtype=np.uint8)
        dbl2 = DoubleBufferedArray("id", (10,), np.uint8)
        dbl3 = DoubleBufferedArray("id", (10,), np.uint8)

        dbl1[:] = 123

        try:
            for arr in (dbl1, dbl2, dbl3):
                assert not np.all(arr == 123)
            DoubleBufferedArray("id", (10,), np.uint8).swap()
            for arr in (dbl1, dbl2, dbl3):
                assert np.all(arr == 123)
        finally:
            dbl1.close()
            dbl2.close()
            dbl3.close()

    @contextmanager
    def dbl_buffered(self, initial_data):
        dbl = DoubleBufferedArray.create("id", array=np.array(initial_data))
        try:
            yield dbl
        finally:
            dbl.close()

    @contextmanager
    def mutating_operations_tester(self, initial_data, expected_data):
        arr = np.array(initial_data)
        dbl = DoubleBufferedArray.create("id", array=arr)
        try:
            yield dbl
            # some operations done with dbl here
        finally:
            assert isinstance(dbl, DoubleBufferedArray)
            assert all(dbl[:] == initial_data[:])
            dbl.swap()
            assert all(dbl[:] == expected_data[:])
            dbl.close()
