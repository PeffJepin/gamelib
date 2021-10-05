import multiprocessing
from contextlib import contextmanager
from multiprocessing import shared_memory

import numpy as np

from src.gamelib.sharedmem import SharedArray, DoubleBufferedArray, SharedBlock


class TestSharedArray:
    def test_not_allocated_on_init(self):
        arr = SharedArray("id", (10,), int)
        assert arr._arr is None

    def test_create_returns_an_allocated_instance(self):
        arr = SharedArray.create("id", (10,), int)
        try:
            assert all(arr == 0)
        finally:
            arr.unlink()

    def test_automatically_tries_to_load_into_an_existing_allocation(self):
        arr = SharedArray.create("id", (10,), int)
        arr1 = SharedArray("id", (10,), int)
        try:
            arr[0] = 33
            assert arr1[0] == 33
        finally:
            arr1.close()
            arr.unlink()

    def test_can_open_a_view_into_given_shm(self):
        arr = SharedArray("id", (10,), int)
        try:
            blank = np.zeros((10,), int)
            shm = shared_memory.SharedMemory(
                "some name", create=True, size=blank.nbytes
            )
            arr.open_view(shm)
            shm = None

            assert all(arr[:] == 0)
        finally:
            arr.unlink()

    def test_close_only_effects_one_instance(self):
        initial = np.array([1, 2, 3])
        arr1 = SharedArray.create("id", array=initial)
        arr2 = SharedArray("id", arr1._shape, arr1._dtype)
        arr1.close()

        try:
            assert all(initial == arr2)
        finally:
            arr2.unlink()

    def test_is_open(self):
        arrays = [SharedArray(i, (1,), int) for i in range(2)]
        try:
            assert not any([arr.is_open for arr in arrays])

            for arr in arrays:
                arr.allocate()
                assert arr.is_open
                arr.close()
                assert not arr.is_open
        finally:
            for arr in arrays:
                arr.unlink()

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
            arr.unlink()


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

    def test_flip_effects_all_instances(self):
        dbl1 = DoubleBufferedArray.create("id", shape=(10,), dtype=np.uint8)
        dbl2 = DoubleBufferedArray("id", (10,), np.uint8)
        dbl3 = DoubleBufferedArray("id", (10,), np.uint8)
        dbl1[:] = 123
        try:
            for arr in (dbl1, dbl2, dbl3):
                assert all(arr == 0)
            dbl1.flip()
            for arr in (dbl1, dbl2, dbl3):
                assert all(arr == 123)
        finally:
            dbl3.close()
            dbl2.close()
            dbl1.unlink()

    @contextmanager
    def dbl_buffered(self, initial_data):
        dbl = DoubleBufferedArray.create("id", array=np.array(initial_data))
        try:
            yield dbl
        finally:
            dbl.unlink()

    @contextmanager
    def mutating_operations_tester(self, initial_data, expected_data):
        arr = np.array(initial_data)
        dbl = DoubleBufferedArray.create("id", array=arr)
        try:
            yield dbl
            assert isinstance(dbl, DoubleBufferedArray)
            assert all(dbl[:] == initial_data[:])
            dbl.flip()
            assert all(dbl[:] == expected_data[:])
        finally:
            dbl.unlink()


class TestSharedBlock:
    def test_shared_arrays_are_automatically_allocated(self):
        arrays = [SharedArray(i, (20,), int) for i in range(4)]
        blk = SharedBlock(arrays)

        try:
            assert all(arr.is_open for arr in arrays)
        finally:
            blk.unlink()

    def test_keys_can_be_mutated(self):
        arrays = [SharedArray(i, (20,), int) for i in range(4)]
        blk = SharedBlock(arrays, name_extra="_mutated")

        view0 = SharedArray("0_mutated", (20,), int)
        try:
            arrays[0][:] = 100
            assert all(view0[:] == 100)
        finally:
            view0.unlink()
            blk.unlink()

    def test_allocated_arrays_can_connect_new_views(self):
        arrays = [SharedArray(i, (20,), int) for i in range(4)]
        blk = SharedBlock(arrays, name_extra="_mutated")

        view0 = SharedArray(0, (20,), int)
        view1 = SharedArray(1, (20,), int)
        try:
            assert not view0.is_open and not view1.is_open
            blk.connect_arrays(view0, view1)
            assert view0.is_open and view1.is_open
            arrays[0][:] = 100
            arrays[1][:] = 200
            assert all(view0[:] == arrays[0][:])
            assert all(view1[:] == arrays[1][:])
        finally:
            view0.unlink()
            view1.unlink()
            blk.unlink()

    def test_can_serve_shm_to_arrays_across_a_pipe(self):
        a, b = multiprocessing.Pipe()
        arrays = [SharedArray(i, (20,), int) for i in range(4)]
        blk1 = SharedBlock(arrays, name_extra="_mutated")

        a.send(blk1)
        assert b.poll(1)
        blk2 = b.recv()

        view = SharedArray(0, (20,), int)
        try:
            assert not view.is_open
            blk2.connect_arrays(view)
            assert view.is_open
            arrays[0][:] = 100
            assert all(view[:] == arrays[0][:])
        finally:
            view.unlink()
            blk2.close()
            blk1.unlink()
