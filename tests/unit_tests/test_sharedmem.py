import multiprocessing
from contextlib import contextmanager

import numpy as np

from src.gamelib.sharedmem import DoubleBufferedArray, SharedBlock, ArraySpec


class TestDoubleBufferedArray:
    def test_not_open_upon_init(self):
        dbl = DoubleBufferedArray("my name", float)
        assert not dbl.is_open

    def test_is_open_after_connecting_to_shared_block(self):
        dbl = DoubleBufferedArray("my name", int)
        blk = SharedBlock(dbl.specs, 8)
        dbl.connect(blk)

        try:
            assert dbl.is_open
        finally:
            blk.unlink_shm()

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
        dbls = [DoubleBufferedArray("id", int) for _ in range(3)]
        blk = SharedBlock(dbls[0].specs, 16)
        for dbl in dbls:
            dbl.connect(blk)

        try:
            dbls[0][:] = 123

            for arr in dbls:
                assert all(arr == 0)

            dbls[0].flip()

            for arr in dbls:
                assert all(arr == 123)

        finally:
            blk.unlink_shm()

    @contextmanager
    def dbl_buffered(self, initial_data):
        dbl = DoubleBufferedArray("id", type(initial_data[0]))
        blk = SharedBlock(dbl.specs, len(initial_data))
        dbl.connect(blk)
        for arr in (dbl._read_arr, dbl._write_arr):
            arr[:] = initial_data[:]

        try:
            yield dbl
        finally:
            blk.unlink_shm()

    @contextmanager
    def mutating_operations_tester(self, initial_data, expected_data):
        dbl = DoubleBufferedArray("id", type(initial_data[0]))
        blk = SharedBlock(dbl.specs, len(initial_data))
        dbl.connect(blk)
        for arr in (dbl._read_arr, dbl._write_arr):
            arr[:] = initial_data[:]

        try:
            yield dbl
            assert isinstance(dbl, DoubleBufferedArray)
            assert all(dbl[:] == initial_data[:])
            dbl.flip()
            assert all(dbl[:] == expected_data[:])
        finally:
            blk.unlink_shm()


class TestSharedBlock:
    def test_shared_arrays_are_automatically_allocated(self):
        specs = [ArraySpec(str(i), int) for i in range(4)]
        blk = SharedBlock(specs, 100)
        arr01 = blk[specs[0]]

        try:
            all(arr01 == 0)
        finally:
            blk.unlink_shm()

    def test_max_entities_dictates_length_of_array(self):
        spec1 = ArraySpec("id1", int)
        spec2 = ArraySpec("id2", int)
        blk1 = SharedBlock([spec1], max_entities=10)
        blk2 = SharedBlock([spec2], max_entities=20)

        arr1 = blk1[spec1]
        arr2 = blk2[spec2]

        try:
            assert len(arr1) == 10
            assert len(arr2) == 20
        finally:
            blk1.unlink_shm()
            blk2.unlink_shm()

    def test_arrays_share_the_same_memory_when_created_across_a_pipe_connection(self):
        a, b = multiprocessing.Pipe()
        spec = ArraySpec("some id", int)
        blk1 = SharedBlock([spec], 8)
        arr1 = blk1[spec]

        a.send(blk1)
        assert b.poll(1)
        blk2 = b.recv()
        arr2 = blk2[spec]

        try:
            assert all(arr1 == arr2)
            arr1 += 100
            assert all(arr1 == arr2)
        finally:
            blk2.close_shm()
            blk1.unlink_shm()
