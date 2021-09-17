from contextlib import contextmanager

import numpy as np

from src.gamelib import sharedmem


class TestSharedArrays:
    def test_single_array_creation(self):
        spec = ("my_array", (10, 20), np.uint8)
        with managed_memory(
            sharedmem.SharedArrays([spec], create=True)
        ) as shared_arrays:
            arr = shared_arrays["my_array"]
            assert arr.shape == (10, 20) and arr.dtype == np.uint8

    def test_multiple_array_creation(self):
        specs = [("arr1", (4, 8), float), ("arr2", (32, 10), np.ubyte)]
        with managed_memory(
            sharedmem.SharedArrays(specs, create=True)
        ) as shared_arrays:
            arr1 = shared_arrays["arr1"]
            arr2 = shared_arrays["arr2"]
            assert (arr1.shape, arr2.shape) == ((4, 8), (32, 10))
            assert (arr1.dtype, arr2.dtype) == (float, np.ubyte)

    def test_access_from_separate_object_after_creation(self):
        specs = [("arr", (10,), np.uint8)]
        with managed_memory(
            sharedmem.SharedArrays(specs, create=True)
        ) as shared_arrays:
            different_accessor = sharedmem.SharedArrays(specs)
            shared_arrays["arr"][:] = 111
            assert np.all(different_accessor["arr"] == 111)


class TestDoubleBufferedArrays:
    def test_reads_and_writes_occur_on_separate_buffers(self):
        specs = [("arr", (10,), np.uint8)]
        with managed_memory(
            sharedmem.DoubleBufferedArrays(specs, create=True)
        ) as dbl_buffered:
            arr = dbl_buffered["arr"]
            before = arr[:]
            arr[:] = 213
            assert np.all(arr == before)

    def test_reads_are_made_visible_after_a_swap(self):
        specs = [("arr", (10,), np.uint8)]
        with managed_memory(
            sharedmem.DoubleBufferedArrays(specs, create=True)
        ) as dbl_buffered:
            arr = dbl_buffered["arr"]
            arr[:] = 213
            dbl_buffered.swap()
            assert np.all(arr == 213)

    def test_a_buffer_swap_from_elsewhere_is_seen_everywhere(self):
        specs = [("arr", (10,), np.uint8)]
        with managed_memory(
            sharedmem.DoubleBufferedArrays(specs, create=True)
        ) as dbl_buffered:
            another_view = sharedmem.DoubleBufferedArrays(specs)

            arr = dbl_buffered["arr"]
            arr[:] = 213
            another_view.swap()
            assert np.all(arr == 213)

    def test_supports_numpy_function_calls(self):
        specs = [("arr", (10,), np.uint8)]
        with managed_memory(
            sharedmem.DoubleBufferedArrays(specs, create=True)
        ) as dbl_buffered:
            arr = dbl_buffered["arr"]
            arr[:] = 2
            dbl_buffered.swap()

            assert np.all(np.add(arr, 2) == 4)


@contextmanager
def managed_memory(shared: sharedmem.SharedArrays):
    """Test use only. Make sure tests remain independent of one another."""
    try:
        yield shared
    finally:
        shared.release()
