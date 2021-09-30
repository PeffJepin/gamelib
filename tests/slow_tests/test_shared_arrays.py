import multiprocessing as mp

import numpy as np

from src.gamelib.sharedmem import SharedArray


class TestSharedArrays:
    def test_a_write_in_a_second_process_is_seen_in_the_first_process(self):
        arr = SharedArray.create("test", (10, 10), np.uint8)
        arr[:, :] = 0

        process = mp.Process(target=self.run_in_process)
        process.start()
        process.join()

        try:
            assert np.all(arr == 100)
        finally:
            arr.unlink()

    @classmethod
    def run_in_process(cls):
        arr = SharedArray("test", (10, 10), np.uint8)
        arr[:, :] = 100
