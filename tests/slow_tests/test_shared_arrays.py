import multiprocessing as mp

import numpy as np

from src.gamelib.sharedmem import SharedArrays


class TestSharedArrays:
    def test_a_write_in_a_second_process_is_seen_in_the_first_process(self):
        array_spec = ("test", (10, 10), np.uint8)
        shared_arrays = SharedArrays([array_spec], create=True)
        shared_arrays["test"][:, :] = 0

        process = mp.Process(target=self.run_in_process, args=(array_spec,))
        process.start()
        process.join()

        assert np.all(shared_arrays["test"] == 100)

    @classmethod
    def run_in_process(cls, array_spec):
        shared_arrays = SharedArrays([array_spec])
        shared_arrays["test"][:, :] = 100
