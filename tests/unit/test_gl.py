import pytest
import numpy as np
from gamelib import gl


@pytest.mark.parametrize(
    "dtype, data_in, data_out",
    [
        ("vec2", np.arange(4), np.array([[0, 1], [2, 3]], "f4")),
        (
            "mat2x3",
            np.arange(12),
            np.array(
                [[(0, 1), (2, 3), (4, 5)], [(6, 7), (8, 9), (10, 11)]], "f4"
            ),
        ),
        ("uint", np.arange(10), np.arange(10, dtype="u4")),
    ],
)
def test_array_coercion(dtype, data_in, data_out):
    assert np.all(gl.coerce_array(data_in, dtype) == data_out)
