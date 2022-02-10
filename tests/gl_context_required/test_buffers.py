import pytest
import numpy as np

import gamelib
from gamelib.core import gl
from gamelib.rendering import buffers


@pytest.fixture(autouse=True, scope="module")
def init_opengl_context():
    gamelib.init(headless=True)


class FakeLock:
    def __init__(self):
        self.times_used = 0

    def __enter__(self):
        self.times_used += 1

    def __exit__(self, *args):
        pass


class TestBuffer:
    def test_initializing_a_buffer_with_an_array(self):
        array = np.arange(10, dtype=gl.float)
        buffer = buffers.Buffer(array, gl.float)

        assert buffer.dtype == gl.float
        assert buffer.size == array.nbytes
        assert np.all(buffer.read() == array)

    def test_initializing_a_buffer_with_bytes(self):
        array = np.arange(10, dtype=gl.float)
        buffer = buffers.Buffer(array.tobytes(), gl.float)

        assert buffer.dtype == gl.float
        assert buffer.size == array.nbytes
        assert np.all(buffer.read() == array)

        with pytest.raises(Exception):
            # must supply dtype when using bytes
            buffer = buffers.Buffer(array.tobytes())

    def test_initializing_a_buffer_with_a_callable(self):
        array = np.arange(10, dtype=gl.float)
        proxy = lambda: array
        buffer = buffers.Buffer(proxy, gl.float)

        assert buffer.dtype == gl.float
        assert buffer.size == array.nbytes
        assert np.all(buffer.read() == array)

        with pytest.raises(Exception):
            # for now callable must point to ndarray, may change in future.
            buffer = buffers.Buffer(lambda: (1, 2, 3), gl.float)

    def test_writing_new_array_of_the_same_size(self):
        array = np.arange(10, dtype=gl.float)
        buffer = buffers.Buffer(array, gl.float)
        glo = buffer.gl

        array += 1
        buffer.write(array)

        assert buffer.gl is glo
        assert np.all(buffer.read() == array)

    def test_writing_new_bytes_of_the_same_size(self):
        array = np.arange(10, dtype=gl.float)
        buffer = buffers.Buffer(array, gl.float)
        glo = buffer.gl

        array += 1
        buffer.write(array.tobytes())

        assert buffer.gl is glo
        assert np.all(buffer.read() == array)

    def test_writing_smaller_array(self):
        array = np.arange(10, dtype=gl.float)
        buffer = buffers.Buffer(array, gl.float)
        glo = buffer.gl

        array = np.arange(5, dtype=gl.float)
        buffer.write(array)

        assert glo is not buffer.gl
        assert np.all(buffer.read() == array)

    def test_writing_smaller_bytes(self):
        array = np.arange(10, dtype=gl.float)
        buffer = buffers.Buffer(array, gl.float)
        glo = buffer.gl

        array = np.arange(5, dtype=gl.float)
        buffer.write(array.tobytes())

        assert glo is not buffer.gl
        assert np.all(buffer.read() == array)

    def test_writing_larger_array(self):
        array = np.arange(10, dtype=gl.float)
        buffer = buffers.Buffer(array, gl.float)
        glo = buffer.gl

        array = np.arange(15, dtype=gl.float)
        buffer.write(array)

        assert glo is not buffer.gl
        assert np.all(buffer.read() == array)

    def test_writing_larger_bytes(self):
        array = np.arange(10, dtype=gl.float)
        buffer = buffers.Buffer(array, gl.float)
        glo = buffer.gl

        array = np.arange(15, dtype=gl.float)
        buffer.write(array.tobytes())

        assert glo is not buffer.gl
        assert np.all(buffer.read() == array)

    def test_writing_a_buffer_with_empty_array(self):
        array = np.arange(10, dtype=gl.float)
        buffer = buffers.Buffer(array, gl.float)

        array = np.array([], gl.float)
        buffer.write(array)

        assert len(buffer) == 0

    def test_dtype_conversion_on_init(self):
        array = np.arange(10)
        buffer = buffers.Buffer(array, dtype=gl.float)

        assert array.astype(gl.float).tobytes() == buffer.read(bytes=True)

    def test_dtype_conversion_on_write(self):
        array1 = np.arange(10, dtype=gl.float)
        array2 = np.arange(10)
        buffer = buffers.Buffer(array1, gl.float)

        buffer.write(array2)

        assert array2.astype(gl.float).tobytes() == buffer.read(bytes=True)

    @pytest.mark.parametrize(
        "gl_type, expected",
        (
            (gl.float, 12),
            (gl.vec2, 6),
            (gl.vec3, 4),
            (gl.vec4, 3),
            (gl.mat3x2, 2),
            (gl.mat3x4, 1),
        ),
    )
    def test_length(self, gl_type, expected):
        array = np.arange(12)

        buffer = buffers.Buffer(array, gl_type)
        assert len(buffer) == expected


class TestAutoBuffer:
    def test_allocates_more_space_than_is_necessary(self):
        array = np.arange(10)
        buffer = buffers.AutoBuffer(array, gl.byte)

        assert buffer.size > 10

    def test_does_not_create_new_buffer_for_small_changes(self):
        array = np.arange(10)
        buffer = buffers.AutoBuffer(array, gl.byte)
        glo = buffer.gl

        buffer.write(np.arange(11))

        assert buffer.gl is glo

    def test_allocates_more_space_when_limit_is_exceeded(self):
        array = np.arange(10)
        buffer = buffers.AutoBuffer(array, gl.byte)

        new_size = buffer.size + 10
        buffer.write(np.arange(new_size))

        assert buffer.size > new_size

    def test_does_not_allocate_extra_space_until_limit_is_exceeded(self):
        array = np.arange(10)
        buffer = buffers.AutoBuffer(array, gl.byte)
        glo = buffer.gl

        new_size = buffer.size
        buffer.write(np.arange(new_size))

        assert buffer.gl is glo
        assert buffer.size == new_size

    def test_creates_a_new_smaller_buffer_when_below_25_percent_capacity(self):
        array = np.arange(10)
        buffer = buffers.AutoBuffer(array, gl.byte)
        start_size = buffer.size
        glo = buffer.gl

        new_size = int(buffer.size * 0.25) - 1
        buffer.write(np.arange(new_size))

        assert glo is not buffer.gl
        assert buffer.size < start_size

    def test_read_does_not_read_unused_space(self):
        array = np.arange(10, dtype=gl.byte)
        buffer = buffers.AutoBuffer(array, gl.byte)

        assert np.all(buffer.read() == array)

    def test_read_returns_correct_size_after_a_write(self):
        array = np.arange(10, dtype=gl.byte)
        buffer = buffers.AutoBuffer(array, gl.byte)

        array = np.arange(5, dtype=gl.byte)
        buffer.write(array)
        assert np.all(buffer.read() == array)

        array = np.arange(3, dtype=gl.byte)
        buffer.write(array.tobytes())
        assert np.all(buffer.read() == array)

    def test_takes_updates_from_a_given_source_array(self):
        array = np.zeros(10, gl.float)
        buffer = buffers.AutoBuffer(array, gl.float)

        array += 111
        buffer.update()

        assert np.all(buffer.read() == array)

    def test_can_be_sourced_with_a_callable_as_an_array_proxy(self):
        array = np.zeros(10, gl.float)
        buffer = buffers.AutoBuffer(lambda: array, gl.float)

        array += 111
        buffer.update()

        assert np.all(buffer.read() == array)

    def test_length_keeps_track_of_elements_in_buffer(self):
        array = np.arange(10)
        buffer = buffers.AutoBuffer(array, gl.int)
        assert len(buffer) == 10

        buffer.write(np.arange(5))
        assert len(buffer) == 5

    def test_setting_the_source_array(self):
        array1 = np.zeros(10, gl.float)
        buffer = buffers.AutoBuffer(array1, gl.float)

        array2 = np.arange(10, dtype=gl.float)
        buffer.use_array(array2)
        assert np.all(buffer.read() == array2)

        array2 += 123
        buffer.update()
        assert np.all(buffer.read() == array2)
