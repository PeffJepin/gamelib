# TODO: In the future it would be great if the buffer interface was an
#  extension of np.ndarray directly with added support for writing to and
#  reading from an internal OpenGL buffer object.

import numpy as np

from typing import Callable

import gamelib
from gamelib.core import gl


class Buffer:
    """A wrapper around an OpenGL buffer object."""

    def __init__(self, data, dtype=None, dynamic=False):
        """Initialize a buffer object from a ndarray or bytes.

        Parameters
        ----------
        data : np.ndarray | bytes
        dtype : np.dtype
        dynamic : bool
        """

        if isinstance(dtype, str):
            getattr(gl, dtype)
        self.gl = None
        self._dynamic = dynamic
        self.dtype = dtype or data.dtype
        self.write(data)

    def __len__(self):
        """How many elements in this buffer."""

        return self.size // self.dtype.itemsize

    @property
    def size(self):
        """The size of the buffer in bytes."""

        if self.gl is None:
            return 0
        return self.gl.size

    def write(self, data):
        """Write provided data into the buffer. If data is provided as a
        ndarray, it gets converted to self.dtype.

        This will release the old buffer and create a new one if the number
        of bytes provided changes.

        Parameters
        ----------
        data : np.ndarray | bytes
        """

        if isinstance(data, Callable):
            data = data()
            assert isinstance(data, np.ndarray)

        if isinstance(data, np.ndarray):
            self._write_array(data)
        elif isinstance(data, bytes):
            self._write_bytes(data)

    def read(self, *, bytes=False):
        """Reads the data from the OpenGL buffer object.

        Parameters
        ----------
        bytes : bool, optional
            Flag to return bytes instead of ndarray.

        Returns
        -------
        bytes | np.ndarray
            Depending on `bytes` flag. The array will have the same datatype
            given to this buffer.
        """

        if bytes:
            return self.gl.read()
        return np.frombuffer(self.gl.read(), self.dtype)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(len={len(self)}, "
            f"size={self.size} dtype={self.dtype!r})>"
        )

    def _write_array(self, array, copy=True):
        if copy:
            array = array.copy()
        self._write_bytes(gl.coerce_array(array, self.dtype).tobytes())

    def _write_bytes(self, data):
        if self.gl is None or self.size != len(data):
            self._make_opengl_buffer(len(data))
        if self.gl is None:
            return
        self.gl.write(data)

    def _make_opengl_buffer(self, nbytes):
        nbytes = int(nbytes)
        if self.gl is not None:
            self.gl.release()
        if nbytes == 0:
            self.gl = None
            return
        self.gl = gamelib.get_context().buffer(
            reserve=nbytes, dynamic=self._dynamic
        )


class AutoBuffer(Buffer):
    """A buffer than reads in data automatically from a source array and
    allocates itself dynamically as the size of is dataset changes."""

    def __init__(self, source, dtype, *, shrink=True):
        """
        Parameters
        ----------
        source : np.ndarray | callable
            The source can either be an array to read from continually or a
            callable that will provide access on demand to such an array.
        dtype : np.dtype
        shrink : optional, bool
            Should this buffer automatically shrink?
        """

        self.dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        self._source = source
        self._shrink = shrink
        super().__init__(self._source, dtype, True)
        self._num_elements = self._source.nbytes // self.dtype.itemsize

    @property
    def _source(self):
        """Internal to get the array depending on how it was provided.

        Returns
        -------
        ndarray:
            Converted to the correct dtype.
        """

        if self._using_array_proxy:
            return gl.coerce_array(self._array_source(), self.dtype)
        return self._array_source

    @_source.setter
    def _source(self, source):
        """Sets the data source.

        Parameters
        ----------
        source : np.ndarray | Callable
        """

        if isinstance(source, np.ndarray):
            self._using_array_proxy = False
            self._array_source = gl.coerce_array(source, self.dtype)
        else:
            self._using_array_proxy = True
            self._array_source = source

    def __len__(self):
        """The length of an AutoBuffer is equal to the number of elements
        that are written into it. The capacity is unrelated.

        Returns
        -------
        int
        """

        return self._num_elements

    def use_array(self, array):
        """Use the given array as the new source for this AutoBuffer.

        Parameters
        ----------
        array : np.ndarray
        """

        self._source = array
        self.update()

    def update(self):
        """Update the contents of the OpenGL buffer with the source array."""

        self.write(self._source)

    def read(self, bytes=False):
        """Unlike in Buffer, AutoBuffer will only read back what has been
        written into the buffer. Meaning if the buffer has spare space, no
        garbage data will be returned."""

        data = super().read(bytes=bytes)
        if bytes:
            return data[: self._num_elements * self.dtype.itemsize]
        else:
            return data[: self._num_elements]

    def _write_array(self, array):
        super()._write_array(array, copy=False)

    def _write_bytes(self, data):
        nbytes = len(data)
        self._num_elements = nbytes // self.dtype.itemsize

        if self.gl is None or nbytes > self.size:
            self._make_opengl_buffer(nbytes)
        if self._shrink and nbytes < self.size / 4:
            self._make_opengl_buffer(nbytes)
        if self.gl is None:
            return

        self.gl.write(data)

    def _make_opengl_buffer(self, nbytes):
        super()._make_opengl_buffer(nbytes * 1.5)
