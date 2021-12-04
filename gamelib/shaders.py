import pathlib
from typing import NamedTuple, List

import numpy as np

from . import gl
from . import resources


class AutoBuffer:
    """A wrapper around moderngl.Buffer

    Offers up the moderngl interface while adding direct support
    for working with numpy ndarrays.
    """

    def __init__(
        self,
        source=None,
        dtype=None,
        max_elements=-1,
        *,
        lock=None,
        ctx=None,
    ):
        """Initialize the buffer. If a `source` ndarray isn't given, then
        `dtype` and `max_elements` must instead be given in order to specify
        the structure of the buffer.

        Parameters
        ----------
        source : np.ndarray, optional
            The buffer will be updated from this source before being used
            for rendering/transforms.
        dtype :  np.dtype | str, optional
            Should probably be a dtype defined in the `gl.py` module, though
            can work with and numpy compatible dtype string.
        max_elements : int
            Used to determine buffer reserve size. The buffer has space for
            this many instances of the buffers dtype.
        lock : Lock | None
            Really can be any context manager. Used when accessing the
            internal array if provided.
        ctx : moderngl.Context | None
            A reference to the context should be provided when if there
            is no global value to pull from.
        """

        if source is None:
            assert max_elements > 0 and dtype

        self._num_elements = 0
        self._max_elements = max_elements
        self._ctx = ctx or gl.context
        self._lock = lock
        self._array = source

        if isinstance(dtype, str):
            try:
                dtype = getattr(gl, dtype)
            except AttributeError:
                # fallback to interpreting as np.dtype
                dtype = np.dtype(dtype)
        self._gl_dtype = dtype or source.dtype

        if self._max_elements > 0:
            reserve = self._gl_dtype.itemsize * self._max_elements
        else:

            reserve = gl.coerce_array(source, self._gl_dtype).nbytes
        self.gl = self._ctx.buffer(dynamic=True, reserve=reserve)
        self.update()

    def __len__(self):
        """The length of the buffer is determined by what was last written
        into it. Not to be confused with `size`.

        Returns
        -------
        int
        """

        return self._num_elements

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(len={len(self)}, "
            f"size={self.size} dtype={self._gl_dtype!r})>"
        )

    @property
    def element_size(self):
        """Size in bytes of a single element belonging to this buffer."""

        return self._gl_dtype.itemsize

    @property
    def size(self):
        """The full size of the buffer, regardless of how much was written
        into it."""

        return self.gl.size

    def use_array(self, array):
        """Use the given array as the new source for this AutoBuffer.

        Parameters
        ----------
        array : np.ndarray
        """

        self._array = array
        self.update()

    def update(self):
        """Update the contents of the OpenGL buffer with the source array."""

        if self._array is None:
            return
        self.write(self._array)

    def write(self, data):
        """Write provided data into the buffer. If data is provided as an
        ndarray, the array will be coerced into the correct datatype / shape.

        Parameters
        ----------
        data : np.ndarray | bytes
        """

        if isinstance(data, np.ndarray):
            self._write_array(data)
        elif isinstance(data, bytes):
            self._write_bytes(data)

    def read(self, *, bytes=False):
        """Reads the data from the opengl buffer.

        Parameters
        ----------
        bytes : bool, optional
            Flag to return bytes instead of ndarray.

        Returns
        -------
        bytes | np.ndarray
            Depending on `bytes` flag. The array will have the same datatype
            given to this buffer.

        Raises
        ------
        ValueError:
            If the buffer isn't sourced with an ndarray either bytes flag or
            dtype params must be used in order to read back the data.
        """

        nbytes = self._gl_dtype.itemsize * self._num_elements
        if bytes:
            return self.gl.read(size=nbytes)
        if self._gl_dtype is None:
            raise ValueError(
                "This buffer wasn't last written with a detectable dtype "
                "either specify one or use option `bytes=True`"
            )
        return np.frombuffer(self.gl.read(size=nbytes), self._gl_dtype)

    def _write_array(self, array):
        if self._lock:
            with self._lock:
                array = gl.coerce_array(array, self._gl_dtype)
                self._write_bytes(array.tobytes())
        else:
            array = gl.coerce_array(array, self._gl_dtype)
            self._write_bytes(array.tobytes())

    def _write_bytes(self, data):
        if len(data) > self.size:
            raise MemoryError(f"{len(data)} bytes too large for {self!r}.")
        self.gl.write(data)
        self._num_elements = len(data) // self._gl_dtype.itemsize


class OrderedIndexBuffer(AutoBuffer):
    """Simple extension to AutoBuffer that manages a repeating order of
    indices.
    """

    def __init__(self, order, num_entities=0, max_entities=1000, **kwargs):
        """Initialize the buffer.

        Parameters
        ----------
        order : iterable[int]
            The index order that should be repeated for each entity.
            For instance, if you were rendering a bunch of individual quads
            with vertices ordered like so:
                1-2
                |/|
                0-3
            Then order might be given as:
                (0, 1, 2, 0, 2, 3)

        num_entities : int
            How many repetitions the buffer should currently represent.
        """

        self._num_entities = 0
        self.indices_per_entity = len(order)
        order = np.array(order, "u4")
        index_offsets = np.array(np.arange(max_entities) * (max(order) + 1))
        index_offsets = np.repeat(index_offsets, len(order))
        indices = np.tile(order, max_entities) + index_offsets
        super().__init__(
            indices,
            "u4",
            max_elements=max_entities * self.indices_per_entity,
            **kwargs,
        )
        self.num_entities = num_entities

    @property
    def num_entities(self):
        """Returns the current value for num_entities.

        Returns
        -------
        int
        """

        return self._num_entities

    @num_entities.setter
    def num_entities(self, value):
        """Sets the number of entities (number of repetitions of the initial
        ordering) and writes the new indices to the buffer.

        Parameters
        ----------
        value : int
        """

        if self._num_entities == value:
            return
        self._num_entities = value
        stop = self.indices_per_entity * self._num_entities
        self.write(self._array[:stop])

    def update(self):
        # This buffer is updated only when you change `num_entities`
        pass


class _AutoUniform:
    """Helper class for ShaderProgram to keep track of uniform sources."""

    def __init__(self, array, dtype, name):
        """
        Parameters
        ----------
        array : np.ndarray
        dtype : np.dtype | str
        name : str
        """
        self.array = array
        self.dtype = dtype
        self.name = name

    def update(self, prog):
        prog[self.name].write(self._data)

    @property
    def _data(self):
        return gl.coerce_array(self.array, self.dtype).tobytes()


class ShaderProgram:
    """An object that encapsulates the process of rendering with the OpenGL
    programmable pipeline.

    This object will handle assigning buffers to vertex attribute, uniforms,
    etc. This aims to be quick and easy to get working. If the GPU is
    the bottleneck for your program, you'd probably be better off using
    the moderngl API directly for maximum performance.
    """

    def __init__(
        self,
        name=None,
        *,
        ctx=None,
        mode=None,
        varyings=(),
        uniforms=None,
        buffers=None,
        index_buffer=None,
        max_entities=-1,
        **shader_sources,
    ):
        """Initialize a shader from source.

        Parameters
        ----------
        name : str
            Tries to find the source files by name. See resources.py
            for more detailed information on resource discovery.

        ctx : moderngl.Context
            OpenGL context to use. Must be passed in if there is no
            global context such as from gamelib.init().

        mode : int
            Rendering mode constant. Default is TRIANGLES

        varyings : Iterable[str]
            Outputs to be captured by transform feedback. In the future
            this will probably belong to another class of ShaderProgram.

        uniforms : dict[str, np.ndarray]
            The keys should map to uniform names belonging to this shader.
            The values can be standard python values, which will be assigned
            to the shader once, or ndarrays which will act as sources to the
            uniform an be auto-updated. See the moderngl docs for more info
            on formatting single use python values.

        buffers : dict[str, np.ndarray | AutoBuffer]
            Like uniforms, the keys should map to names of vertex attributes
            in the vertex shader. The values should be ndarray which will be
            used to source the OpenGL buffers before using the program.
            VertexBuffers could be used directly, otherwise one will be made
            internally.

        index_buffer : OrderedIndexBuffer, optional
            Used to create an OpenGL index buffer object for rendering.

        **shader_sources : str | pathlib.Path
            Keys must be one of:
                "vertex_shader", "tess_control_shader",
                "tess_evaluation_shader", "geometry_shader",
                "fragment_shader"
            Values can either be the actual source code as a python string
            or should be a path pointing to an appropriate file.

        Examples
        --------
        Here is a simple example. There are many more available
        in the tests/gl_context_required/ directory.

        >>> program = ShaderProgram(
        ...     vertex_shader='''
        ...         #version 330
        ...         in vec2 v_pos;
        ...         in vec3 v_col;
        ...         uniform vec2 offset;
        ...         out vec3 color;
        ...
        ...         void main()
        ...         {
        ...             gl_Position = vec4(v_pos + offset, 0, 1);
        ...             color = v_col;
        ...         }
        ...     ''',
        ...     fragment_shader='''
        ...         #version 330
        ...         in vec3 color;
        ...         out vec4 frag;
        ...
        ...         void main()
        ...         {
        ...             frag = vec4(color, 1);
        ...         }
        ...     ''',
        ...     uniforms={"offset": np.array([0.1, -0.1])},
        ...     buffers={"v_pos": np.array([(-1, -1), (0, 1), (1, -1)]),
        ...              "v_col": np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])}
        ... )

        Notes
        -----
        The moderngl api can be accessed through the `gl` attribute.

        https://moderngl.readthedocs.io/en/latest/reference/program.html
        """

        # set some initial values
        ctx = ctx or gl.context
        buffers = buffers or {}
        uniforms = uniforms or {}
        self._mode = mode
        self._vao = None
        self._max_elements = max_entities
        self._varyings = varyings
        self._foreign_auto_buffers = {}
        self._created_auto_buffers = {}

        # load source, either by given name or kwargs
        if name:
            shader_sources = _load_shader_sources_by_name(name)
        else:
            for shader_type, src in shader_sources.items():
                if isinstance(src, str):
                    if not src.strip().startswith("#version"):
                        shader_sources[shader_type] = _read_shader_source(src)
                elif isinstance(src, pathlib.Path):
                    shader_sources[shader_type] = _read_shader_source(src)
        self.gl = ctx.program(**shader_sources, varyings=varyings)

        # parse shader source code and cache metadata
        self._meta = {
            name: parse_source(src) for name, src in shader_sources.items()
        }
        self._vertex_attrs = {
            i.name: i for i in self._meta["vertex_shader"].inputs
        }
        self._uniforms = dict()
        for meta in self._meta.values():
            self._uniforms.update({u.name: u for u in meta.uniforms})

        # set index buffer
        if isinstance(index_buffer, np.ndarray):
            self._index_buffer = AutoBuffer(index_buffer, gl.uint)
        elif isinstance(index_buffer, OrderedIndexBuffer):
            self._max_elements *= index_buffer.indices_per_entity
            self._index_buffer = index_buffer
        else:
            self._index_buffer = index_buffer

        # process vertex buffers
        self._buffer_format_tuples = []
        for name, buf in buffers.items():
            self._format_buffer(name, buf)
        self._auto_uniforms = {}
        for name, uni in uniforms.items():
            self._format_uniform(name, uni)

        # make vertex array for rendering
        self._make_vao()

    @property
    def vertex_attributes(self):
        """Gets a dictionary of vertex attributes parsed from
        the vertex shader source code.

        Returns
        -------
        dict[str, gl.TokenDesc]
        """
        return self._vertex_attrs

    @property
    def uniforms(self):
        """Gets a dictionary of uniforms parsed from all of the
        shader source strings.

        Returns
        -------
        dict[str, gl.TokenDesc]
        """
        return self._uniforms

    @property
    def _autobuffers(self):
        return {**self._created_auto_buffers, **self._foreign_auto_buffers}

    @property
    def num_elements(self):
        """The number of elements that are detected to be in the current
        buffers.

        Returns
        -------
        int
        """

        if self._index_buffer:
            return len(self._index_buffer)
        else:
            lengths = [len(vbo) for vbo in self._autobuffers.values()]
            return min(lengths)

    def use_buffers(self, **buffers):
        """Assign given arrays to be AutoBuffer sources. Creates new buffers
        if they don't already exist but tries first to update existing
        buffers to use the new source.

        Parameters
        ----------
        buffers : np.ndarray
            Keywords should correspond to vertex attribute names.
        """

        for name, array in buffers.items():
            self._replace_buffer(name, array)
        self._vao = self._make_vao()

    def write_buffers(self, **buffers):
        for name, array in buffers.items():
            self._autobuffers[name].write(array)

    def use_uniforms(self, **uniforms):
        """Use given values for uniform sources. Will replace existing
        sources if one is already being used.

        Parameters
        ----------
        **uniforms: **np.ndarray
        """
        for name, uniform in uniforms.items():
            self._format_uniform(name, uniform)

    def render(self, vertices=None):
        """Writes data into buffers from given ndarray sources and issues
        render command.

        Parameters
        ----------
        vertices : int, optional
            How many vertices to render. If not given the value will be
            detected based on linked buffers.
        """

        self._auto_update()
        self._vao.render(vertices=vertices or self.num_elements)

    def transform(self, vertices=None):
        """Use this program for transform feedback.

        Parameters
        ----------
        vertices : int, optional
            How many vertices to process. If not given the value will be
            detected based on what's written into the linked buffers.

        Returns
        -------
        np.ndarray
            The transform feedback buffer will be read into a np.ndarray
            If there is only one varying attribute being transformed, the
            array will be a standard, unstructured array with the dtype
            of the captured attribute. Otherwise this will return a structured
            ndarray. See tests for concrete examples.
        """

        self._auto_update()
        out_dtype = np.dtype(
            [
                (o.name, o.dtype)
                for o in self._meta["vertex_shader"].outputs
                if o.name in self._varyings
            ]
        )

        if vertices:
            reserve = out_dtype.itemsize * vertices
        else:
            vertices = self.num_elements
            reserve = vertices * out_dtype.itemsize

        result_buffer = self.gl.ctx.buffer(reserve=reserve)
        self._vao.transform(result_buffer, vertices=vertices)
        array = np.frombuffer(result_buffer.read(), out_dtype)
        if len(self._varyings) == 1:
            return array[self._varyings[0]]
        return array

    def _auto_update(self):
        """Update AutoBuffers and _AutoUniforms."""

        for name, buf in self._autobuffers.items():
            buf.update()
        for uni in self._auto_uniforms.values():
            uni.update(self.gl)

    def _format_buffer(self, name, buf):
        """Given a name and data source, create a new AutoBuffer or register
        a new one given as a parameter."""

        dtype = self.vertex_attributes[name].dtype
        if isinstance(buf, np.ndarray):
            auto = AutoBuffer(
                buf, dtype, max_elements=self._max_elements, ctx=self.gl.ctx
            )
            self._created_auto_buffers[name] = auto
            vbo = auto.gl

        elif isinstance(buf, AutoBuffer):
            if not buf.size >= self._max_elements * dtype.itemsize:
                raise MemoryError(
                    f"{buf!r} is too large for this program "
                    f"with max_entities={self._max_elements}."
                )
            self._foreign_auto_buffers[name] = buf
            vbo = buf.gl

        else:
            raise ValueError(
                "Expected buffer source to be either ndarray"
                f"or VertexBuffer. Instead got {type(buf)}"
            )

        moderngl_attr = self.gl[name]
        strtype = moderngl_attr.shape
        if strtype == "I":
            # conform to moderngl expected strfmt dtypes
            # eventually I'd like to move towards doing
            # all the shader source code inspection myself,
            # as the moderngl api doesn't offer all the
            # metadata I would like it to and weird issues
            # like this one.
            strtype = "u"
        strfmt = f"{moderngl_attr.dimension}{strtype}"
        self._buffer_format_tuples.append((vbo, strfmt, name))

    def _format_uniform(self, name, value):
        """Process a uniform input value to either be written as the value
        of the uniform or used as an _AutoUniform source."""

        if isinstance(value, np.ndarray):
            dtype = self.uniforms[name].dtype
            self._auto_uniforms[name] = _AutoUniform(value, dtype, name)
        else:
            self.gl[name] = value

    def _remove_buffer(self, name):
        """Clean-up a buffer that is no longer needed."""

        for i, desc in enumerate(self._buffer_format_tuples.copy()):
            if desc[2] == name:
                self._buffer_format_tuples.pop(i)
        if auto := self._created_auto_buffers.pop(name, None):
            auto.gl.release()
        else:
            self._foreign_auto_buffers.pop(name, None)

    def _replace_buffer(self, name, array):
        """Try to replace an existing buffer before making a new one."""

        try:
            auto = self._created_auto_buffers.get(name, None)
            auto = auto or self._foreign_auto_buffers[name]
            auto.use_array(array)
        except (MemoryError, KeyError):
            self._remove_buffer(name)
            self._format_buffer(name, array)

    def _make_vao(self):
        """Create the moderngl.VertexArray object used to render."""

        if self._vao is not None:
            self._vao.release()

        if self._index_buffer is not None:
            ibo = self._index_buffer.gl
            element_size = self._index_buffer.element_size
        else:
            ibo = None
            element_size = 4

        vao = self.gl.ctx.vertex_array(
            self.gl,
            self._buffer_format_tuples,
            index_buffer=ibo,
            index_element_size=element_size,
        )
        self._vao = vao
        return vao


class TokenDesc(NamedTuple):
    """Data describing a token parsed from glsl source code."""

    name: str
    dtype: np.dtype
    len: int  # array length if token is an array, else 1


class ShaderMetaData(NamedTuple):
    """A collection of metadata on tokens parsed from a single
    glsl source file."""

    inputs: List[TokenDesc]
    outputs: List[TokenDesc]
    uniforms: List[TokenDesc]


def parse_source(src):
    """Parse glsl source code for relevant metadata.

    Parameters
    ----------
    src : str
        glsl source code

    Returns
    -------
    ShaderMetaData
    """
    inspections = {"in": [], "out": [], "uniform": []}
    for line in src.split("\n"):
        tokens = line.strip().split(" ")
        for i, token in enumerate(tokens):
            if token in ("in", "out", "uniform"):
                raw_name = tokens[i + 2][:-1]
                if raw_name.endswith("]"):
                    length = int(raw_name[-2])
                    name = raw_name[:-3]
                else:
                    length = 1
                    name = raw_name
                dtype = getattr(gl, tokens[i + 1])
                assert isinstance(dtype, np.dtype)
                inspections[token].append(
                    TokenDesc(name=name, dtype=dtype, len=length)
                )
                break
    return ShaderMetaData(
        inputs=inspections["in"],
        outputs=inspections["out"],
        uniforms=inspections["uniform"],
    )


def _read_shader_source(path):
    with open(path, "r") as f:
        return f.read()


_EXT_MAPPING = {
    ".vert": "vertex_shader",
    ".frag": "fragment_shader",
    ".tese": "tess_evaluation_shader",
    ".tesc": "tess_control_shader",
    ".geom": "geometry_shader",
}


def _load_shader_sources_by_name(name):
    shader_files = resources.find_shader(name)
    return {
        _EXT_MAPPING[p.name[-5:]]: _read_shader_source(p) for p in shader_files
    }
