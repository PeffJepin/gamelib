import pathlib
from typing import NamedTuple, List

import numpy as np

from . import gl
from . import resources


class IndexBuffer:
    """A simple wrapper around moderngl.Buffer that handles creating
    a repeating ordered sequence of indices for use as an index array
    buffer.
    """

    def __init__(self, order, entities=(), num_entities=0):
        """Initialize the buffer. Either entities or num_entities should
        be used, not both.

        Parameters
        ----------
        order : iterable[int]
            The index order that should be repeated for each entity.
            For instance, if you were rending a bunch of individual quads
            with vertices ordered like so:
                1-2
                |/|
                0-3
            Then order might be given as (0, 1, 2, 0, 2, 3)

        entities : iterable[int]
            Given the example above, if entities=(0, 1, 3)
            then the index buffer should look contain the following:
                (0, 1, 2, 0, 2, 3)
                (4, 5, 6, 4, 6, 7)
                (12, 13, 14, 12, 14, 15)

        num_entities : int
            Shorthand for a range of entities.. The following are equivalent:
                num_entities = n
                entities = range(n)

        Raises
        ------
        ValueError:
            When both entities and num_entities are given.
        """

        if entities and num_entities:
            raise ValueError(
                "Expected to receive either `num_entities` or "
                "`entities` as parameters. Instead got both."
            )
        self._num_entities = num_entities
        self._entities = entities
        self._base = np.array(order, "u4")
        self._buffer = None
        self.dirty = False
        self.element_size = 4

    @property
    def entities(self):
        """Returns the current value for entities.

        Returns
        -------
        iterable[int]
        """
        return self._entities

    @entities.setter
    def entities(self, value):
        """Sets the current value for entities, clears any existing value
        for num_entities and invalidates an existing buffer so that one can
        be made with the new values.

        Parameters
        ----------
        value : iterable[int]
        """

        self._entities = value
        self._num_entities = 0
        self._invalidate_buffer()

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
        """Sets the current value for num entities and clears any existing
        value being used for entities. Invalidates the current buffer if
        one exists so that the new values can be used.

        Parameters
        ----------
        value : int
        """
        self._num_entities = value
        self._entities = ()
        self._invalidate_buffer()

    @property
    def gl(self):
        """Returns the underling moderngl.Buffer object. Will create one
        from the current state if it doesn't already exist.

        Returns
        -------
        moderngl.Buffer
        """
        if self._buffer is not None:
            return self._buffer
        self._create_buffer()
        return self._buffer

    @property
    def indices(self):
        """Returns the current indices. Mostly for debugging.

        Returns
        -------
        np.ndarray
        """
        return np.frombuffer(self.gl.read(), "u4")

    def _create_buffer(self):
        if self.num_entities:
            index_offsets = np.array(
                np.arange(self.num_entities) * (max(self._base) + 1)
            )
            instances = self.num_entities
        else:
            index_offsets = np.array(self.entities) * (max(self._base) + 1)
            instances = len(self.entities)

        index_offsets = np.repeat(index_offsets, len(self._base))
        indices = np.tile(self._base, instances) + index_offsets
        self._buffer = gl.context.buffer(indices.astype("u4").tobytes())

    def _invalidate_buffer(self):
        if self._buffer is not None:
            self._buffer.release()
        self._buffer = None
        self.dirty = True


class VertexBuffer:
    """A wrapper around moderngl.Buffer

    Offers up the moderngl interface while adding direct support
    for working with numpy ndarrays.
    """

    def __init__(
        self,
        source,
        dtype=None,
        *,
        reserve=0,
        lock=None,
        ctx=None,
    ):
        """Initialize the buffer.

        Parameters
        ----------
        source : np.ndarray | moderngl.Buffer
        dtype :  np.dtype | str
            numpy compatible dtype. This will override array dtype if given.
        reserve : int
            number of bytes to reserve
        lock : Lock | None
            Really can be any context manager. Used when writing if provided.
        ctx : moderngl.Context | None
            A reference to the context should be provided when if there
            is no global value to pull from.
        """

        self._ctx = ctx or gl.context
        self._lock = lock
        self._gl_dtype = dtype or source.dtype
        self._array = source
        data = gl.coerce_array(source, self._gl_dtype).tobytes()
        self.gl = self._ctx.buffer(data, dynamic=True, reserve=reserve)

    def update(self):
        """Using a source np.ndarray from __init__, update the buffer."""
        self.write(self._array)

    def write(self, data):
        """Write provided data into the buffer.

        If using data is an ndarray, then it will be converted to
        the correct datatype if it isn't already.

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
        bytes : bool
            Optional flag to return bytes instead of ndarray.

        Returns
        -------
        data : bytes | np.ndarray

        Raises
        ------
        ValueError:
            If the buffer isn't sourced with an ndarray either bytes or
            dtype params must be used in order to read back the data.
        """

        if bytes:
            return self.gl.read()
        if self._gl_dtype is None:
            raise ValueError(
                "This buffer wasn't last written with a detectable dtype "
                "either specify one or use option `bytes=True`"
            )
        return np.frombuffer(self.gl.read(), self._gl_dtype)

    def _write_array(self, array):
        if self._lock:
            with self._lock:
                array = gl.coerce_array(array, self._gl_dtype)
                self.gl.write(array.tobytes())
        else:
            array = gl.coerce_array(array, self._gl_dtype)
            self.gl.write(array.tobytes())

    def _write_bytes(self, data):
        if self._lock:
            with self._lock:
                self.gl.write(data)
        else:
            self.gl.write(data)


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
    the moderngl api directly for maximum performance.
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

        buffers : dict[str, np.ndarray | VertexBuffer]
            Like uniforms, the keys should map to names of vertex attributes
            in the vertex shader. The values should be ndarray which will be
            used to source the OpenGL buffers before using the program.
            VertexBuffers could be used directly, otherwise one will be made
            internally.

        index_buffer : IndexBuffer, optional
            Used to create and OpenGL index buffer object for rendering.

        **shader_sources : ** str | pathlib.Path
            Keys must be one of:
                "vertex_shader", "tess_control_shader",
                "tess_evaluation_shader", "geometry_shader", "fragment_shader"
            Values can either be the actual source code as a standard string
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
        https://moderngl.readthedocs.io/en/latest/reference/program.html
        """

        ctx = ctx or gl.context
        buffers = buffers or {}
        uniforms = uniforms or {}
        self._mode = mode
        self._varyings = varyings
        self._index_buffer = index_buffer
        self._foreign_auto_buffers = {}
        self._created_auto_buffers = {}

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

        self._meta = {
            name: parse_source(src) for name, src in shader_sources.items()
        }
        self._vertex_attrs = {
            i.name: i for i in self._meta["vertex_shader"].inputs
        }
        self._uniforms = dict()
        for meta in self._meta.values():
            self._uniforms.update({u.name: u for u in meta.uniforms})

        self._buffer_format_tuples = []
        for name, buf in buffers.items():
            self._format_buffer(name, buf)
        self._auto_uniforms = {}
        for name, uni in uniforms.items():
            self._format_uniform(name, uni)
        self._vao = self._make_vao()

    @property
    def vertex_attributes(self):
        """Gets a dictionary of vertex attributes parsed from
        the vertex shader source code.

        Returns
        -------
        attrs : dict[str, gl.TokenDesc]
        """
        return self._vertex_attrs

    @property
    def uniforms(self):
        """Gets a dictionary of uniforms parsed from all of the
        shader source strings.

        Returns
        -------
        uniforms : dict[str, gl.TokenDesc]
        """
        return self._uniforms

    def use_buffers(self, **buffers):
        """Use given values for buffer sources. Will remove existing values
        and if the existing OpenGL buffer was created by this object it will
        be released.

        Parameters
        ----------
        **buffers : **np.ndarray
        """
        for name, buffer in buffers.items():
            self._remove_buffer(name)
            self._format_buffer(name, buffer)
        self._vao = self._make_vao()

    def use_uniforms(self, **uniforms):
        """Use given values for uniform sources. Will replace existing
        sources if one is already being used.

        Parameters
        ----------
        **uniforms: **np.ndarray
        """
        for name, uniform in uniforms.items():
            self._format_uniform(name, uniform)

    def render(self, vertices=-1):
        """Writes data into buffers from given ndarray sources and issues
        render command.

        Parameters
        ----------
        vertices : int
            How many vertices to render, -1 will autodetect from the buffers.
        """
        self._auto_update()
        self._vao.render(vertices=vertices)

    def transform(self, vertices=-1):
        """Use this program for transform feedback.

        Parameters
        ----------
        vertices : int
            How many vertices to process. -1 will autodetect from buffers.

        Returns
        -------
        transformed : np.ndarray
            The transform feedback buffer will be read into a np.ndarray
            If there is only one varying attribute being transformed, the
            array will be a standard, unstructured array with the dtype
            of the captured attribute. Otherwise this will return a structured
            ndarray. See tests for concrete examples.
        """
        self._auto_update()
        vertex_meta = self._meta["vertex_shader"]
        out_dtype = np.dtype(
            [
                (o.name, o.dtype)
                for o in vertex_meta.outputs
                if o.name in self._varyings
            ]
        )

        if vertices != -1:
            reserve = out_dtype.itemsize * vertices
        else:
            buf, fmt, name = self._buffer_format_tuples[0]
            buf_dtype = self.vertex_attributes[name].dtype
            nverts = buf.size // buf_dtype.itemsize
            reserve = nverts * out_dtype.itemsize

        result_buffer = self.gl.ctx.buffer(reserve=reserve)
        self._vao.transform(result_buffer, vertices=vertices)
        array = np.frombuffer(result_buffer.read(), out_dtype)
        if len(self._varyings) == 1:
            return array[self._varyings[0]]
        return array

    def _auto_update(self):
        for name, buf in self._foreign_auto_buffers.items():
            buf.update()
        for name, buf in self._created_auto_buffers.items():
            buf.update()
        for uni in self._auto_uniforms.values():
            uni.update(self.gl)
        if self._index_buffer and self._index_buffer.dirty:
            self._vao = self._make_vao()

    def _format_buffer(self, name, buf):
        if isinstance(buf, np.ndarray):
            dtype = self.vertex_attributes[name].dtype
            auto = VertexBuffer(buf, dtype, ctx=self.gl.ctx)
            self._created_auto_buffers[name] = auto
            vbo = auto.gl
        elif isinstance(buf, VertexBuffer):
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
        if isinstance(value, np.ndarray):
            dtype = self.uniforms[name].dtype
            self._auto_uniforms[name] = _AutoUniform(value, dtype, name)
        else:
            self.gl[name] = value

    def _remove_buffer(self, name):
        for i, desc in enumerate(self._buffer_format_tuples.copy()):
            if desc[2] == name:
                self._buffer_format_tuples.pop(i)
        if auto := self._created_auto_buffers.pop(name, None):
            auto.gl.release()
        else:
            self._foreign_auto_buffers.pop(name, None)

    def _make_vao(self):
        try:
            if self._vao is not None:
                self._vao.release()
            ibo = self._index_buffer.gl if self._index_buffer else None
            element_size = self._index_buffer.element_size if ibo else 4
            vao = self.gl.ctx.vertex_array(
                self.gl,
                self._buffer_format_tuples,
                index_buffer=ibo,
                index_element_size=element_size,
            )
            return vao
        except AttributeError:
            self._vao = None
            return self._make_vao()


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
