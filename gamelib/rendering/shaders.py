import pathlib

import numpy as np

import gamelib
from gamelib import gl
from gamelib.rendering import glslutils
from gamelib.rendering import buffers


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
        **shader_code,
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
            uniform and be auto-updated. See the moderngl docs for more info
            on formatting single use python values.

        buffers : dict[str, np.ndarray | buffers.Buffer]
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
        ctx = ctx or gamelib.get_context()
        buffers = buffers or dict()
        uniforms = uniforms or dict()
        self._initialized = False
        self._mode = mode
        self.__vao = None
        self._vao_dirty = True
        self._max_elements = max_entities
        self._max_element_multiplier = 1
        self._varyings = varyings
        self._foreign_auto_buffers = dict()
        self._created_auto_buffers = dict()
        self._buffers_in_use = dict()

        # load source, either by given name or kwargs
        if name:
            shader_data = glslutils.ShaderData.read_file(name)
        elif source := shader_code.get("source", None):
            shader_data = glslutils.ShaderData.read_string(source)
        else:
            shader_data = glslutils.ShaderData.read_strings(**shader_code)

        self.gl = ctx.program(
            vertex_shader=shader_data.code.vert,
            tess_control_shader=shader_data.code.tesc,
            tess_evaluation_shader=shader_data.code.tese,
            geometry_shader=shader_data.code.geom,
            fragment_shader=shader_data.code.frag,
            varyings=varyings,
        )

        # parse shader source code and cache metadata
        self._meta = shader_data.meta
        self._vertex_attrs = {
            desc.name: desc for desc in self._meta.attributes
        }
        self._uniforms = {desc.name: desc for desc in self._meta.uniforms}

        # set index buffer
        self.use_indices(index_buffer)

        # process vertex buffers
        self._buffer_format_tuples = []
        for name, buf in buffers.items():
            self._format_buffer(name, buf)
        self._auto_uniforms = {}
        for name, uni in uniforms.items():
            self._format_uniform(name, uni)

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
        """Gets a dictionary of uniforms parsed from all the shader source
        strings.

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

    @property
    def _vao(self):
        if self._vao_dirty:
            self.__vao = self._make_vao()
        return self.__vao

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
            auto = (
                self._created_auto_buffers.get(name, None) or
                self._foreign_auto_buffers.get(name, None)
            )
            if not auto:
                self._format_buffer(name, array)
            else:
                auto.use_array(array)
        self._vao_dirty = True

    def use_indices(self, index_buffer):
        if isinstance(index_buffer, np.ndarray):
            self._index_buffer = buffers.AutoBuffer(index_buffer, gl.uint)
        elif isinstance(index_buffer, buffers.OrderedIndexBuffer):
            self._max_element_multiplier = index_buffer.indices_per_entity
            self._index_buffer = index_buffer
        else:
            self._index_buffer = index_buffer
        self._vao_dirty = True

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
                (desc.name, desc.dtype)
                for desc in self._meta.vertex_outputs
                if desc.name in self._varyings
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
            buffer = buffers.AutoBuffer(buf, dtype)
            self._created_auto_buffers[name] = buffer
        elif isinstance(buf, buffers.AutoBuffer):
            if not buf.size >= self._max_elements * dtype.itemsize:
                raise MemoryError(
                    f"{buf!r} is too large for this program "
                    f"with max_entities={self._max_elements}."
                )
            self._foreign_auto_buffers[name] = buf
            buffer = buf
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
        self._buffers_in_use[name] = buffer

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

    def _make_vao(self):
        """Create the moderngl.VertexArray object used to render."""

        if self.__vao is not None:
            self.__vao.release()

        if self._index_buffer is not None:
            ibo = self._index_buffer.gl
            element_size = self._index_buffer.element_size
        else:
            ibo = None
            element_size = 4

        format_tuples = []
        for name, buffer in self._buffers_in_use.items():
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
            format_tuples.append((buffer.gl, strfmt, name))

        self._vao_dirty = False
        return self.gl.ctx.vertex_array(
            self.gl,
            format_tuples,
            index_buffer=ibo,
            index_element_size=element_size,
        )
