import time
import numpy as np

from typing import Callable

import gamelib
from gamelib.core import resources
from gamelib.core import gl
from gamelib.rendering import _global
from gamelib.rendering import uniforms
from gamelib.rendering import shaders
from gamelib.rendering import buffers
from gamelib.rendering import textures

_cached_assets = dict()


class GPUInstructions:
    """Common functions for issuing commands to the gpu."""

    def __init__(
        self,
        shader,
        mode=gl.TRIANGLES,
        instanced=(),
        automatic_hot_reloading=False,
        **data_sources,
    ):
        """Initialize a new instruction.

        Parameters
        ----------
        shader : Any
            This should either be a single source string or a filename for the
            shader to be used. See shaders.py docstring for more info.
        mode : int, optional
            OpenGL constant mode for rendering.
        instanced : tuple, optional
            The names of vertex attributes that should be instanced attributes.
        automatic_hot_reloading : bool
            Should changes in shader files be listened for and shaders reloaded
            automatically when a change is detected?
        **data_sources : Any
            The keys should map to inputs for the specified shader and the
            values can be either np.ndarrays or buffers.Buffer instances.
            Other values will be attempted to be interpreted as ndarrays.
        """

        if isinstance(shader, str) and "#version" in shader:
            self.shader = shaders.Shader(src=shader)
        else:
            self._shader_name = shader
            self.shader = shaders.Shader(self._shader_name)

        self.vao = VertexArray(
            self.shader, mode=mode, instanced=instanced, **data_sources
        )

        if "dt" in self.shader.meta.uniforms:
            self.vao.source_uniforms(dt=self._dt)

        self._instanced = instanced
        self._mode = mode
        self._textures = dict()
        self._fetch_textures()
        self._prev = time.time()
        self._dt = np.zeros(1, gl.float)
        self._hot_reloading = automatic_hot_reloading

    def source(self, **data_sources):
        """Use these data sources."""

        self.vao.use_sources(**data_sources)

    def update(self):
        t = time.time()
        dt = self._prev - t
        self._prev = t
        self._dt[0] = dt
        if self._hot_reloading and self.shader.has_been_modified:
            if self.shader.try_hot_reload():
                self.vao.flag_dirty()
        self.vao.update()

    def _bind_textures(self):
        for binding, texture in self._textures.items():
            texture.gl.use(binding)

    def _fetch_textures(self):
        i = 0
        samplers = dict()
        for sampler in self.shader.meta.samplers:
            asset = _cached_assets.get(sampler.name, None)
            if asset is None:
                path = resources.get_image_file(sampler.name)
                asset = textures.ImageAsset(sampler.name, path)
                _cached_assets[sampler.name] = asset
            if asset.texture is None:
                asset.upload_texture(gamelib.get_context())
            value_wrapper = np.array([i], gamelib.core.gl.sampler2D)
            samplers[sampler.name] = value_wrapper
            self._textures[i] = asset.texture
            i += 1
        self.source(**samplers)


class TransformFeedback(GPUInstructions):
    """Use the GPU to transform some data."""

    def __init__(
        self,
        shader,
        mode=gl.POINTS,
        automatic_hot_reloading=False,
        **data_sources,
    ):
        self.sources = data_sources
        super().__init__(
            shader, mode, automatic_hot_reloading=automatic_hot_reloading
        )

    def source(self, **data_sources):
        self.sources.update(data_sources)

    def transform(self, vertices=None, **data_sources):
        """Issue the transform feedback command. This will block and read back
        the buffer into a np.ndarray.

        Parameters
        ----------
        vertices : optional, int
            How many vertices to transform. If not given the VertexArray will
            try to calculate it based on buffer lengths.
        **data_sources : Any
            If not provided in __init__ the input buffers can be given now.

        Returns
        -------
        np.ndarray:
            This will return a structured array if the shader has multiple
            outputs, otherwise it will be a regular array converted to the
            output datatype.
        """

        self.sources.update(data_sources)
        self.vao = VertexArray(self.shader, **self.sources)
        if self.vao.glo is None:
            return None
        self.update()
        out_dtype = np.dtype(
            [
                (desc.name, desc.dtype)
                for desc in self.shader.meta.vertex_outputs.values()
            ]
        )
        vertices = vertices or self.vao.num_elements
        reserve = vertices * out_dtype.itemsize
        result_buffer = gamelib.get_context().buffer(reserve=reserve)
        self.vao.glo.transform(result_buffer, vertices=vertices)
        array = np.frombuffer(result_buffer.read(), out_dtype)
        if len(self.shader.meta.vertex_outputs) == 1:
            return array[next(iter(self.shader.meta.vertex_outputs.keys()))]
        return array


class Renderer(GPUInstructions):
    """Issues draw calls."""

    def render(self, vertices=None, instances=None):
        """Issue a draw call to OpenGL. The module will try to detect the
        correct number of vertices/indices if they are not given.

        Parameters
        ----------
        vertices : int, optional
            Override the number of vertices to be rendered.
        instances : int, optional
            Override the number of instances to be rendered.
        """

        self.update()
        if self.vao.glo is None:
            return

        self._bind_textures()

        vertices = vertices or self.vao.num_elements
        instances = instances or self.vao.num_instances

        self.vao.glo.render(
            vertices=vertices, instances=instances, mode=self._mode
        )

    def source_indices(self, indices):
        """Shortcut to source an index buffer.

        Parameters
        ----------
        indices : np.ndarray
        """

        self.vao.source_indices(indices)


class VertexArray:
    """Responsible for mapping data on the CPU side to shader inputs on the
    GPU."""

    def __init__(
        self,
        shader,
        instanced=(),
        indices=None,
        auto=True,
        mode=gl.TRIANGLES,
        **data_sources,
    ):
        """Initialize a vertex array.

        Parameters
        ----------
        shader : shaders.Shader
        instanced : optional, Sequence
        indices : optional, np.ndarray | buffers.Buffer
        auto : optional, bool
        mode : optional, int
        **data_sources : Any
            The keys must map to either a buffer or uniform within the
            given shader. The values for buffers can be either np.ndarray or
            buffers.Buffer. The uniforms can be either np.ndarray or python
            values.
        """

        self._glo = None
        self._dirty = True
        self._auto = auto
        self._mode = mode
        self._index_buffer = None
        self._instanced_attibutes = set(instanced)
        self._buffers_in_use = dict()
        self._uniforms_in_use = dict()
        self._buffer_ids = dict()
        self._generated_buffers = list()

        self.shader = shader
        self.use_sources(**data_sources)
        self.source_indices(indices)
        self.check_global_uniforms(**data_sources)

    @property
    def glo(self):
        """The underlying moderngl object."""

        for k, buf in self._buffers_in_use.items():
            if id(buf.gl) != self._buffer_ids[k]:
                self._dirty = True
        if self._glo is None or self._dirty:
            self._make_glo()
        return self._glo

    @property
    def num_elements(self):
        """The number of elements that are detected to be in the current
        buffers. This is first by an index buffer if present, otherwise by
        the smallest buffer.

        Returns
        -------
        int
        """

        if self._index_buffer:
            return len(self._index_buffer)

        lengths = [
            len(vbo)
            for name, vbo in self._buffers_in_use.items()
            if name not in self._instanced_attibutes
        ]
        return min(lengths)

    @property
    def num_instances(self):
        """Autodetect the number of instances that are represented through the
        attached buffers.

        Returns
        -------
        int
        """

        if not self._instanced_attibutes:
            return -1

        lengths = [
            len(vbo)
            for name, vbo in self._buffers_in_use.items()
            if name in self._instanced_attibutes
        ]
        return min(lengths)

    @property
    def _buffer_format_tuples(self):
        """(buffer_obj, buffer_format, buffer_name) formatting tuples."""

        format_tuples = []
        for name, buffer in self._buffers_in_use.items():
            moderngl_attr = self.shader.glo[name]
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
            if name in self._instanced_attibutes:
                strfmt += " /i"
            format_tuples.append((buffer.gl, strfmt, name))
        return format_tuples

    def update(self):
        """Updates _AutoUniform and AutoBuffer objects."""

        for buffer in self._buffers_in_use.values():
            if isinstance(buffer, buffers.AutoBuffer):
                buffer.update()
        for uniform in self._uniforms_in_use.values():
            uniform.update(self.shader.glo)

    def use_source(self, name, source):
        """Set a source uniform/buffer.

        Parameters
        ----------
        name : str
        source : np.ndarray | buffers.Buffer | Any
            If sourcing a buffer, this should be either a np.ndarray or a
            buffers.Buffer.
            If sourcing a uniform this can be a np.ndarray or a python number
            or tuple.
        """

        if name in self.shader.meta.attributes:
            self._integrate_buffer(name, source)
        elif name in self.shader.meta.uniforms:
            self._integrate_uniform(name, source)
        else:
            self._raise_invalid_source(name)

    def use_sources(self, **data_sources):
        """Shorthand for many use_source calls. See use_source"""

        for name, source in data_sources.items():
            self.use_source(name, source)

    def source_buffers(self, **buffer_sources):
        """Set a buffer source.

        Parameters
        ----------
        buffer_sources : buffers.Buffer | np.ndarray
        """

        for name, buffer in buffer_sources.items():
            self._integrate_buffer(name, buffer)
        self._dirty = True

    def source_indices(self, indices):
        """Set the index buffer.

        Parameters
        ----------
        indices : np.ndarray | buffers.Buffer
        """

        if self._index_buffer is not None:
            if isinstance(indices, np.ndarray):
                self._index_buffer.write(indices)
            elif isinstance(indices, buffers.Buffer):
                self._remove_buffer(self._index_buffer)
                self._index_buffer = indices
        else:
            self._index_buffer = self._generate_buffer(
                indices, gl.uint, auto=False
            )
        self._dirty = True

    def source_uniforms(self, **uniform_sources):
        """Source uniform values.

        Parameters
        ----------
        **uniform_sources : np.ndarray | tuple | int | float
            If sourced with a np.ndarray, this uniform will continually be
            updated from that array.
            If sourced from a python value it will set the value just once.
        """

        for name, uniform in uniform_sources.items():
            self._integrate_uniform(name, uniform)
        self._dirty = True

    def check_global_uniforms(self, **data_sources):
        glob = _global.global_uniforms.todict()
        for name in self.shader.meta.uniforms:
            if name not in data_sources and name in glob:
                self._integrate_uniform(name, glob[name])

    def flag_dirty(self):
        self._dirty = True

    def _integrate_buffer(self, attribute, source):
        if attribute not in self.shader.meta.attributes:
            self._raise_invalid_source(attribute)

        current_buffer = self._buffers_in_use.get(attribute, None)
        dtype = self.shader.meta.attributes[attribute].dtype
        if current_buffer is None:
            buffer = self._generate_buffer(source, dtype)
            self._buffers_in_use[attribute] = buffer
            self._buffer_ids[attribute] = id(buffer.gl)
        else:
            if isinstance(source, buffers.Buffer):
                self._remove_buffer(current_buffer)
                self._buffers_in_use[attribute] = source
                self._buffer_ids[attribute] = id(source.gl)
            elif source is not None:
                if not isinstance(source, np.ndarray):
                    source = np.asarray(source, dtype)
                if isinstance(current_buffer, buffers.AutoBuffer):
                    current_buffer.use_array(source)
                else:
                    current_buffer.write(source)

    def _integrate_uniform(self, name, source):
        dtype = self.shader.meta.uniforms[name].dtype

        if not isinstance(source, np.ndarray):
            source = np.array(source, dtype)

        self._uniforms_in_use[name] = uniforms.AutoUniform(source, dtype, name)

    def _generate_buffer(self, source, dtype, auto=None):
        self._dirty = True

        if isinstance(source, buffers.Buffer):
            return source

        auto = auto if auto is not None else self._auto
        buf_type = buffers.AutoBuffer if auto else buffers.Buffer

        if isinstance(source, np.ndarray):
            buf = buf_type(source, dtype)
            self._generated_buffers.append(buf)
            return buf
        elif isinstance(source, Callable):
            assert isinstance(source(), np.ndarray)
            buf = buf_type(source, dtype)
            self._generated_buffers.append(buf)
            return buf
        elif source is not None:
            # fallback to trying to interpret as an array
            array = np.asarray(source, dtype)
            buf = buf_type(array, dtype)
            self._generated_buffers.append(buf)
            return buf

    def _remove_buffer(self, buffer):
        self._dirty = True
        if buffer in self._generated_buffers:
            self._generated_buffers.remove(buffer)
            buffer.gl.release()

    def _make_glo(self):
        if any(len(buf) == 0 for buf in self._buffers_in_use.values()):
            return None
        if self._glo is not None:
            self._glo.release()
        ibo = self._index_buffer.gl if self._index_buffer else None
        self._glo = gamelib.get_context().vertex_array(
            self.shader.glo,
            self._buffer_format_tuples,
            index_buffer=ibo,
            index_element_size=4,
        )
        self._dirty = False

    def _raise_invalid_source(self, name):
        raise ValueError(
            f"{name!r} is not a valid uniform/buffer name for this shader. "
            f"Valid uniforms: {tuple(self.shader.meta.uniforms.keys())!r}, "
            f"valid buffers: {tuple(self.shader.meta.attributes.keys())!r}"
        )


def hot_reload_shaders():
    gamelib.post_event(HotReloadEvent())


class HotReloadEvent:
    pass
