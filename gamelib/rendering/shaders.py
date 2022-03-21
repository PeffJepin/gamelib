"""This module is responsible for inspecting and manipulating glsl shader
code.

Notes
--------
Given a simple shader like shown below, the following shows the formats gamelib
expects a shader to take.

// vertex shader
#version 330
in vec3 v_pos;
void main() {
    gl_Position = vec4(v_pos, 1.0);
}

// fragment shader
#version 330
out vec4 frag;
void main() {
    frag = vec4(1.0, 1.0, 1.0, 1.0);
}

Each stage can be given separately, with the shader stage specified in whatever
function call it is being passed into.

Alternatively, this module includes some pre-processing directives for
specifying a shader as a single string like so:

c   #version 330

    #vert
v   in vec3 v_pos;
v   void main() {
v       gl_Position = vec4(v_pos, 1.0);
v   }

    #frag
f   out vec4 frag;
f   void main() {
f       frag = vec4(1.0, 1.0, 1.0, 1.0);
f   }

In the 'gutter' above the letters v and f indicate which lines will end up in
the vertex and fragment shaders respectively. The shader stage directives are
as follows:
    "#vert" : vertex shader
    "#tesc" : tessellation control shader
    "#tese" : tessellation evaluation shader
    "#geom" : geometry shader
    "#frag" : fragment shader

A shader stage marked by a directive like shown above begins on the line
immediately after the directive, and continues until the end of main().

Note that #version 330 is marked with a c for common. Anything outside of a
marked shader stage is considered common to all stages, and will be injected
at the beginning of the shader before its own source code.

Shaders can also be provided as files on disk. Much like using strings, shaders
from a file can either be a collection of files:
    (shader.vert, shader.frag)
Or they can be given as a single file likes:
    shader.glsl

Finally, this module also implements an #include directive for injecting some
code. Given the following:

    // colors.glsl
    vec4 my_red_color = vec4(1.0, 0.0, 0.0, 1.0);
    vec4 my_blue_color = vec4(0.0, 1.0, 0.0, 1.0);
    vec4 my_green_color = vec4(0.0, 0.0, 1.0, 1.0);

    // fragment shader
    #include <colors>
    out vec4 frag;
    void main() {
        frag = my_red_color;
    }
    // #include <colors.glsl> also acceptable

This fragment shader would expand to:
    // fragment shader
    vec4 my_red_color = vec4(1.0, 0.0, 0.0, 1.0);
    vec4 my_blue_color = vec4(0.0, 1.0, 0.0, 1.0);
    vec4 my_green_color = vec4(0.0, 0.0, 1.0, 1.0);
    out vec4 frag;
    void main() {
        frag = my_red_color;
    }
"""

import pathlib
import re

from typing import Dict
from typing import List
from typing import Optional
from typing import NamedTuple

import numpy as np

from gamelib.core import resources
from gamelib.core import gl


# TODO: Test function defualt arguments
# TODO: Test line number debugging
# TODO: This module docstring and documentation needs to be updated.
# TODO: Look over this modules entire API before finishing with it
# TODO: Include some default gamelib glsl libraries available for includes
# TODO: Documentation once finished with this module

# Shaders loaded from files are cached with a record of the file's
# previously modified time to facilitate automatic hot reloading
_cache: Dict[pathlib.Path, "Shader"] = dict()


class TokenDesc(NamedTuple):
    """Data describing a token parsed from glsl source code."""

    name: str
    dtype: np.dtype
    length: int  # array length if token is an array, else 1
    dtype_str: Optional[str] = None

    def __eq__(self, other):
        return (
            isinstance(other, TokenDesc)
            and self.name == other.name
            and self.dtype == other.dtype
            and self.length == other.length
        )


class FunctionDesc(NamedTuple):
    name: str
    arguments: tuple
    defaults: tuple


class ShaderMetaData(NamedTuple):
    """A collection of metadata on tokens parsed from a single
    glsl source file."""

    attributes: Dict[str, TokenDesc]
    vertex_outputs: Dict[str, TokenDesc]
    uniforms: Dict[str, TokenDesc]
    functions: Dict[str, FunctionDesc]
    includes: List["_IncludeShader"]


class ShaderSourceCode(NamedTuple):
    """Source code strings for an OpenGl program."""

    common: str
    vert: Optional[str] = None
    tesc: Optional[str] = None
    tese: Optional[str] = None
    geom: Optional[str] = None
    frag: Optional[str] = None


class Shader:
    """Entry point into the module for preprocessing glsl code."""

    _PREPROCESSOR_KWARGS = {}

    code: ShaderSourceCode
    meta: ShaderMetaData
    file: Optional[pathlib.Path] = None

    _initialized: bool
    _gl_initialized: bool
    _glo: gl.GLShader

    def __new__(cls, name=None, *, src=None, no_cache=False, **kwargs):
        cls._usage(name, src)

        # we might want return an existing shader instead of passing
        # control over to __init__
        if name is not None and not no_cache:
            path = resources.get_shader_file(name)
            if path in _cache:
                return _cache[path]

        obj = object.__new__(cls)
        obj._initialized = False
        obj._gl_initialized = False
        return obj

    def __init__(self, name=None, *, src=None, init_gl=True, **kwargs):
        if self._initialized:
            return

        if name is not None:
            path = resources.get_shader_file(name)
            with open(path, "r") as f:
                src = f.read()
            _cache[path] = self
            self._mtime_ns = path.stat().st_mtime_ns
        else:
            path = None
            self._mtime_ns = None

        code, meta = _ShaderPreProcessor(
            src, **self._PREPROCESSOR_KWARGS
        ).compile()
        self.code = code
        self.meta = meta
        self.file = path

        if init_gl:
            self._init_gl()

        self._initialized = True

    def __hash__(self):
        return hash(self.code)

    def __repr__(self):
        lines = []
        if self.code.vert:
            lines.append("vertex shader:")
            lines.append(self.code.vert)
        if self.code.tesc:
            lines.append("tesselation control shader:")
            lines.append(self.code.tesc)
        if self.code.tese:
            lines.append("tesselation evaluation shader:")
            lines.append(self.code.tese)
        if self.code.geom:
            lines.append("geometry shader:")
            lines.append(self.code.geom)
        if self.code.frag:
            lines.append("fragment shader:")
            lines.append(self.code.frag)
        return "\n".join(lines)

    @property
    def has_been_modified(self):
        if self.file is None:
            return False
        return any(
            shader.file.stat().st_mtime_ns != shader._mtime_ns
            for shader in [self] + self.meta.includes
        )

    @property
    def glo(self):
        if self._glo is None:
            self._init_gl()
        return self._glo

    def try_hot_reload(self):
        # FIXME: The need for a logging module is growing
        if not self.has_been_modified:
            print("This shader hasn't been modified")
            return False
        try:
            code, meta = self._recompile()
            glo = self._make_glo(code, meta)
            self._glo = glo
            self.code = code
            self.meta = meta
            self._set_file_mod_times()
            return True
        except gl.Error as exc:
            print(exc)
            self._set_file_mod_times()
            return False

    def _recompile(self):
        with open(self.file, "r") as f:
            src = f.read()
        return _ShaderPreProcessor(src, no_cache=True).compile()

    def _set_file_mod_times(self):
        for shader in [self] + self.meta.includes:
            shader._mtime_ns = shader.file.stat().st_mtime_ns

    @staticmethod
    def _usage(name, src):
        if name is not None and src is not None:
            raise ValueError(
                "Shaders can either be sourced from a python "
                "source string, or from a file on disk. So "
                "`name` and `src` are mutually exclusive."
            )
        elif name is None and src is None:
            raise ValueError(
                "No source specified for this shader, please supply "
                "either a filename or source string."
            )

    def _init_gl(self):
        self._glo = self._make_glo(self.code, self.meta)

    @staticmethod
    def _make_glo(code, meta):
        return gl.make_shader_glo(
            vert=code.vert,
            tesc=code.tesc,
            tese=code.tese,
            geom=code.geom,
            frag=code.frag,
            varyings=meta.vertex_outputs,
        )


class _IncludeShader(Shader):
    _PREPROCESSOR_KWARGS = {"include": True}

    def _init_gl(self):
        return


class _ShaderPreProcessor:
    _ENSURE_COMMON_ERROR = (
        "at a minimum the gamelib glsl preprocessor expects there to be "
        "a #version directive within the `common` shader stage."
    )
    _stages_regex = re.compile(
        r"""
            (?P<tag> (\#vert | \#tesc | \#tese | \#geom | \#frag | \A)\s*?\n)
            (?P<body> .*?)
            (?= (\#vert | \#tesc | \#tese | \#geom | \#frag | \Z))
        """,
        re.VERBOSE | re.DOTALL,
    )
    _points_of_interest_regex = re.compile(
        r"""
            (?P<include> \#include \s .*? $)
            | (?P<function> \b \w+ \( [^;{]* \) )
            | (?P<uniform> \b uniform \s \w+ \s \w+ (\[\d+\])?;)
            | (?P<attribute> \b in \s \w+ \s \w+ (\[\d+\])?;)
            | (?P<vertex_output> \b out \s \w+ \s \w+ (\[\d+\])?;)
        """,
        re.VERBOSE | re.DOTALL | re.MULTILINE,
    )

    def __init__(self, src, include=False, no_cache=False):
        self.common = ""
        self.stages = {
            "vert": "",
            "tesc": "",
            "tese": "",
            "geom": "",
            "frag": "",
        }
        self.src = src
        self.meta = ShaderMetaData({}, {}, {}, {}, [])
        self._current_stage = None
        self._include = include
        self._no_cache = no_cache

    def compile(self):
        if self._include:
            return self._compile_include_shader()
        else:
            return self._compile_base_shader()

    def _compile_include_shader(self):
        # an include shader is not split into stages
        self.common = self.src
        self._process_common()
        return ShaderSourceCode(self.common), self.meta

    def _compile_base_shader(self):
        self._split_stages()
        self._process_common()
        self._process_stages()

        code = ShaderSourceCode(
            self.common,
            "".join(self.stages["vert"]) or None,
            "".join(self.stages["tesc"]) or None,
            "".join(self.stages["tese"]) or None,
            "".join(self.stages["geom"]) or None,
            "".join(self.stages["frag"]) or None,
        )

        return code, self.meta

    def _process_common(self):
        self.common = self._process_stage(self.common)

    def _process_stages(self):
        for k, v in self.stages.items():
            if not v:
                continue
            self._current_stage = k
            self.stages[k] = self.common + self._process_stage(v)

    def _process_stage(self, stage_src):
        return self._points_of_interest_regex.sub(
            self._handle_replacement, stage_src
        )

    def _split_stages(self):
        for m in self._stages_regex.finditer(self.src):
            tag = m.group("tag").strip()
            if not tag:
                self.common = m.group("body")
            else:
                for k in self.stages:
                    if k in tag:
                        self.stages[k] = m.group("body")
                        break
        assert self.common, self._ENSURE_COMMON_ERROR

    def _handle_replacement(self, m):
        kind = m.lastgroup
        value = m.group(kind)

        if kind == "include":
            return self._handle_include(value)
        elif kind == "function":
            return self._handle_function(value)
        elif kind == "uniform":
            return self._handle_uniform(value)
        elif kind == "attribute":
            return self._handle_attribute(value)
        elif kind == "vertex_output":
            return self._handle_vertex_output(value)

    def _handle_include(self, directive):
        _, raw_filename = directive.split()
        filename = raw_filename.strip(" <>'\"\n")

        shader = _IncludeShader(filename, no_cache=self._no_cache)
        self.meta.functions.update(shader.meta.functions)
        self.meta.uniforms.update(shader.meta.uniforms)
        self.meta.includes.append(shader)
        self.meta.includes.extend(shader.meta.includes)

        return shader.code.common

    def _handle_function(self, function):
        return function

    def _handle_uniform(self, uniform):
        desc = self._create_token_desc(uniform)
        self.meta.uniforms[desc.name] = desc

        return uniform

    def _handle_attribute(self, attribute):
        if self._current_stage != "vert":
            pass
        else:
            desc = self._create_token_desc(attribute)
            self.meta.attributes[desc.name] = desc

        return attribute

    def _handle_vertex_output(self, output):
        if self._current_stage != "vert":
            pass
        else:
            desc = self._create_token_desc(output)
            self.meta.vertex_outputs[desc.name] = desc

        return output

    def _create_token_desc(self, raw):
        # given a line like the following:
        # (glsl_keyword) (glsl_dtype) (name) [length (optional)];
        # returns the token desc object
        _, glsl_type, name = raw.split()
        name = name.strip()[:-1]  # remove ; and surrounding whitespace
        if name.endswith("]"):
            name, raw_lenstr = name.split("[")
            length = int(raw_lenstr[:-1])
        else:
            length = 1
        return TokenDesc(name, getattr(gl, glsl_type), length, glsl_type)
