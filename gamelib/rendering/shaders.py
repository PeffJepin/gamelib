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
from typing import Tuple
from typing import Optional
from typing import NamedTuple

import numpy as np

from gamelib.core import resources
from gamelib.core import gl


# TODO: Test nested includes
# TODO: Test hot reloading
# TODO: Test function defualt arguments
# TODO: Test line number debugging
# TODO: Maybe shaders loaded as includes should be their own object, since they don't have stages
# TODO: Shaders with includes need to know that they have includes for hot reloading
# TODO: This module docstring and documentation needs to be updated.
# TODO: Look over this modules entire API before finishing with it
# TODO: Include some default gamelib glsl libraries available for includes

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

    def __new__(cls, name=None, *, src=None):
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

        # we might want return an existing shader instead of passing
        # control over to __init__
        if name is not None:
            path = resources.get_shader_file(name)
            if path in _cache:
                return _cache[path]

        return object.__new__(cls)

    def __init__(self, name=None, *, src=None):
        if name is not None:
            path = resources.get_shader_file(name)
            with open(path, "r") as f:
                src = f.read()
            _cache[path] = self
        else:
            path = None

        code, meta = _ShaderPreProcessor(src).compile(
            **self._PREPROCESSOR_KWARGS
        )
        self.code = code
        self.meta = meta
        self.file = path

    def __hash__(self):
        return hash(self.code)


class _IncludeShader(Shader):
    _PREPROCESSOR_KWARGS = {"include": True}


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

    def __init__(self, src):
        self.common = ""
        self.stages = {
            "vert": "",
            "tesc": "",
            "tese": "",
            "geom": "",
            "frag": "",
        }
        self.src = src
        self.meta = ShaderMetaData({}, {}, {}, {})
        self._current_stage = None

    def compile(self, include=False):
        if include:
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

        shader = _IncludeShader(filename)
        self.meta.functions.update(shader.meta.functions)
        self.meta.uniforms.update(shader.meta.uniforms)

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
        name = name.strip()[:-1]  # remove ;
        if name.endswith("]"):
            name, raw_lenstr = name.split("[")
            length = int(raw_lenstr[:-1])
        else:
            length = 1
        return TokenDesc(name, getattr(gl, glsl_type), length, glsl_type)
