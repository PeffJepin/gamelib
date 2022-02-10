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

import dataclasses
import pathlib

from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from gamelib.core import resources
from gamelib.core import gl


@dataclasses.dataclass
class TokenDesc:
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


@dataclasses.dataclass
class ShaderMetaData:
    """A collection of metadata on tokens parsed from a single
    glsl source file."""

    attributes: Dict[str, TokenDesc]
    vertex_outputs: Dict[str, TokenDesc]
    uniforms: Dict[str, TokenDesc]


@dataclasses.dataclass(frozen=True, eq=True)
class ShaderSourceCode:
    """Source code strings for an OpenGl program."""

    vert: Optional[str]
    tesc: Optional[str]
    tese: Optional[str]
    geom: Optional[str]
    frag: Optional[str]


@dataclasses.dataclass
class Shader:
    """Entry point into the module for preprocessing glsl code."""

    code: ShaderSourceCode
    meta: ShaderMetaData
    files: Optional[List[pathlib.Path]] = None

    def __hash__(self):
        return hash(self.code)

    @classmethod
    def read_string(cls, code):
        """Preprocesses and inspects a shader provided as a single string.

        Parameters
        ----------
        code : str

        Returns
        -------
        Shader
        """

        code = _ShaderPreProcessor.process_single_string(code)
        meta = _parse_metadata(code)
        return cls(code, meta)

    @classmethod
    def read_strings(cls, vert="", tesc="", tese="", geom="", frag=""):
        """Preprocesses and inspects a shader provided as individual strings.

        Parameters
        ----------
        vert : str
        tesc : str
        tese : str
        geom : str
        frag : str

        Returns
        -------
        Shader
        """

        code = _ShaderPreProcessor.process_separate_strings(
            vert, tesc, tese, geom, frag
        )
        meta = _parse_metadata(code)
        return cls(code, meta)

    @classmethod
    def read_file(cls, filename):
        """Locates and loads a shader on disk with the resources module.

        Parameters
        ----------
        filename : str

        Returns
        -------
        Shader
        """

        paths = resources.get_shader_files(filename)

        if len(paths) == 1 and paths[0].name.endswith(".glsl"):
            with open(paths[0], "r") as f:
                src = f.read()
                code = _ShaderPreProcessor.process_single_string(src)

        else:
            src = dict()
            for path in paths:
                ext = path.name[-4:]
                assert ext in ("vert", "tesc", "tese", "geom", "frag")
                with open(path, "r") as f:
                    src[ext] = f.read()
            code = _ShaderPreProcessor.process_separate_strings(**src)

        meta = _parse_metadata(code)
        return cls(code, meta, paths)


class _ShaderPreProcessor:
    def __init__(self):
        self.vert_stage = []
        self.tesc_stage = []
        self.tese_stage = []
        self.geom_stage = []
        self.frag_stage = []
        self.common = []
        self.version_tag = None

        self.current_stage = self.common
        self.seeking_stage_end = False
        self.seeking_block_closure = False
        self.curly_braces = [0, 0]

    def compose(self) -> ShaderSourceCode:
        vert = (
            "\n".join((self.version_tag, *self.common, *self.vert_stage))
            if self.vert_stage
            else None
        )
        tesc = (
            "\n".join((self.version_tag, *self.common, *self.tesc_stage))
            if self.tesc_stage
            else None
        )
        tese = (
            "\n".join((self.version_tag, *self.common, *self.tese_stage))
            if self.tese_stage
            else None
        )
        geom = (
            "\n".join((self.version_tag, *self.common, *self.geom_stage))
            if self.geom_stage
            else None
        )
        frag = (
            "\n".join((self.version_tag, *self.common, *self.frag_stage))
            if self.frag_stage
            else None
        )
        return ShaderSourceCode(vert, tesc, tese, geom, frag)

    def process_line(self, line):
        cleaned = line.strip()

        if self.handle_gamelib_directives(cleaned):
            return
        if cleaned.startswith("//"):
            return
        if cleaned in ("", "\n"):
            return
        if cleaned.startswith("#version"):
            self.version_tag = cleaned
            return

        self.current_stage.append(cleaned)

        if self.seeking_stage_end:
            self.handle_seeking_stage_end(cleaned)

    def handle_seeking_stage_end(self, line):
        if line.startswith("void main()"):
            self.seeking_block_closure = True

        if self.seeking_block_closure:
            for c in line:
                if c == "{":
                    self.curly_braces[0] += 1
                elif c == "}":
                    self.curly_braces[1] += 1

            opening, closing = self.curly_braces
            if opening == closing and opening > 0:
                self.seeking_stage_end = False
                self.seeking_block_closure = False
                self.curly_braces = [0, 0]
                self.current_stage = self.common

    def handle_gamelib_directives(self, line) -> bool:
        if line == "#vert":
            self.seeking_stage_end = True
            self.current_stage = self.vert_stage
            return True
        if line == "#tesc":
            self.seeking_stage_end = True
            self.current_stage = self.tesc_stage
            return True
        if line == "#tese":
            self.seeking_stage_end = True
            self.current_stage = self.tese_stage
            return True
        if line == "#geom":
            self.seeking_stage_end = True
            self.current_stage = self.geom_stage
            return True
        if line == "#frag":
            self.seeking_stage_end = True
            self.current_stage = self.frag_stage
            return True
        if line.startswith("#include"):
            chars = []
            collect_name = False
            for c in line:
                if c == "<":
                    collect_name = True
                    continue
                elif c == ">":
                    break
                if collect_name:
                    chars.append(c)
            filename = "".join(chars)
            if not filename.endswith(".glsl"):
                filename += ".glsl"
            file = resources.get_file(filename)
            with open(file, "r") as f:
                self.current_stage.extend(f.readlines())
            return True
        return False

    @classmethod
    def process_separate_strings(
        cls, vert="", tesc="", tese="", geom="", frag=""
    ) -> ShaderSourceCode:
        self = cls()

        self.current_stage = self.vert_stage
        for line in vert.splitlines():
            self.process_line(line)

        self.current_stage = self.tesc_stage
        for line in tesc.splitlines():
            self.process_line(line)

        self.current_stage = self.tese_stage
        for line in tese.splitlines():
            self.process_line(line)

        self.current_stage = self.geom_stage
        for line in geom.splitlines():
            self.process_line(line)

        self.current_stage = self.frag_stage
        for line in frag.splitlines():
            self.process_line(line)

        return self.compose()

    @classmethod
    def process_single_string(cls, string) -> ShaderSourceCode:
        self = cls()

        for line in string.splitlines():
            self.process_line(line)

        return self.compose()


def _parse_metadata(src: ShaderSourceCode) -> ShaderMetaData:
    meta = ShaderMetaData({}, {}, {})

    for line in src.vert.splitlines():
        first, *values = line.split(" ")
        if first == "in":
            token = _create_token_desc(values)
            meta.attributes[token.name] = token
        elif first == "out":
            token = _create_token_desc(values)
            meta.vertex_outputs[token.name] = token
        elif first == "uniform":
            token = _create_token_desc(values)
            meta.uniforms[token.name] = token

    for code in (src.tesc, src.tese, src.geom, src.frag):
        if not code:
            continue
        for line in code.splitlines():
            first, *values = line.split(" ")
            if first == "uniform":
                token = _create_token_desc(values)
                meta.uniforms[token.name] = token

    return meta


def _create_token_desc(values) -> TokenDesc:
    """After finding a keyword like uniform, in, out... etc, this will get
    the rest of the tokens following the keyword and creates a TokenDesc.

    For the line "uniform int my_uniform_array[3];"
                values=("int", "my_uniform_array[3];")
    """

    raw_name = values[1][:-1]

    if raw_name.endswith("]"):
        length = int(raw_name[-2])
        name = raw_name[:-3]
    else:
        length = 1
        name = raw_name

    dtype = getattr(gl, values[0])
    assert isinstance(dtype, np.dtype)

    return TokenDesc(name, dtype, length, dtype_str=values[0])
