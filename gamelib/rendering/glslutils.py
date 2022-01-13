import dataclasses

from typing import List
from typing import Optional

import numpy as np

from gamelib import gl
from gamelib.core import resources


@dataclasses.dataclass(eq=True)
class TokenDesc:
    """Data describing a token parsed from glsl source code."""

    name: str
    dtype: np.dtype
    length: int  # array length if token is an array, else 1


@dataclasses.dataclass
class ShaderMetaData:
    """A collection of metadata on tokens parsed from a single
    glsl source file."""

    attributes: List[TokenDesc]
    vertex_outputs: List[TokenDesc]
    uniforms: List[TokenDesc]


@dataclasses.dataclass
class ShaderSourceCode:
    """Source code strings for an OpenGl program."""

    vert: Optional[str]
    tesc: Optional[str]
    tese: Optional[str]
    geom: Optional[str]
    frag: Optional[str]


@dataclasses.dataclass
class ShaderData:
    code: ShaderSourceCode
    meta: ShaderMetaData

    @classmethod
    def read_string(cls, string):
        code = _ShaderPreProcessor.process_single_string(string)
        meta = _parse_metadata(code)
        return cls(code, meta)

    @classmethod
    def read_strings(cls, vert="", tesc="", tese="", geom="", frag=""):
        code = _ShaderPreProcessor.process_separate_strings(
            vert, tesc, tese, geom, frag
        )
        meta = _parse_metadata(code)
        return cls(code, meta)

    @classmethod
    def read_file(cls, name):
        paths = resources.get_shader_files(name)

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
        return cls(code, meta)


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

    def handle_gamelib_directives(self, line):
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
    ):
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
    def process_single_string(cls, string):
        self = cls()

        for line in string.splitlines():
            self.process_line(line)

        return self.compose()


def _parse_metadata(src: ShaderSourceCode) -> ShaderMetaData:
    meta = ShaderMetaData([], [], [])

    for line in src.vert.splitlines():
        first, *values = line.split(" ")
        if first == "in":
            meta.attributes.append(_create_token_desc(values))
        elif first == "out":
            meta.vertex_outputs.append(_create_token_desc(values))
        elif first == "uniform":
            desc = _create_token_desc(values)
            if desc not in meta.uniforms:
                meta.uniforms.append(desc)

    for code in (src.tesc, src.tese, src.geom, src.frag):
        if not code:
            continue
        for line in code.splitlines():
            first, *values = line.split(" ")
            if first == "uniform":
                desc = _create_token_desc(values)
                if desc not in meta.uniforms:
                    meta.uniforms.append(desc)

    return meta


def _create_token_desc(values) -> TokenDesc:
    raw_name = values[1][:-1]

    if raw_name.endswith("]"):
        length = int(raw_name[-2])
        name = raw_name[:-3]
    else:
        length = 1
        name = raw_name

    dtype = getattr(gl, values[0])
    assert isinstance(dtype, np.dtype)

    return TokenDesc(name, dtype, length)
