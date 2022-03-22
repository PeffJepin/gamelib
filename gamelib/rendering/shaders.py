import pathlib
import re

from typing import Dict
from typing import List
from typing import Optional
from typing import NamedTuple

import numpy as np

from gamelib.core import resources
from gamelib.core import gl


# TODO: One pass at refactoring and some documentation where needed and this module should be good for some time.

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

    def __repr__(self):
        lines = []
        if self.vert:
            lines.append("vertex shader:")
            lines.append(self.vert)
        if self.tesc:
            lines.append("tesselation control shader:")
            lines.append(self.tesc)
        if self.tese:
            lines.append("tesselation evaluation shader:")
            lines.append(self.tese)
        if self.geom:
            lines.append("geometry shader:")
            lines.append(self.geom)
        if self.frag:
            lines.append("fragment shader:")
            lines.append(self.frag)
        return "\n".join(lines)


class Shader:
    """Entry point into the module for preprocessing glsl code."""

    _PREPROCESSOR_KWARGS = {}

    code: ShaderSourceCode
    meta: ShaderMetaData
    file: Optional[pathlib.Path] = None

    _initialized: bool
    _mtime_ns: float
    _gl_initialized: bool
    _glo: gl.GLShader

    def __new__(cls, name=None, *, src=None, no_cache=False, **kwargs):
        cls._usage(name, src)
        return cls._get_object(name, no_cache)

    def __init__(self, name=None, *, src=None, init_gl=True, **kwargs):
        if self._initialized:
            # __new__ might return a cached shader..
            # in that case we don't want to recompile the shader
            return

        if name is not None:
            self._init_from_file(name)
        else:
            self._init_from_src(src)

        if init_gl:
            self._init_gl()

        self._initialized = True

    def __hash__(self):
        return hash(self.code)

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
            return True
        except gl.Error as exc:
            print(exc)
            return False
        finally:
            self._set_file_mod_times()

    def _recompile(self):
        assert (
            self.file
        ), "cannot recompile shader sourced with a python string"
        with open(self.file, "r") as f:
            src = f.read()
        return _ShaderPreProcessor(src, no_cache=True).compile()

    def _set_file_mod_times(self):
        for shader in [self] + self.meta.includes:
            shader._mtime_ns = shader.file.stat().st_mtime_ns

    def _init_gl(self):
        self._glo = self._make_glo(self.code, self.meta)

    def _init_from_src(self, src):
        self.file = None
        self._mtime_ns = None
        code, meta = _ShaderPreProcessor(
            src, **self._PREPROCESSOR_KWARGS
        ).compile()
        self.code = code
        self.meta = meta

    def _init_from_file(self, filename):
        path = resources.get_shader_file(filename)
        with open(path, "r") as f:
            src = f.read()
        self._mtime_ns = path.stat().st_mtime_ns
        self.file = path
        code, meta = _ShaderPreProcessor(
            src, **self._PREPROCESSOR_KWARGS
        ).compile()
        self.code = code
        self.meta = meta

    @classmethod
    def _get_object(cls, name, no_cache):
        if name is not None and not no_cache:
            path = resources.get_shader_file(name)
            if path in _cache:
                return _cache[path]
        obj = object.__new__(cls)
        obj._initialized = False
        obj._gl_initialized = False
        if name is not None:
            _cache[path] = obj
        return obj

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
    _stages_regex = re.compile(
        r"""
            (?P<tag> (\#vert | \#tesc | \#tese | \#geom | \#frag | \A)\s*?\n?)
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
            | (?P<newline> \n )
        """,
        re.VERBOSE | re.DOTALL | re.MULTILINE,
    )

    def __init__(self, src, include=False, no_cache=False):
        self.lines = 1
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
        self.common = "#line 1\n" + self.common
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
        prefix = "\n" if self.common[-1] != "\n" else ""
        if prefix == "\n":
            self.lines += 1
        self.common += f"{prefix}#line {self.lines}\n"

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
        if not self.common:
            raise ValueError(
                "at a minimum the gamelib glsl preprocessor expects there to be "
                "a #version directive within the `common` shader stage."
            )

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
        elif kind == "newline":
            self.lines += 1
            return "\n"

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
        if self._function_is_glsl_builtin(function):
            return function

        desc, new_lines = self._build_function_desc(function)

        definition = self.meta.functions.get(desc.name)
        if definition is not None:
            cleaned = self._match_definition_signature(definition, desc)
        else:
            self.meta.functions[desc.name] = desc
            cleaned = self._clean_function_definition(desc)
        # keep line breaks for accurate error messages when glsl
        # code has errors.
        keepme = "\n" * new_lines
        return f"{cleaned[:-1]}{keepme})"

    def _function_is_glsl_builtin(self, function):
        name = function.split("(")[0]
        if (
            name in gl.GLSL_DTYPE_STRINGS
            or name in gl.GLSL_BUILTIN_FUNCTION_STRINGS
        ):
            return True

    def _match_definition_signature(self, defdesc, funcdesc):
        # a definition might look something like:
        # func(int i, int j=2, int k=3)
        # and a call might look something like this:
        # func(1, 2, k=4)
        if len(defdesc.arguments) == 0:
            return f"{funcdesc.name}()"

        args = [None] * len(defdesc.arguments)
        for i, (arg, value) in enumerate(
            zip(funcdesc.arguments, funcdesc.defaults)
        ):
            if value is None:
                # we already know that poistional and kwargs are in the
                # correct order so positional args can be left unchanged
                args[i] = arg
            else:
                # we can determine the correct order for kwargs from the
                # definition, remember in the definition dtypes are present
                for i, dtype_and_name in enumerate(defdesc.arguments):
                    name = dtype_and_name.split(" ")[1]
                    if name == arg:
                        args[i] = value
                        break
        # fill any remaining None values with defaults
        for i, v in enumerate(args):
            if v is None:
                args[i] = defdesc.defaults[i]

        return f"{funcdesc.name}({', '.join(args)})"

    def _clean_function_definition(self, funcdesc):
        # we just need to strip out the default values
        return f"{funcdesc.name}({', '.join(funcdesc.arguments)})"

    def _build_function_desc(self, function):
        i = function.find("(")
        name, args_str = function[:i], function[i + 1 : -1]
        if not args_str.strip():
            desc = FunctionDesc(name, (), ())
            new_lines = 0
        else:
            args, defaults, new_lines = self._split_function_args(args_str)
            if not self._check_default_arguments_syntax(defaults):
                raise SyntaxError(
                    "Arguments with default values should not appear before "
                    f"purely positional arguments: {function}"
                )
            desc = FunctionDesc(name, args, defaults)
        return desc, new_lines

    def _split_function_args(self, args_str):
        args = []
        prev = 0
        open_parens = 0
        close_parens = 0
        new_lines = 0
        for i, c in enumerate(args_str):
            if c == "(":
                open_parens += 1
            elif c == ")":
                close_parens += 1
            elif c == "," and open_parens == close_parens:
                args.append(args_str[prev:i])
                prev = i + 1
            elif c == "\n":
                new_lines += 1
        args.append(args_str[prev:])
        return *self._split_argument_names_from_defaults(args), new_lines

    def _split_argument_names_from_defaults(self, args_list):
        args = []
        defaults = []
        for a in args_list:
            split_on_eq = a.split("=")
            if len(split_on_eq) == 1:
                args.append(split_on_eq[0].strip())
                defaults.append(None)
            else:
                args.append(split_on_eq[0].strip())
                defaults.append(split_on_eq[1].strip())

        return tuple(args), tuple(defaults)

    def _check_default_arguments_syntax(self, defaults):
        can_be_positional = True

        for val in defaults:
            if val is not None:
                can_be_positional = False
            elif can_be_positional is False:
                return False

        return True

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
