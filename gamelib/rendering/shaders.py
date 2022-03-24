"""
This module handles loading and doing some additional preprocessing on glsl
source code files. It is generally expected that glsl source code will be
located within a directory discoverable by the resources module, however glsl
source can also be given as a python string.

The most important thing to understand about this module is that it includes
a custom glsl preprocessor, so there are a few things you need to understand to
write shaders compatible with this module.


Preprocessor Features
---------------------

Here is a list of features implemented by the gamelib preprocessor.
Note that the only one you must be familiar with is the stage directive.

 - Stage Directives (#vert, #tesc, #tese, #geom, #frag)
 - Include Directive (#include "anotherfile.glsl")
 - Keyword / default function parameters
 - Metadata parsing


Examples
--------

STAGES

 Generally an OpenGL program would be specified as some number of separate
 strings, each representing a different shader stage. Gamelib expects glsl
 source to be provided as a single file / string with stage directives to mark
 the different shader stages. Note the `c`, `v`, `f` values to the left of the
 following glsl code indicate which shader stage the code falls into.


 c   #version 330        // The version directive should still come first
 c                       // Anything before the first stage is considered
 c                       // `common` and included at the head of each stage
     #vert
 v   in vec2 v_pos;      // All `c` marked lines will be injected automatically
 v   void main()         // before the code written into a stage section
 v   {
 v       ...
 v   }
 v
     #frag
 f   out vec4 frag;
 f   void main()
 f   {
 f       ...
 f   }


INCLUDES

 Using just the common section of the stages you could include shared functions
 or values for example, but this still wouldn't allow for sharing any code
 between shaders. For this there is also an #include directive implemented.
 A shader included with the #include directive should not include a #version
 directive at it's head and should not be marked up into stages. The files to be
 included should be discoverable by the resources module.

 // The preprocessor will load this file, process it and replace this line.
 #include my_library.glsl


DEFAULT FUNCTION PARAMETERS

 The preprocessor also adds keyword / default arguments:

 // The preprocessor will parse this signature and replace it with valid glsl
 void my_function(int i, int j=1, vec2 p=vec2(1, 2))
 {
     ...
 }

 // This definition is replaced with:
 void my_function(int i, int j, vec2 p)
 {
     ...
 }

 // The preprocessor will analyze function calls and restructure them into valid
 // glsl code.

 my_function(1);
 my_function(1, 1, vec2(1, 2));      // fills in defaults

 my_function(2, p=vec2(1, 1), j=3);
 my_function(2, 3, vec2(1, 1));      // parses and corrects kwarg order

 my_function(1, 2, vec2(3, 4));
 my_function(1, 2, vec2(3, 4));      // can still be called normally

 my_function();                      // SyntaxError, positional args must be given


METADATA PARSING

 The preprocessor also collects metadata about the shader, which is mostly for
 internal use. Check out the `ShaderMetaData` class to see what is collected.
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


_cache: Dict[pathlib.Path, "Shader"] = dict()


class GLSLUniform(NamedTuple):
    name: str
    dtype: np.dtype
    length: int


class GLSLAttribute(NamedTuple):
    name: str
    dtype: np.dtype
    length: int


class GLSLVertexOutput(NamedTuple):
    name: str
    dtype: np.dtype
    length: int


class GLSLSampler(NamedTuple):
    name: str
    dtype_str: str


class GLSLFunctionDefinition:
    name: str
    params: tuple
    defaults: tuple
    index_lookup: dict

    def __init__(self, name, params, defaults):
        self.name = name
        self.params = params
        self.defaults = defaults
        self.index_lookup = {p.split()[1]: i for (i, p) in enumerate(params)}

    def __repr__(self):
        sig = []
        for p, d in zip(self.params, self.defaults):
            if d is None:
                sig.append(p)
            else:
                sig.append(f"{p}={d}")
        return f"{self.name}({', '.join(sig)})"

    def __eq__(self, other):
        return repr(self) == repr(other)


class ShaderMetaData(NamedTuple):
    """A collection of metadata on tokens parsed from a single
    glsl source file."""

    attributes: Dict[str, GLSLAttribute]
    vertex_outputs: Dict[str, GLSLVertexOutput]
    uniforms: Dict[str, GLSLUniform]
    functions: Dict[str, GLSLFunctionDefinition]
    includes: List["_IncludeShader"]
    samplers: List[GLSLSampler]

    @classmethod
    def empty(cls):
        return cls(dict(), dict(), dict(), dict(), list(), list())


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
        lines.append("common:")
        lines.append(self.common)
        if self.vert:
            lines.append("vertex shader:")
            lines.append(self.vert.replace(self.common, ""))
        if self.tesc:
            lines.append("tesselation control shader:")
            lines.append(self.tesc.replace(self.common, ""))
        if self.tese:
            lines.append("tesselation evaluation shader:")
            lines.append(self.tese.replace(self.common, ""))
        if self.geom:
            lines.append("geometry shader:")
            lines.append(self.geom.replace(self.common, ""))
        if self.frag:
            lines.append("fragment shader:")
            lines.append(self.frag.replace(self.common, ""))
        return "\n".join(lines)

    def __iter__(self):
        yield self.common
        for stage in (self.vert, self.tesc, self.tese, self.geom, self.frag):
            if stage:
                yield stage.replace(self.common, "")


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
        """Either return a cached shader or create a new one."""

        cls._usage(name, src)
        return cls._get_object(name, no_cache)

    def __init__(self, name=None, *, src=None, init_gl=True, **kwargs):
        """Initialize the shader object. This includes preprocessing and
        potentially initializing an OpenGL program object.

        Parameters
        ----------
        name : str, optional
            The filename of this shader if it exists as a file on disk.
            This file should be discoverable by the resources module.
        src : str, optional
            If not sourced from a file you can provide a string directly.
        no_cache : bool
            Mainly for internal use, bypasses the cache if True.
        init_gl : bool
            If True this object will immediately generate and OpenGL program.
        """

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

    @property
    def has_been_modified(self):
        """This simply checks the previous modification time on this shaders
        file all files included in it and returns True if any of them have been
        modified since last checked. Always False if this shader was initialized
        from a python string.

        Returns
        -------
        bool
        """

        if self.file is None:
            return False
        return any(
            shader.file.stat().st_mtime_ns != shader._mtime_ns
            for shader in [self] + self.meta.includes
        )

    @property
    def glo(self):
        """Gets the internal OpenGL object for use in rendering. If that object
        has not already been initialized access to this property will do so, so
        you must have an initialized OpenGL context before accessing this."""

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
        except GLSLCompilerError as exc:
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

    def _make_glo(self, code, meta):
        try:
            return gl.make_shader_glo(
                vert=code.vert,
                tesc=code.tesc,
                tese=code.tese,
                geom=code.geom,
                frag=code.frag,
                varyings=meta.vertex_outputs,
            )
        except gl.Error as exc:
            raise GLSLCompilerError(exc, self)

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


class _IncludeShader(Shader):
    _PREPROCESSOR_KWARGS = {"include": True}

    # used with glsl #line preprocessor directive to identify which
    # file an error is coming from.
    source_string_number: int

    def __init__(self, *args, source_string_number, **kwargs):
        self.source_string_number = source_string_number
        super().__init__(*args, **kwargs)

    def _init_gl(self):
        # an include shader is just going to have it's source
        # included in another shader. We should never try to
        # initialize an OpenGL program object with this.
        return


class GLSLCompilerError(Exception):
    """Since our preprocessor can build shaders from multiple files we have to
    do some extra work to provide a clear source for an error raised by the
    GLSL compiler. This exception takes care of that."""

    def __init__(self, exc: gl.Error, shader: "Shader"):
        self._inspect_gl_exception(exc)
        error_message = str(exc)

        offending_shader = self._find_offending_shader(shader)
        if not offending_shader:
            return super().__init__(error_message)
        offending_code = self._find_offending_line(offending_shader)

        improved_error_message = (
            f"GLSLCompilerError occured in the file:\n{offending_shader.file}\n"
            f"\nThis is the offending code (line {self.line_number}):\n{offending_code}\n"
            f"\nThis is the original error message:\n{error_message}"
        )
        super().__init__(improved_error_message)

    def _inspect_gl_exception(self, exc):
        error_message = str(exc)
        m = re.search(r"(\d+:\d+)", error_message)
        if not m:
            return super().__init__(error_message)

        source_string_number, line_number = map(int, m.group().split(":"))
        self.source_string_number = source_string_number
        self.line_number = line_number

    def _find_offending_shader(self, base_shader):
        if self.source_string_number == 0:
            return base_shader
        for shader in base_shader.meta.includes:
            if shader.source_string_number == self.source_string_number:
                return shader

    def _find_offending_line(self, shader):
        entire_code = []
        for stage in shader.code:
            entire_code.extend(stage.splitlines())

        skip = False
        ln = 1
        for line in entire_code:
            if line.strip().startswith("#line"):
                # the #line directive can appear in two forms:
                # either #line (line number) or
                # #line (line number) (shader source number)
                _, directive_ln, *directive_ssn = line.split()
                directive_ssn = int(directive_ssn[0]) if directive_ssn else 0
                directive_ln = int(directive_ln)
                if directive_ssn != self.source_string_number:
                    # this case means that the following lines belong
                    # to a shader included with the offending shader from
                    # a separate file. we are going to ignore lines from this
                    # until we come across another #line directive that matches
                    # the source string number of the shader that the error
                    # actually originated from
                    skip = True
                else:
                    # we now know that we are no longer in an included shader
                    # so we can correct our line number and continue searching
                    # for the line that threw the error
                    skip = False
                    ln = directive_ln
            elif not skip and ln == self.line_number:
                return line.strip()
            elif not skip:
                ln += 1

        if ln == self.line_number:
            return line
        return "couldn't locate error line in source code"


class _ShaderPreProcessor:
    """This class is responsible for preprocessing GLSL shaders to add
    new features not included in the built-in glsl preprocessor. It is meant
    for internal use and will be invoked when Shader object are created.
    Refer to the module docstring for documentation on what features this
    preprocessor implements."""

    _STAGES_REGEX = re.compile(
        r"""
            (?P<tag> (\#vert | \#tesc | \#tese | \#geom | \#frag | \A)\s*?\n?)
            (?P<body> .*?)
            (?= (\#vert | \#tesc | \#tese | \#geom | \#frag | \Z))
        """,
        re.VERBOSE | re.DOTALL,
    )
    _POINTS_OF_INTEREST_REGEX = re.compile(
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
        self._common = ""
        self._stages = {
            "vert": "",
            "tesc": "",
            "tese": "",
            "geom": "",
            "frag": "",
        }
        self._src = src
        self._meta = ShaderMetaData.empty()
        self._function_handler = _FunctionPreprocessor(self._meta)
        self._line_number = 1
        self._include_number = 1
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
        self._common = self._src
        self._process_common()
        self._common = f"#line 1 {self._include_number}\n" + self._common
        return ShaderSourceCode(self._common), self._meta

    def _compile_base_shader(self):
        self._split_stages()
        self._process_common()
        self._process_stages()

        code = ShaderSourceCode(
            self._common,
            "".join(self._stages["vert"]) or None,
            "".join(self._stages["tesc"]) or None,
            "".join(self._stages["tese"]) or None,
            "".join(self._stages["geom"]) or None,
            "".join(self._stages["frag"]) or None,
        )

        return code, self._meta

    def _process_common(self):
        self._common = self._process_points_of_interest(self._common)

    def _process_stages(self):
        for k, v in self._stages.items():
            if not v:
                continue
            self._current_stage = k
            self._stages[k] = self._common + self._process_stage(v)

    def _process_stage(self, stage_src):
        # we've stripped out the #stage directive so take account for that
        self._line_number += 1
        # remember the line number for the start of this stage
        ln = self._line_number
        processed = self._process_points_of_interest(stage_src)
        return f"#line {ln} 0\n{processed}"

    def _process_points_of_interest(self, src):
        return self._POINTS_OF_INTEREST_REGEX.sub(
            self._handle_replacement, src
        )

    def _split_stages(self):
        for m in self._STAGES_REGEX.finditer(self._src):
            tag = m.group("tag").strip()
            if not tag:
                self._common = m.group("body")
            else:
                for k in self._stages:
                    if k in tag:
                        self._stages[k] = m.group("body")
                        break
        if not self._common:
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
            self._line_number += 1
            return "\n"

    def _handle_include(self, directive):
        _, raw_filename = directive.split()
        filename = raw_filename.strip(" <>'\"\n")

        shader = _IncludeShader(
            filename,
            source_string_number=self._include_number,
            no_cache=self._no_cache,
        )
        self._include_number += 1

        self._meta.functions.update(shader.meta.functions)
        self._meta.uniforms.update(shader.meta.uniforms)
        self._meta.includes.append(shader)
        self._meta.includes.extend(shader.meta.includes)
        return shader.code.common

    def _handle_function(self, function):
        try:
            return self._function_handler.handle_function_replacement(function)
        except SyntaxError as exc:
            # capture syntax errors from the function preprocessor
            # and add line number information before raising the error
            raise SyntaxError(
                f"This error occured on line {self._line_number}:\n"
                f"{str(exc)}"
            )

    def _handle_uniform(self, raw_match):
        desc = self._create_uniform_desc(raw_match)
        if isinstance(desc, GLSLSampler):
            self._meta.samplers.append(desc)
        else:
            self._meta.uniforms[desc.name] = desc
        return raw_match

    def _handle_attribute(self, raw_match):
        if self._current_stage == "vert":
            desc = self._create_attribute_desc(raw_match)
            self._meta.attributes[desc.name] = desc
        return raw_match

    def _handle_vertex_output(self, raw_match):
        if self._current_stage == "vert":
            desc = self._create_vertex_output_desc(raw_match)
            self._meta.vertex_outputs[desc.name] = desc
        return raw_match

    def _create_uniform_desc(self, raw):
        _, dtype, name, length = self._parse_kw_dtype_name_len(raw)
        if dtype == "sampler2D":
            return GLSLSampler(name, dtype)
        return GLSLUniform(name, dtype, length)

    def _create_vertex_output_desc(self, raw):
        _, dtype, name, length = self._parse_kw_dtype_name_len(raw)
        return GLSLVertexOutput(name, dtype, length)

    def _create_attribute_desc(self, raw):
        _, dtype, name, length = self._parse_kw_dtype_name_len(raw)
        return GLSLAttribute(name, dtype, length)

    def _parse_kw_dtype_name_len(self, raw):
        # given a line like the following, parse out each component
        # (glsl_keyword) (glsl_dtype) (name) [length (optional)];
        m = re.search(
            r"""
            (?P<keyword> \w+) \s
            (?P<dtype> \w+) \s
            (?P<name> \w+)
            \[? (?P<length>  \d+ )? \]? ;
            """,
            raw,
            re.VERBOSE,
        )
        kw, dtype, name, maybe_length = m.groups()
        length = int(maybe_length) if maybe_length else 1
        return kw, getattr(gl, dtype), name, length


class _FunctionPreprocessor:
    """This class handles GLSL function related preprocessing. It is separate
    simply because it's starting to accrue quite a lot of logic."""

    _current_raw_match: str
    _current_newline_count: int

    def __init__(self, metadata):
        self._meta = metadata
        self._current_raw_match = ""
        self._current_newline_count = 0

    def handle_function_replacement(self, raw_match):
        name, sig = self._split_name_and_signature(raw_match)
        if (
            name in gl.GLSL_DTYPE_STRINGS
            or name in gl.GLSL_BUILTIN_FUNCTION_STRINGS
        ):
            # if this is a glsl built-in then we're not going to touch it
            return raw_match
        elif name not in self._meta.functions:
            # if it's not a built-in function and it's not already registered
            # in the metadata, then this must be the function definition.
            return self._handle_function_definition(name, sig)
        else:
            # otherwise its a call to a user defined function
            return self._handle_function_call(name, sig)

    def _handle_function_definition(self, name, sig):
        desc = self._create_function_definition_object(name, sig)
        self._meta.functions[name] = desc
        sig = ", ".join(desc.params)
        return f"{name}({sig}{self._line_cnt_preservation})"

    def _create_function_definition_object(self, name, sig):
        names, defaults = self._split_signature(sig) if sig else ((), ())
        return GLSLFunctionDefinition(name, names, defaults)

    def _handle_function_call(self, name, sig):
        definition_desc = self._meta.functions[name]
        if not sig:
            missing_positional_params = [
                param
                for (param, default) in zip(
                    definition_desc.params, definition_desc.defaults
                )
                if default is None
            ]
            if any(missing_positional_params):
                raise SyntaxError(
                    f"Missing positional parameters {missing_positional_params}"
                    f" in function call: {self._current_raw_match}"
                )

            sig = ", ".join(definition_desc.defaults)
            return f"{name}({sig}{self._line_cnt_preservation})"

        return self._build_proper_glsl_function_call(
            name, sig, definition_desc
        )

    def _build_proper_glsl_function_call(self, name, sig, definition_desc):
        # we always have a left value, and sometimes have a right value
        # if right is not None, then we're looking at a keyword argument
        # if right is None, then we're looking at a positional argument
        left, right = self._split_signature(sig)
        args = [None] * len(definition_desc.params)

        for i, (vl, vr) in enumerate(zip(left, right)):
            if vr is None:
                # at this point we know positional args come first, so we can
                # consume positional args out of left as long as right is None
                args[i] = vl
            else:
                # once right is no longer None we're handling kwargs
                argindex = definition_desc.index_lookup.get(vl)
                if argindex is None:
                    raise SyntaxError(
                        "Unknown keyword argument {vl!r} found in "
                        f"the function call {self._current_raw_match}"
                    )
                args[argindex] = vr

        # finally we need to fill in remaining None values with defaults
        for i, v in enumerate(args):
            if v is None:
                default = definition_desc.defaults[i]
                if default is None:
                    raise SyntaxError(
                        f"Missing value given for parameter {definition_desc.params[i]} "
                        f"in the function call {self._current_raw_match}"
                    )
                args[i] = default

        sig = ", ".join(args)
        return f"{name}({sig}{self._line_cnt_preservation})"

    @property
    def _line_cnt_preservation(self):
        # might look for a more elegant solution to this problem in
        # the future but this works just fine for now
        ln_cnt = self._current_newline_count
        self._current_newline_count = 0
        return "\n" * ln_cnt

    def _split_signature(self, sig):
        # given a signature my_func(int i, int j=1, vec2 p=vec2(1, 2))
        # split into ('int i', 'int j', 'vec2 p'), (None, '1', 'vec2(1, 2)')

        # this might also be used on a function call, something like:
        # my_func(1, 2, p=vec2(1, 1))
        # splits to: (1, 2, 'p'), (None, None, 'vec2(1, 1)')

        params = self._split_signature_into_parameters(sig)
        name_or_value, value_or_none = self._split_positional_and_kwds(params)
        self._error_if_positional_args_after_kwargs(value_or_none)
        return name_or_value, value_or_none

    def _split_signature_into_parameters(self, sig):
        # given a signature my_func(int i, int j=1, vec2 p=vec2(1, 2))
        # split into ['int i', 'int j=1', 'vec2 p=vec2(1, 2)']
        params = []
        prev = 0
        open_parens = 0
        close_parens = 0
        for i, c in enumerate(sig):
            if c == "(":
                open_parens += 1
            elif c == ")":
                close_parens += 1
            elif c == "," and open_parens == close_parens:
                params.append(sig[prev:i])
                prev = i + 1
            elif c == "\n":
                self._current_newline_count += 1
        params.append(sig[prev:])
        return params

    def _split_positional_and_kwds(self, args_list):
        # given a list ['int i', 'int j=1', 'vec2 p=vec2(1, 2)']
        # split into ('int i', 'int j', 'vec2 p'), (None, '1', 'vec2(1, 2)')
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

    def _error_if_positional_args_after_kwargs(self, kwds):
        # pass in an iterable that evaluates to true where there is a kwarg
        can_be_positional = True
        for kw in kwds:
            if kw:
                can_be_positional = False
            elif not can_be_positional:
                raise SyntaxError(
                    "Arguments with default values should not appear before "
                    f"purely positional arguments: {self._current_raw_match}"
                )

    @staticmethod
    def _split_name_and_signature(raw_match):
        i = raw_match.find("(")
        name, sig = raw_match[:i], raw_match[i + 1 : -1]
        # important not to strip the signature because we want to
        # keep count of newlines to preserve line number for errors
        return name.strip(), sig
