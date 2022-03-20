import pytest
import shutil

from gamelib.rendering import shaders
from gamelib.core import resources

from ..conftest import compare_glsl


@pytest.fixture
def shaderdir(tempdir):
    directory = tempdir / "shaders"
    directory.mkdir()
    yield directory
    shutil.rmtree(directory)


@pytest.fixture
def shader_writer(shaderdir):
    def writer(filename, src):
        fn = filename if filename.endswith(".glsl") else filename + ".glsl"
        with open(shaderdir / fn, "w") as f:
            f.write(src)
        # update resource module after writing new files
        resources.set_content_roots(shaderdir)

    return writer


def test_loading_shader_from_a_file(shader_writer):
    src = """
#version 330

#vert
in vec2 v_pos;
out vec2 f_pos;
void main()
{
    f_pos = v_pos;
}
"""
    shader_writer("test", src)
    # unit tests should ensure strings are parsed correctly, we just want to
    # be sure we can load this shader by name once it's been written to disk.
    assert shaders.Shader("test") is not None


@pytest.mark.parametrize(
    "syntax",
    (
        "#include <test_include.glsl>",
        "#include <test_include>",
        '#include "test_include.glsl"',
        '#include "test_include"',
        "#include 'test_include.glsl'",
        "#include 'test_include'",
        "#include test_include.glsl",
        "#include test_include",
    ),
)
def test_include_directive(shader_writer, syntax):
    incl_src = """
int some_int;
float some_float;
vec2 some_vec2;
"""
    shader_src = f"""
#version 400
#vert
{syntax}
void main() {{}}
"""
    expected = f"""
#version 400
{incl_src}
void main() {{}}
"""
    shader_writer("test_include", incl_src)
    shader_writer("test", shader_src)

    shader = shaders.Shader("test")
    assert compare_glsl(shader.code.vert, expected)


def test_nested_includes(shader_writer):
    include1 = "#include include2"
    include2 = "hello"
    shader_src = """
#version 330
#include include1
#vert
void main() {}
"""
    expected = """
#version 330
hello
void main() {}
"""
    shader_writer("include1", include1)
    shader_writer("include2", include2)
    shader_writer("test", shader_src)

    shader = shaders.Shader("test")
    assert compare_glsl(shader.code.vert, expected)


def test_shader_loaded_from_a_file_keeps_reference_to_the_filepath(
    shader_writer,
):
    src = """
#version 330
#vert
void main() {}
"""
    str_shader = shaders.Shader(src=src)
    assert str_shader.file is None

    shader_writer("test", src)

    file_shader = shaders.Shader("test")
    assert file_shader.file is not None
