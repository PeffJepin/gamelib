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


def test_loading_shader_from_a_file(shaderdir):
    with open(shaderdir / "test.glsl", "w") as f:
        f.write(
            """
#version 330

#vert
in vec2 v_pos;
out vec2 f_pos;
void main()
{
    f_pos = v_pos;
}
"""
        )

    # unit tests should ensure strings are parsed correctly, we just want to
    # be sure we can load this shader by name once it's been written to disk.
    resources.set_content_roots(shaderdir)
    assert shaders.Shader.read_file("test") is not None


@pytest.mark.parametrize(
    "syntax",
    (
        "#include <test.glsl>",
        "#include <test>",
        # '#include "test.glsl"',
        # '#include "test"',
        # "#include 'test.glsl'",
        # "#include 'test'",
        # "#include test.glsl",
        # "#include test",
    ),
)
def test_include_directive(shaderdir, syntax):
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
    with open(shaderdir / "test.glsl", "w") as f:
        f.write(incl_src)
    with open(shaderdir / "test_shader.glsl", "w") as f:
        f.write(shader_src)
    resources.set_content_roots(shaderdir)

    shader = shaders.Shader.read_file("test_shader")
    assert compare_glsl(shader.code.vert, expected)


def test_shader_loaded_from_a_file_keeps_reference_to_the_filepath(shaderdir):
    src = """
#version 330
#vert
void main() {}
"""
    str_shader = shaders.Shader.parse(src)
    assert str_shader.file is None

    filepath = shaderdir / "test_shader.glsl"
    with open(filepath, "w") as f:
        f.write(src)
    resources.set_content_roots(shaderdir)

    file_shader = shaders.Shader.read_file("test_shader")
    assert file_shader.file == filepath
