import pytest
import shutil
import time

from gamelib.rendering import shaders
from gamelib.core import resources

from ..conftest import compare_glsl


MINIMAL_SRC = """
#version 330
#vert
void main() {}
"""


def test_loading_shader_from_a_file(write_shader_to_disk):
    write_shader_to_disk("test", MINIMAL_SRC)
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
def test_include_directive(write_shader_to_disk, syntax):
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
    write_shader_to_disk("test_include", incl_src)
    write_shader_to_disk("test", shader_src)

    shader = shaders.Shader("test")
    assert compare_glsl(shader.code.vert, expected)


def test_nested_includes(write_shader_to_disk):
    include1 = "#include include2"
    include2 = "int i = 0;"
    shader_src = """
#version 330
#include include1
#vert
void main() {}
"""
    expected = """
#version 330
int i = 0;
void main() {}
"""
    write_shader_to_disk("include1", include1)
    write_shader_to_disk("include2", include2)
    write_shader_to_disk("test", shader_src)

    shader = shaders.Shader("test")
    assert compare_glsl(shader.code.vert, expected)


def test_shader_loaded_from_a_file_keeps_reference_to_the_filepath(
    write_shader_to_disk,
):
    str_shader = shaders.Shader(src=MINIMAL_SRC)
    assert str_shader.file is None

    write_shader_to_disk("test", MINIMAL_SRC)

    file_shader = shaders.Shader("test")
    assert file_shader.file is not None


def test_shaders_from_file_sources_are_cached(write_shader_to_disk):
    write_shader_to_disk("test", MINIMAL_SRC)

    s1 = shaders.Shader("test")
    s2 = shaders.Shader("test")

    assert s1 is s2


def test_include_tracking_base_case(write_shader_to_disk):
    src = """
#version 330
#include test_include1
#include test_include2
#vert
void main() {}
"""
    incl1 = "int i = 0;"
    incl2 = "int j = 0;"

    write_shader_to_disk("test_include1", incl1)
    write_shader_to_disk("test_include2", incl2)
    write_shader_to_disk("test", src)
    shader = shaders.Shader("test")
    included_filenames = [shader.file.name for shader in shader.meta.includes]

    assert len(shader.meta.includes) == 2
    assert "test_include1.glsl" in included_filenames
    assert "test_include2.glsl" in included_filenames


def test_nested_include_tracking(write_shader_to_disk):
    src = """
#version 330
#include test_include1
#vert
void main() {}
"""
    incl1 = "#include test_include2"
    incl2 = "int i = 0;"

    write_shader_to_disk("test_include1", incl1)
    write_shader_to_disk("test_include2", incl2)
    write_shader_to_disk("test", src)
    shader = shaders.Shader("test")
    included_filenames = [shader.file.name for shader in shader.meta.includes]

    assert len(shader.meta.includes) == 2
    assert "test_include1.glsl" in included_filenames
    assert "test_include2.glsl" in included_filenames


def test_shader_is_not_dirty_when_left_unmodified(write_shader_to_disk):
    write_shader_to_disk("test", MINIMAL_SRC)
    shader = shaders.Shader("test")

    assert not shader.has_been_modified


def test_shader_has_been_modified_when_modified(write_shader_to_disk):
    write_shader_to_disk("test", MINIMAL_SRC)
    shader = shaders.Shader("test")

    time.sleep(0.01)
    write_shader_to_disk("test", MINIMAL_SRC)

    assert shader.has_been_modified


def test_shader_has_been_modified_when_includes_are_modified(
    write_shader_to_disk,
):
    src = """
#version 330
#include test_include
#vert
void main() {}
"""
    incl = "int i = 0;"

    write_shader_to_disk("test_include", incl)
    write_shader_to_disk("test", src)
    shader = shaders.Shader("test")

    time.sleep(0.01)
    write_shader_to_disk("test_include", incl)

    assert shader.has_been_modified


def test_hot_reloading_when_shader_hasnt_been_modified(write_shader_to_disk):
    write_shader_to_disk("test", MINIMAL_SRC)
    shader = shaders.Shader("test")
    glo = shader.glo

    assert shader.try_hot_reload() is False
    assert shader.glo is glo
    assert shader.has_been_modified is False


def test_hot_reloading_when_modification_is_valid(write_shader_to_disk):
    write_shader_to_disk("test", MINIMAL_SRC)
    shader = shaders.Shader("test")
    glo = shader.glo

    time.sleep(0.01)
    write_shader_to_disk("test", MINIMAL_SRC)

    assert shader.try_hot_reload() is True
    assert shader.glo is not glo
    assert shader.has_been_modified is False


def test_hot_reloading_when_modification_is_invalid(
    write_shader_to_disk, capsys
):
    write_shader_to_disk("test", MINIMAL_SRC)
    shader = shaders.Shader("test")
    glo = shader.glo

    time.sleep(0.01)
    write_shader_to_disk("test", MINIMAL_SRC + "not code")

    assert shader.try_hot_reload() is False
    assert glo is shader.glo
    assert capsys.readouterr().out
    assert shader.has_been_modified is False
