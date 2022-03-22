import pytest

import time

from gamelib.rendering import shaders
from gamelib.core import gl


MINIMAL_SRC = """
#version 330
#vert
void main() {}
"""


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


def test_line_number_on_error_base_case(write_shader_to_disk):
    src = """#version 330
    #vert
    error on line 3
    void main(){}
    """
    write_shader_to_disk("test", src)

    with pytest.raises(gl.Error) as excinfo:
        shader = shaders.Shader("test")

    assert "0:3" in str(excinfo.value)


def test_line_number_on_error_with_includes(write_shader_to_disk):
    src = """#version 330
    #include test_include
    #vert
    error on line 4
    void main(){}
    """
    include = """
    int i = 0;
    int j = 1;
    """

    write_shader_to_disk("test", src)
    write_shader_to_disk("test_include", include)

    with pytest.raises(gl.Error) as excinfo:
        shader = shaders.Shader("test")

    assert "0:4" in str(excinfo.value)


def test_line_number_on_error_after_multiline_functions(write_shader_to_disk):
    src = """#version 330
    #vert

    void my_func(
        int i,
        int j=1
    ) {}

    void main()
    {
        my_func(1,
                2);
        error on line 13
    }
    """

    write_shader_to_disk("test", src)

    with pytest.raises(gl.Error) as excinfo:
        shader = shaders.Shader("test")

    assert "0:13" in str(excinfo.value)


def test_error_lines_number_from_inside_an_include_shader(
    write_shader_to_disk,
):
    src = """#version 330
    #include test_include
    #vert
    void main(){}
    """
    include = """int i = 0;
    int j = 1;
    error on line 3
    """

    write_shader_to_disk("test", src)
    write_shader_to_disk("test_include", include)

    with pytest.raises(gl.Error) as excinfo:
        shader = shaders.Shader("test")

    assert "0:3" in str(excinfo.value)
