import time

from gamelib.rendering import shaders


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
