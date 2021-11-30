import gamelib
from gamelib import shaders


def test_loading_a_shader_by_name():
    window = gamelib.init()
    shader = shaders.ShaderProgram(name="basic")
    while not window.is_closing:
        window.clear()
        shader.render(vertices=3)
        window.swap_buffers()


if __name__ == "__main__":
    test_loading_a_shader_by_name()
