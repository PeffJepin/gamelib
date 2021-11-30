import gamelib
import numpy as np
from gamelib import shaders


def test_shader_rendering():
    window = gamelib.init()
    shader = shaders.ShaderProgram(
        vertex_shader="""
            #version 330
            in vec2 v_pos;
            in vec3 v_col;
            uniform vec2 offset;
            out vec3 color;
            
            void main()
            {
                gl_Position = vec4(v_pos + offset, 0, 1);
                color = v_col;
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 color;
            out vec4 frag; 
            
            void main() 
            {
                frag = vec4(color, 1);
            }
        """,
        uniforms={"offset": np.array([0.1, -0.1])},
        buffers={
            "v_pos": np.array([(-1, -1), (0, 1), (1, -1)]),
            "v_col": np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
        },
    )
    while not window.is_closing:
        window.clear()
        shader.render()
        window.swap_buffers()


if __name__ == "__main__":
    test_shader_rendering()
