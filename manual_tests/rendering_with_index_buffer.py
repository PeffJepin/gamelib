import numpy as np

import gamelib
from gamelib import shaders


def main():
    window = gamelib.init()
    index_buffer = shaders.IndexBuffer((0, 1, 2, 0, 2, 3), entities=(0, 1, 3))
    pos_buffer = np.array([
        (-0.9, -0.9), (-0.9, -0.1), (-0.1, -0.1), (-0.1, -0.9),
        (-0.9, 0.1), (-0.9, 0.9), (-0.1, 0.9), (-0.1, 0.1),
        (0.1, 0.1), (0.1, 0.9), (0.9, 0.9), (0.9, 0.1),
        (0.1, -0.9), (0.1, -0.1), (0.9, -0.1), (0.9, -0.9)
    ])
    col_buffer = np.array([
        (1, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0),
        (1, 1, 1), (1, 1, 0), (0, 0, 1), (0, 1, 0),
        (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1),
        (0.1, 0, 0), (0, 1, 0), (1, 1, 1), (1, 0, 0),
    ])
    shader = shaders.ShaderProgram(
        vertex_shader="""
            #version 330
            
            in vec2 v_pos;
            in vec3 v_col;
            
            out vec3 color;
            
            void main()
            {
                gl_Position = vec4(v_pos, 0, 1);
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
        index_buffer=index_buffer,
        buffers={"v_pos": pos_buffer, "v_col": col_buffer}
    )
    while not window.is_closing:
        window.clear()
        shader.render()
        window.swap_buffers()


if __name__ == "__main__":
    main()
