import numpy as np

import gamelib
from gamelib.rendering import gpu


gamelib.init()
instructions = gpu.Renderer(
    shader="""
        #version 330
        
        #vert
        in vec2 v_pos;
        in vec3 v_col;
        uniform vec2 offset;
        out vec3 color;
        
        void main()
        {
            gl_Position = vec4(v_pos + offset, 0, 1);
            color = v_col;
        }
        
        #frag
        in vec3 color;
        out vec4 frag; 
        
        void main() 
        {
            frag = vec4(color, 1);
        }
    """,
    offset=np.array([0.1, -0.1]),
    v_pos=np.array([(-1, -1), (0, 1), (1, -1)]),
    v_col=np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
)
gamelib.set_draw_commands(instructions.render)
gamelib.run()
