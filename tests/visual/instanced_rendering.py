import numpy as np
import gamelib


gamelib.init()

instructions = gamelib.rendering.Renderer(
    shader="""
        #version 330
        #vert
        in vec2 v_pos;
        in vec2 v_off;
        in vec3 v_col;

        out vec3 f_col;

        void main()
        {
            gl_Position = vec4(v_pos + v_off, 0, 1);
            f_col = v_col;
        }

        #frag
        in vec3 f_col;
        out vec4 frag;

        void main()
        {
            frag = vec4(f_col, 1.0);
        }
    """,
    v_pos=np.array(
        [(-0.1, -0.1), (-0.1, 0.1), (0.1, 0.1), (0.1, -0.1)], gamelib.gl.vec2
    ),
    indices=np.array([0, 1, 2, 0, 2, 3], gamelib.gl.uint),
    v_off=np.array(
        [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)], gamelib.gl.vec2
    ),
    v_col=np.array(
        [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)], gamelib.gl.vec3
    ),
    instanced=("v_col", "v_off"),
)

gamelib.set_draw_commands(instructions.render)
gamelib.run()
