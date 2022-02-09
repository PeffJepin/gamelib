import gamelib

gamelib.init()

instructions = gamelib.rendering.Renderer(
    shader="""
    #version 330
    #vert
    in vec2 v_pos;
    void main()
    {
        gl_Position = vec4(v_pos, 0, 1);
    }

    #frag
    uniform float time;
    void main()
    {
        float r = (sin(time) + 1) / 2;
        float g = (cos(time) + 1) / 2;
        float b = (r + g) / 2;

        gl_FragColor = vec4(r, g, b, 1);
    }
    """,
    v_pos=[(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)],
)

gamelib.set_draw_commands(instructions.render)
gamelib.run()
