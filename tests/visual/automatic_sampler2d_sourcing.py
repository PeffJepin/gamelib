import gamelib

gamelib.init()

instructions = gamelib.rendering.Renderer(
    """
    #version 330
    #vert

    in vec2 v_pos;
    in vec2 uv;

    out vec2 f_uv;

    void main()
    {
        f_uv = uv;
        gl_Position = vec4(v_pos, 0, 1);
    }

    #frag

    in vec2 f_uv;

    uniform sampler2D grad1;

    void main()
    {
        gl_FragColor = texture(grad1, f_uv);
    }
    """,
    v_pos=[(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)],
    uv=[(0, 0), (0, 1), (1, 1), (0, 0), (1, 1), (1, 0)],
)

gamelib.set_draw_commands(instructions.render)
gamelib.run()
