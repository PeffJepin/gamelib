import gamelib

gamelib.init()

model = gamelib.geometry.load_model("knight")
camera = gamelib.rendering.PerspectiveCamera(
    (-3, -3, 3), (1, 1, -1), controller=True
)

for x in range(30):
    for y in range(30):
        gamelib.ecs.Transform.create((4 * x, 4 * y, 0), theta=4 * x * y)

instructions = gamelib.rendering.Renderer(
    """
    #version 330

    #vert
    in vec3 v_pos;
    in mat4 model;

    uniform mat4 proj;
    uniform mat4 view;

    out vec3 world_pos;

    void main()
    {
        vec4 world_space_transform = model * vec4(v_pos, 1.0);
        world_pos = world_space_transform.xyz;
        gl_Position = proj * view * world_space_transform;
    }

    #frag
    in vec3 world_pos;
    out vec4 frag;

    vec3 light_color = vec3(0.9, 0.6, 0.3);
    vec3 light_pos = vec3(-400.0, -125.0, 1000.0);
    float ambient_strength = 0.15;

    void main()
    {
        vec3 dx = dFdx(world_pos);
        vec3 dy = dFdy(world_pos);
        vec3 face_normal = normalize(cross(dx, dy));
        vec3 light_dir = normalize(light_pos - world_pos);

        float diffuse_strength = max(dot(face_normal, light_dir), 0.0);
        vec3 diffuse = diffuse_strength * light_color;
        vec3 ambient = light_color * ambient_strength;

        frag = vec4(ambient + diffuse, 1.0);
    }
    """,
    instanced=("model",),
    indices=model.indices,
    v_pos=model.vertices,
    proj=camera.projection_matrix,
    view=camera.view_matrix,
    model=gamelib.ecs.Transform.model_matrix,
)

gamelib.set_draw_commands(instructions.render)
gamelib.run()
