import gamelib

from gamelib import rendering
from gamelib import geometry


gamelib.init()
context = gamelib.get_context()
context.wireframe = True
scale = 1000
lod = 100

mesh = geometry.GridMesh(lod=lod, scale=scale)
perspective = rendering.PerspectiveCamera(
    pos=(-33, -33, 100),
    dir=(500, 500, 1),
    up=(0, 0, 1),
    fovy=50,
    far=10_000,
    controller=True
)
ortho = rendering.OrthogonalCamera(
    px_per_unit=1,
    pos=(scale/2, scale/2, -5),
    up=(0, 1, 0),
    dir=(0, 0, -1),
    controller=True
)
camera = perspective


shader = rendering.ShaderProgram(
    vertex_shader="""
        #version 330
        
        in vec3 pos;
        out vec3 color;
        
        uniform mat4 view;
        uniform mat4 proj;
        
        void main()
        {
            gl_Position = proj * view * vec4(pos, 1);
            float diagonal = sqrt(1000*1000 + 1000*1000);
            float red = distance(pos.xy, vec2(800, 200)) / diagonal;
            float blue = distance(pos.xy, vec2(200, 800)) / diagonal;
            float green = distance(pos.xy, vec2(0, 0)) / diagonal;
            color = vec3(red, green, blue);
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
    buffers={"pos": mesh.vertices},
    uniforms={"view": camera.view_matrix, "proj": camera.projection_matrix},
    index_buffer=mesh.indices,
)


gamelib.set_draw_commands(shader.render)
gamelib.run()
