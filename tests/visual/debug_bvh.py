import numpy as np
import gamelib

from gamelib.geometry import collisions
from gamelib.geometry import transforms
from gamelib.rendering import gpu


gamelib.init()

model = gamelib.geometry.load_model("knight")
model.anchor((0.5, 0.5, 0))
bvh = collisions.BVH.create_tree(model)
camera = gamelib.rendering.PerspectiveCamera(
    (3, 3, 4), (-1, -1, -1), controller=True
)

knight_instructions = gpu.Renderer(
    "simple_faceted",
    auto=False,
    indices=model.triangles,
    v_pos=model.vertices,
    view=camera.view_matrix,
    proj=camera.projection_matrix,
    model=transforms.Mat4.identity(),
)


def convert_node_to_vertices(node, color):
    # fmt: off
    vertices = np.array([
        tuple(node.aabb.min), (node.aabb.min.x, node.aabb.max.y, node.aabb.min.z),
        tuple(node.aabb.min), (node.aabb.max.x, node.aabb.min.y, node.aabb.min.z),
        (node.aabb.max.x, node.aabb.min.y, node.aabb.min.z), (node.aabb.max.x, node.aabb.max.y, node.aabb.min.z),
        (node.aabb.min.x, node.aabb.max.y, node.aabb.min.z), (node.aabb.max.x, node.aabb.max.y, node.aabb.min.z),
        tuple(node.aabb.max), (node.aabb.min.x, node.aabb.max.y, node.aabb.max.z),
        tuple(node.aabb.max), (node.aabb.max.x, node.aabb.min.y, node.aabb.max.z),
        (node.aabb.min.x, node.aabb.max.y, node.aabb.max.z), (node.aabb.min.x, node.aabb.min.y, node.aabb.max.z),
        (node.aabb.max.x, node.aabb.min.y, node.aabb.max.z), (node.aabb.min.x, node.aabb.min.y, node.aabb.max.z),
        tuple(node.aabb.min), (node.aabb.min.x, node.aabb.min.y, node.aabb.max.z),
        (node.aabb.min.x, node.aabb.max.y, node.aabb.min.z), (node.aabb.min.x, node.aabb.max.y, node.aabb.max.z),
        (node.aabb.max.x, node.aabb.min.y, node.aabb.min.z), (node.aabb.max.x, node.aabb.min.y, node.aabb.max.z),
        tuple(node.aabb.max), (node.aabb.max.x, node.aabb.max.y, node.aabb.min.z)
    ], gamelib.gl.vec3)
    # fmt: on
    colors = np.empty(len(vertices), gamelib.gl.vec4)
    colors[:] = color
    return vertices, colors


node = bvh
prev = []


def get_bvh_vertex_data():
    if not rendering_only_leaves:
        nodes = [node]
        vertices, colors = convert_node_to_vertices(node, (0, 1, 0, 1))
        if node.left is not None:
            vl, cl = convert_node_to_vertices(node.left, (1, 0, 0, 1))
            vertices = np.concatenate((vertices, vl))
            colors = np.concatenate((colors, cl))
        if node.right is not None:
            vr, cr = convert_node_to_vertices(node.right, (0, 0, 1, 1))
            vertices = np.concatenate((vertices, vr))
            colors = np.concatenate((colors, cr))
    else:
        nodes = [node for node in bvh if node.triangles is not None]
        vert_list = []
        color_list = []
        for n in nodes:
            v, c = convert_node_to_vertices(n, (0, 1, 0, 1))
            vert_list.append(v)
            color_list.append(c)
        vertices = np.concatenate(vert_list)
        colors = np.concatenate(color_list)

    return vertices, colors


rendering_only_leaves = False
should_render_model = True
vertices, colors = get_bvh_vertex_data()
bvh_visualizer = gpu.Renderer(
    shader="""
        #version 330
        #vert
        in vec3 v_pos;
        in vec4 v_color;
        out vec4 f_color;
        uniform mat4 view;
        uniform mat4 proj;
        void main()
        {
            f_color = v_color;
            gl_Position = proj * view * vec4(v_pos, 1.0);
        }
        #frag
        in vec4 f_color;
        out vec4 frag;
        void main()
        {
            frag = f_color;
        }
    """,
    mode=gamelib.gl.LINES,
    v_pos=vertices,
    v_color=colors,
    view=camera.view_matrix,
    proj=camera.projection_matrix,
)


def update_visualizer_buffer():
    vertices, colors = get_bvh_vertex_data()
    bvh_visualizer.source(v_pos=vertices, v_color=colors)


def advance_left():
    global node

    if node.left is not None:
        prev.append(node)
        node = node.left
    update_visualizer_buffer()


def advance_right():
    global node

    if node.right is not None:
        prev.append(node)
        node = node.right
    update_visualizer_buffer()


def prev_node():
    global node

    if not prev:
        return

    node = prev.pop(-1)
    update_visualizer_buffer()


def toggle_model():
    global should_render_model
    should_render_model = not should_render_model


def toggle_leaves():
    global rendering_only_leaves
    rendering_only_leaves = not rendering_only_leaves
    update_visualizer_buffer()


def info_dump():
    node_count = 0
    leaf_count = 0
    leaf_tri_count = 0
    for node in bvh:
        node_count += 1
        if node.indices is not None:
            leaf_tri_count += len(node.indices)
            leaf_count += 1
    print(
        f"{node_count=}, {leaf_count=} avg_density={leaf_tri_count / leaf_count}"
    )
    print(
        f"models_triangles={len(model.triangles)}, bvh_triangles={leaf_tri_count}"
    )


schema = gamelib.InputSchema(
    ("left", "press", advance_left),
    ("right", "press", advance_right),
    ("up", "press", prev_node),
    ("tab", "press", toggle_model),
    ("p", "press", info_dump),
    ("l", "press", toggle_leaves),
    ("esc", "press", gamelib.exit),
)

while gamelib.is_running:
    gamelib.clear()
    bvh_visualizer.render()
    if should_render_model:
        knight_instructions.render()
    gamelib.update()

