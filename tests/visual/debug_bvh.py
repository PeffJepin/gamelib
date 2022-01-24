import sys
import time
import dataclasses
import numpy as np
import gamelib

from gamelib.geometry import collisions
from gamelib.geometry import transforms
from gamelib.rendering import gpu

model_to_load = "knight"
bvh_density = 64
if len(sys.argv) > 1:
    model_to_load = sys.argv[1]
if len(sys.argv) > 2:
    bvh_density = int(sys.argv[2])


gamelib.init()

model = gamelib.geometry.load_model(model_to_load)
model.anchor((0.5, 0.5, 0))
transform = transforms.Transform((2, 3, 4), (10, 10, 10), (1, 1, 1), 0)
bvh = collisions.BVH.create_tree(model, target_density=bvh_density)
camera = gamelib.rendering.PerspectiveCamera(
    (-10, -10, 20), (12, 13, -16), controller=True
)

NUM_MODES = 3
RENDER_INTERSECTIONS = 0
RENDER_NODE_EXPLORER = 1
RENDER_BVH_LEAF_ONLY = 2

RAY_COLOR = (1, 0.2, 1, 1)
CURR_NODE_COLOR = (0.2, 0.8, 0.2, 1)
LEFT_NODE_COLOR = (0.3, 0.0, 0.6, 0.5)
RIGHT_NODE_COLOR = (0.6, 0.0, 0.3, 0.5)
LEAF_NODE_COLOR = (0.3, 1.0, 0.6, 1.0)
BACKGROUND_COLOR = (0.03, 0.005, 0.005, 1)


@dataclasses.dataclass
class State:
    node: collisions.BVH
    prev: list

    ray: collisions.Ray
    ray_tri_intersections: np.ndarray
    ray_aabb_intersections: list

    render_mode: int
    render_model: bool


state = State(
    node=bvh,
    prev=list(),
    ray=camera.screen_to_ray(
        gamelib.get_width() / 2, gamelib.get_height() / 2
    ),
    ray_tri_intersections=None,
    ray_aabb_intersections=list(),
    render_mode=RENDER_NODE_EXPLORER,
    render_model=True,
)

triangles_instructions = gpu.Renderer(
    "simple_faceted",
    view=camera.view_matrix,
    proj=camera.projection_matrix,
    model=transform.matrix,
)

line_shader = """
    #version 330
    #vert
    in vec3 v_pos;
    in vec4 v_color;
    out vec4 f_color;
    uniform mat4 view;
    uniform mat4 proj;
    uniform mat4 model;
    void main()
    {
        f_color = v_color;
        gl_Position = proj * view * model * vec4(v_pos, 1.0);
    }
    #frag
    in vec4 f_color;
    out vec4 frag;
    void main()
    {
        frag = f_color;
    }
    """

ray_instructions = gpu.Renderer(
    shader=line_shader,
    mode=gamelib.gl.LINES,
    view=camera.view_matrix,
    proj=camera.projection_matrix,
    model=transforms.Mat4.identity(),
)

bvh_instructions = gpu.Renderer(
    shader=line_shader,
    mode=gamelib.gl.LINES,
    view=camera.view_matrix,
    proj=camera.projection_matrix,
    model=transform.matrix,
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


def convert_ray_to_vertices(ray, color):
    colors = np.array([color, color], gamelib.gl.vec4)
    p1 = tuple(ray.origin + 1_000 * ray.direction)
    p2 = tuple(ray.origin + -1_000 * ray.direction)
    vertices = np.array([p1, p2], gamelib.gl.vec3)
    colors = np.array([color, color], gamelib.gl.vec4)
    return vertices, colors


def get_bvh_node_contained_triangles(node):
    triangles = []
    for n in node:
        if n.indices is not None:
            triangles.append(n.triangles)
    return np.concatenate(triangles)


def get_bvh_vertex_data():
    if state.render_mode == RENDER_BVH_LEAF_ONLY:
        nodes = [node for node in bvh if node.triangles is not None]
        vert_list = []
        color_list = []
        for n in nodes:
            v, c = convert_node_to_vertices(n, LEAF_NODE_COLOR)
            vert_list.append(v)
            color_list.append(c)
        vertices = np.concatenate(vert_list)
        colors = np.concatenate(color_list)

    elif state.render_mode == RENDER_INTERSECTIONS:
        vert_list = []
        color_list = []
        for n in state.ray_aabb_intersections:
            if n.indices is not None:
                v, c = convert_node_to_vertices(n, LEAF_NODE_COLOR)
            else:
                v, c = convert_node_to_vertices(n, RIGHT_NODE_COLOR)
            vert_list.append(v)
            color_list.append(c)
        vertices = np.concatenate(vert_list)
        colors = np.concatenate(color_list)

    elif state.render_mode == RENDER_NODE_EXPLORER:
        priority = LEAF_NODE_COLOR if state.node.indices is not None else None
        vertices, colors = convert_node_to_vertices(
            state.node, priority or CURR_NODE_COLOR
        )
        if state.node.left is not None:
            lvtx, lclr = convert_node_to_vertices(
                state.node.left, priority or LEFT_NODE_COLOR
            )
            vertices = np.concatenate((vertices, lvtx))
            colors = np.concatenate((colors, lclr))
        if state.node.right is not None:
            rvtx, rclr = convert_node_to_vertices(
                state.node.right, priority or RIGHT_NODE_COLOR
            )
            vertices = np.concatenate((vertices, rvtx))
            colors = np.concatenate((colors, rclr))

    return vertices, colors


def get_ray_vertex_data():
    return convert_ray_to_vertices(state.ray, RAY_COLOR)


def get_triangles_vertex_data():
    if state.render_mode == RENDER_BVH_LEAF_ONLY:
        return model.vertices[model.triangles]
    if state.render_mode == RENDER_NODE_EXPLORER:
        return get_bvh_node_contained_triangles(state.node)
    if state.render_mode == RENDER_INTERSECTIONS:
        return state.ray_tri_intersections


def cast_ray():
    state.ray = camera.cursor_to_ray()

    print(f"[RAYCAST] {state.ray}")
    state.ray.to_object_space(transform)
    ts = time.time()
    distances = collisions.ray_triangle_intersections(
        model.vertices[model.triangles], state.ray.origin, state.ray.direction
    )
    try:
        brute_force = np.min(distances[distances != -1])
    except ValueError:
        brute_force = False
    te = time.time()
    ms = f"{(te-ts)*1_000:.3f} ms"
    print(f"\t[BRUTE FORCE]: result = {brute_force:.3f}, time = {ms}")
    ts = time.time()
    regular = state.ray.collides_bvh(bvh)
    te = time.time()
    ms = f"{(te-ts)*1_000:.3f} ms"
    print(f"\t[FULL BVH]   : result = {regular:.3f}, time = {ms}")
    ts = time.time()
    exit_early = state.ray.collides_bvh(bvh, exit_early=True)
    te = time.time()
    ms = f"{(te-ts)*1_000:.3f} ms"
    print(f"\t[EARLY EXIT] : result = {exit_early:.3f}, time = {ms}")

    tri_list = []
    node_list = []
    ntris = 0

    for node in bvh:
        if not state.ray.collides_aabb(node.aabb):
            continue
        else:
            node_list.append(node)
        if node.indices is not None:
            # trace all hit triangles
            triangles = node.triangles
            ntris += len(triangles)
            result = collisions.ray_triangle_intersections(
                triangles, state.ray.origin, state.ray.direction
            )
            intersecting = triangles[result != -1]
            if len(intersecting) > 0:
                tri_list.append(intersecting)
    if tri_list:
        state.ray_tri_intersections = np.concatenate(tri_list)
    else:
        state.ray_tri_intersections = None
    if node_list:
        state.ray_aabb_intersections = node_list

    print(
        f"\t[AABB INFO]  : AABB collisions = {len(node_list)}, BVH leaf triangles = {ntris}"
    )
    camera.move((0.0001, -0.0001, 0.0001))
    state.ray.reset_transform()
    update_all_buffers()


def update_all_buffers():
    update_bvh_instruction_buffers()
    update_ray_instruction_buffers()
    update_tri_instruction_buffers()


def update_ray_instruction_buffers():
    vertices, colors = get_ray_vertex_data()
    ray_instructions.source(v_pos=vertices, v_color=colors)


def update_tri_instruction_buffers():
    vertices = get_triangles_vertex_data()
    triangles_instructions.source(v_pos=vertices)


def update_bvh_instruction_buffers():
    if (
        state.render_mode == RENDER_INTERSECTIONS
        and state.ray_tri_intersections is None
    ):
        return
    vertices, colors = get_bvh_vertex_data()
    if vertices is not None:
        bvh_instructions.source(v_pos=vertices, v_color=colors)


def advance_node_left():
    if state.node.left is not None:
        state.prev.append(state.node)
        state.node = state.node.left
    update_bvh_instruction_buffers()
    update_tri_instruction_buffers()


def advance_node_right():
    if state.node.right is not None:
        state.prev.append(state.node)
        state.node = state.node.right
    update_bvh_instruction_buffers()
    update_tri_instruction_buffers()


def return_to_previous_node():
    if not state.prev:
        return

    state.node = state.prev.pop(-1)
    update_bvh_instruction_buffers()
    update_tri_instruction_buffers()


def next_mode():
    state.render_mode = (state.render_mode + 1) % NUM_MODES
    update_all_buffers()


def toggle_model():
    state.render_model = not state.render_model


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
        f"[INFO]: {node_count=}, {leaf_count=} avg_density={leaf_tri_count / leaf_count}"
    )
    print(
        f"[INFO]: models_triangles={len(model.triangles)}, bvh_triangles={leaf_tri_count}"
    )


def draw():
    gamelib.clear(*BACKGROUND_COLOR)
    ray_instructions.render()
    if (
        state.render_mode == RENDER_INTERSECTIONS
        and state.ray_tri_intersections is None
    ):
        pass
    else:
        bvh_instructions.render()
    if state.render_model:
        triangles_instructions.render()


schema = gamelib.InputSchema(
    ("left", "press", advance_node_left),
    ("right", "press", advance_node_right),
    ("up", "press", return_to_previous_node),
    ("down", "press", return_to_previous_node),
    ("tab", "press", next_mode),
    ("p", "press", info_dump),
    ("esc", "press", gamelib.exit),
    ("mouse1", "press", cast_ray),
    ("space", "press", toggle_model),
)

update_all_buffers()
gamelib.set_draw_commands(draw)
gamelib.run()
