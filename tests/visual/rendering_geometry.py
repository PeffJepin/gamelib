import numpy as np
import gamelib

from gamelib import rendering
from gamelib import geometry


gamelib.init()
camera = rendering.PerspectiveCamera(
    pos=(0, -2, 0), dir=(0, 1, 0), controller=True
)
cube = geometry.Cube()
transform = geometry.Transform(
    scale=(5, 1, 1), pos=(0, 0, 0), axis=(0, 0, 1), theta=0
)
model_matrix = geometry.Mat4.identity()
instructions = rendering.Renderer(
    "simple_faceted",
    v_pos=cube.vertices,
    indices=cube.triangles,
    proj=camera.projection_matrix,
    view=camera.view_matrix,
    model=model_matrix,
)

waiting_for_input = False


def next_test():
    global waiting_for_input
    waiting_for_input = False


def quit():
    global waiting_for_input
    waiting_for_input = False
    gamelib.exit()
    exit()


schema = gamelib.InputSchema(
    ("y", "down", next_test),
    ("n", "down", quit),
    ("q", "down", quit),
    ("escape", "down", quit),
    ("c", "down", "ctrl", quit),
)


def prompt(msg):
    global waiting_for_input
    print(msg)
    print("Continue? y/n")
    print("")
    waiting_for_input = True


prompt("Rotating right.")
theta = 0
while waiting_for_input:
    gamelib.clear()
    theta += 1
    model_matrix[:] = geometry.Mat4.rotate_about_z(theta)
    instructions.render()
    gamelib.update()


prompt("Rotating left.")
theta = 0
while waiting_for_input:
    gamelib.clear()
    theta -= 1
    model_matrix[:] = geometry.Mat4.rotate_about_z(theta)
    instructions.render()
    gamelib.update()


prompt("Rotating right.")
theta = 0
while waiting_for_input:
    gamelib.clear()
    theta += 1
    model_matrix[:] = geometry.Mat4.rotate_about_y(theta)
    instructions.render()
    gamelib.update()


prompt("Rotating left.")
theta = 0
while waiting_for_input:
    gamelib.clear()
    theta -= 1
    model_matrix[:] = geometry.Mat4.rotate_about_y(theta)
    instructions.render()
    gamelib.update()


prompt("Rotating down.")
theta = 0
while waiting_for_input:
    gamelib.clear()
    theta += 1
    model_matrix[:] = geometry.Mat4.rotate_about_x(theta)
    instructions.render()
    gamelib.update()


prompt("Rotating up.")
theta = 0
while waiting_for_input:
    gamelib.clear()
    theta -= 1
    model_matrix[:] = geometry.Mat4.rotate_about_x(theta)
    instructions.render()
    gamelib.update()


camera.pos = (-2, -2, -2)
camera.direction = (2, 2, 2)


prompt("Rotating right.")
theta = 0
while waiting_for_input:
    gamelib.clear()
    theta += 1
    model_matrix[:] = geometry.Mat4.rotate_about_axis(camera.direction, theta)
    instructions.render()
    gamelib.update()


prompt("Rotating left.")
theta = 0
while waiting_for_input:
    gamelib.clear()
    theta -= 1
    model_matrix[:] = geometry.Mat4.rotate_about_axis(camera.direction, theta)
    instructions.render()
    gamelib.update()


camera.pos = (0, -2, 0)
camera.direction = (0, 1, 0)


prompt("Stretching horizontally.")
scale = 1
while waiting_for_input:
    gamelib.clear()
    scale += 0.02
    model_matrix[:] = geometry.Mat4.scale((scale, 1, 1))
    instructions.render()
    gamelib.update()


prompt("Stretching vertically.")
scale = 1
while waiting_for_input:
    gamelib.clear()
    scale += 0.02
    model_matrix[:] = geometry.Mat4.scale((1, 1, scale))
    instructions.render()
    gamelib.update()


camera.pos = (0, 0, 2)
camera.direction = (0, 0, -1)
camera.up = (0, 1, 0)


prompt("Stretching vertically.")
scale = 1
while waiting_for_input:
    gamelib.clear()
    scale += 0.02
    model_matrix[:] = geometry.Mat4.scale((1, scale, 1))
    instructions.render()
    gamelib.update()


camera.pos = (0, -10, 0)
camera.direction = (0, 1, 0)
camera.up = (0, 0, 1)


prompt("Moving left.")
offset = 0
while waiting_for_input:
    gamelib.clear()
    offset += 0.02
    model_matrix[:] = geometry.Mat4.translation((-offset, 0, 0))
    instructions.render()
    gamelib.update()


prompt("Moving right.")
offset = 0
while waiting_for_input:
    gamelib.clear()
    offset += 0.02
    model_matrix[:] = geometry.Mat4.translation((offset, 0, 0))
    instructions.render()
    gamelib.update()


prompt("Moving up.")
offset = 0
while waiting_for_input:
    gamelib.clear()
    offset += 0.02
    model_matrix[:] = geometry.Mat4.translation((0, 0, offset))
    instructions.render()
    gamelib.update()


prompt("Moving down.")
offset = 0
while waiting_for_input:
    gamelib.clear()
    offset += 0.02
    model_matrix[:] = geometry.Mat4.translation((0, 0, -offset))
    instructions.render()
    gamelib.update()


camera.pos = (0, 0, 10)
camera.direction = (0, 0, -1)
camera.up = (0, 1, 0)


prompt("Moving up.")
offset = 0
while waiting_for_input:
    gamelib.clear()
    offset += 0.02
    model_matrix[:] = geometry.Mat4.translation((0, offset, 0))
    instructions.render()
    gamelib.update()


prompt("Moving down.")
offset = 0
while waiting_for_input:
    gamelib.clear()
    offset += 0.02
    model_matrix[:] = geometry.Mat4.translation((0, -offset, 0))
    instructions.render()
    gamelib.update()


camera.pos = (0, -5, 0)
camera.direction = (0, 1, 0)
camera.up = (0, 0, 1)


prompt("Cycle up/down, cycle width, rotate right.")
i = 0
while waiting_for_input:
    gamelib.clear()
    sin = np.sin(i / 66)
    t = geometry.Transform(
        pos=(0, 0, sin),
        scale=(1.5 + sin, 1, 1),
        axis=(0, 0, 1),
        theta=i,
    )
    model_matrix[:] = t.matrix
    instructions.render()
    gamelib.update()
    i += 1


gamelib.exit()
