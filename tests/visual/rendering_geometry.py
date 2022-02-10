import numpy as np
import gamelib

from gamelib import rendering
from gamelib import geometry
from gamelib import Vec3


gamelib.init()
camera = rendering.PerspectiveCamera(
    position=(0, -2, 0), direction=(0, 1, 0), controller=True
)
camera.set_primary()
cube = geometry.Cube()
transform = geometry.Transform(
    scale=(1, 1, 1), pos=(0, 0, 0), axis=(0, 0, 1), theta=0
)
instructions = rendering.Renderer(
    "simple_faceted",
    v_pos=cube.vertices,
    indices=cube.indices,
    model=transform.matrix,
)

waiting_for_input = False


def next_test():
    global waiting_for_input
    waiting_for_input = False


def quit():
    global waiting_for_input
    waiting_for_input = False
    gamelib.exit()


def draw():
    gamelib.clear()
    instructions.render()
    gamelib.update()


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


prompt("1. Rotating right.")
transform.axis = (0, 0, 1)
while waiting_for_input:
    transform.theta += 1
    draw()


prompt("2. Rotating left.")
while waiting_for_input:
    transform.theta -= 1
    draw()


prompt("3. Rotating right.")
transform.axis = (0, 1, 0)
while waiting_for_input:
    transform.theta += 1
    draw()


prompt("4. Rotating left.")
while waiting_for_input:
    transform.theta -= 1
    draw()


prompt("5. Rotating down.")
transform.axis = (1, 0, 0)
while waiting_for_input:
    transform.theta += 1
    draw()


prompt("6. Rotating up.")
while waiting_for_input:
    transform.theta -= 1
    draw()


camera.position = (-2, -2, -2)
camera.direction = (2, 2, 2)


prompt("7. Rotating right.")
transform.axis = camera.direction
while waiting_for_input:
    transform.theta += 1
    draw()


prompt("8. Rotating left.")
while waiting_for_input:
    transform.theta -= 1
    draw()


transform.theta = 0
camera.position = (0, -2, 0)
camera.direction = (0, 1, 0)


prompt("9. Stretching horizontally.")
scale = Vec3(1, 1, 1)
while waiting_for_input:
    scale += (0.01, 0, 0)
    transform.scale = scale
    draw()


prompt("10. Stretching vertically.")
scale = Vec3(1, 1, 1)
while waiting_for_input:
    scale += (0, 0, 0.01)
    transform.scale = scale
    draw()


camera.position = (0, 0, 2)
camera.direction = (0, 0, -1)
camera.up = (0, 1, 0)


prompt("11. Stretching vertically.")
scale = Vec3(1, 1, 1)
while waiting_for_input:
    scale += (0, 0.01, 0)
    transform.scale = scale
    draw()


transform.scale = (1, 1, 1)
camera.position = (0, -10, 0)
camera.direction = (0, 1, 0)
camera.up = (0, 0, 1)


prompt("12. Moving left.")
pos = Vec3(0)
while waiting_for_input:
    pos.x -= 0.01
    transform.pos = pos
    draw()


prompt("13. Moving right.")
pos = Vec3(0)
while waiting_for_input:
    pos.x += 0.01
    transform.pos = pos
    draw()


prompt("14. Moving up.")
pos = Vec3(0)
while waiting_for_input:
    pos.z += 0.01
    transform.pos = pos
    draw()


prompt("15. Moving down.")
pos = Vec3(0)
while waiting_for_input:
    pos.z -= 0.01
    transform.pos = pos
    draw()


camera.position = (0, 0, 10)
camera.direction = (0, 0, -1)
camera.up = (0, 1, 0)


prompt("16. Moving up.")
pos = Vec3(0)
while waiting_for_input:
    pos.y += 0.01
    transform.pos = pos
    draw()


prompt("17. Moving down.")
pos = Vec3(0)
while waiting_for_input:
    pos.y -= 0.01
    transform.pos = pos
    draw()


camera.position = (0, -5, 0)
camera.direction = (0, 1, 0)
camera.up = (0, 0, 1)


prompt("18. Cycle up/down, cycle width, rotate right.")
i = 0
while waiting_for_input:
    sin = np.sin(i / 66)
    transform.pos = (0, 0, sin)
    transform.scale = (1.5 + sin, 1, 1)
    transform.axis = (0, 0, 1)
    transform.theta = i
    i += 1
    draw()


gamelib.exit()
