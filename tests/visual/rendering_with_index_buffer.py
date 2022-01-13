from random import random
import time

import numpy as np

import gamelib
from gamelib.rendering import shaders

gamelib.init()

# layout data for a grid of quads
positions = []
colors = []
for x in range(10):
    for y in range(10):
        bottom_left = (-1 + 0.2 * x, -1 + 0.2 * y)
        positions.append(bottom_left)
        positions.append((bottom_left[0], bottom_left[1] + 0.2))
        positions.append((bottom_left[0] + 0.2, bottom_left[1] + 0.2))
        positions.append((bottom_left[0] + 0.2, bottom_left[1]))

        for _ in range(4):
            colors.append((random(), random(), random()))

# create buffers
index_buffer = shaders.OrderedIndexBuffer(
    (0, 1, 2, 0, 2, 3), num_entities=1, max_entities=100
)
pos_buffer = np.array(positions)
col_buffer = np.array(colors)

# create shader
shader = shaders.ShaderProgram(
    source="""
        #version 330
        
        #vert
        in vec2 v_pos;
        in vec3 v_col;
        out vec3 color;
        
        void main()
        {
            gl_Position = vec4(v_pos, 0, 1);
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
    index_buffer=index_buffer,
    buffers={"v_pos": pos_buffer, "v_col": col_buffer},
    max_entities=100,
)

prev_time = time.time()

while gamelib.is_running():
    # every 50ms increment how many quads should be rendered
    if time.time() - prev_time > 0.05:
        if index_buffer.num_entities == 100:
            index_buffer.num_entities = 1
        else:
            index_buffer.num_entities += 1
        prev_time = time.time()

    gamelib.clear()
    shader.render()
    gamelib.update()
