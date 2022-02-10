import gamelib

from gamelib.core import events
from gamelib.rendering import uniforms
from gamelib.rendering import camera


class GlobalUniformBlock(uniforms.UniformBlock):
    cursor = uniforms.ArrayStorage(gamelib.gl.vec2)
    view = uniforms.ArrayStorage(gamelib.gl.mat4)
    proj = uniforms.ArrayStorage(gamelib.gl.mat4)
    window_size = uniforms.ArrayStorage(gamelib.gl.ivec2)
    time = uniforms.ArrayStorage(gamelib.gl.float)


def _update_global_uniforms(_):
    x, y = gamelib.get_cursor()
    global_uniforms.cursor = (
        x / gamelib.get_width(),
        y / gamelib.get_height(),
    )
    global_uniforms.view = camera.get_primary_view()
    global_uniforms.proj = camera.get_primary_proj()
    global_uniforms.window_size = (gamelib.get_width(), gamelib.get_height())
    global_uniforms.time = gamelib.get_time()


global_uniforms = GlobalUniformBlock()
gamelib.subscribe(events.InternalUpdate, _update_global_uniforms)
