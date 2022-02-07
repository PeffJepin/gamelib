import gamelib

from gamelib.rendering import uniforms
from gamelib.rendering import camera


class GlobalUniformBlock(uniforms.UniformBlock):
    cursor = uniforms.ArrayStorage(gamelib.gl.vec2)
    view = uniforms.ArrayStorage(gamelib.gl.mat4)
    proj = uniforms.ArrayStorage(gamelib.gl.mat4)



def _update_global_uniforms(event):
    x, y = gamelib.get_cursor()
    global_uniforms.cursor = (
        x / gamelib.get_width(),
        y / gamelib.get_height(),
    )
    global_uniforms.view = camera.get_primary_view()
    global_uniforms.proj = camera.get_primary_proj()


global_uniforms = GlobalUniformBlock()
gamelib.subscribe(gamelib.core.events.InternalUpdate, _update_global_uniforms)
