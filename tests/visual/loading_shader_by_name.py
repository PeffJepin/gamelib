import gamelib
from gamelib.rendering import shaders

gamelib.init()
shader = shaders.ShaderProgram(name="basic")
gamelib.set_draw_commands(lambda: shader.render(3))
gamelib.run()

