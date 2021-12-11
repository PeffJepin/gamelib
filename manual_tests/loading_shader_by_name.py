import gamelib
from gamelib import shaders


gamelib.init()
shader = shaders.ShaderProgram(name="basic")
gamelib.set_draw_commands(lambda: shader.render(3))
gamelib.run()

