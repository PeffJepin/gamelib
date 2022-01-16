import gamelib
from gamelib.rendering import gpu

gamelib.init()
instructions = gpu.Renderer("basic")
gamelib.set_draw_commands(lambda: instructions.render(3))
gamelib.run()
