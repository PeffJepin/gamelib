import gamelib

from gamelib.input import InputSchema


class VerboseSchema(InputSchema):
    def __call__(self, event):
        print(event)
        super().__call__(event)


window = gamelib.init()
schema = VerboseSchema()


while not window.is_closing:
    window.clear()
    window.swap_buffers()

