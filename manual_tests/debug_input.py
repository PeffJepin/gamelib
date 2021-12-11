import gamelib

from gamelib.input import InputSchema


class VerboseSchema(InputSchema):
    def __call__(self, event):
        print(event)
        super().__call__(event)


def dummy():
    pass


# is_pressed event wont be checked it nothing is subscribed
schema = VerboseSchema(
    ("a", "is_pressed", dummy),
    ("s", "is_pressed", dummy),
    ("d", "is_pressed", dummy),
    ("f", "is_pressed", dummy)
)


gamelib.config._max_tps = 5
gamelib.run()

