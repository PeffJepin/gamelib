import gamelib


class VerboseSchema(gamelib.InputSchema):
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


# slow down gamelib runtime
gamelib.config.fps = 1
gamelib.config.tps = 5
gamelib.run()

