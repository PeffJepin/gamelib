from gamelib.geometry import wavefront
from gamelib.core import resources


def load_model(name):
    if isinstance(name, str):
        path = resources.get_model_file(name)
    else:
        path = name

    if path.name.endswith(".obj"):
        return wavefront.parse(path)

    raise ValueError(f"File format for {path=} not supported.")
