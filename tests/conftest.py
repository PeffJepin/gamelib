import pathlib
import time
from multiprocessing.connection import Connection
from typing import Tuple, Callable

import pytest
from PIL import Image

from src.gamelib.textures import Asset


class RecordedCallback:
    def __init__(self):
        self.called = 0
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.called += 1


@pytest.fixture
def recorded_callback() -> RecordedCallback:
    return RecordedCallback()


@pytest.fixture
def image_file_maker(tmpdir) -> Callable[[Tuple[int, int]], pathlib.Path]:
    def _maker(size):
        path = pathlib.Path(tmpdir) / (str(time.time()) + ".png")
        img = Image.new("RGBA", size)
        img.save(path)
        return path

    return _maker


@pytest.fixture
def asset_maker(image_file_maker):
    def inner(w, h):
        size = (w, h)
        return Asset(str(time.time()), image_file_maker(size))

    return inner


@pytest.fixture
def pipe_reader():
    def _reader(conn: Connection, timeout=5):
        if not conn.poll(timeout):
            return None
        else:
            return conn.recv()

    return _reader
