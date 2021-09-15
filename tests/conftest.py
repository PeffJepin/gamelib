import pathlib
import time
from collections import Callable
from dataclasses import dataclass
from multiprocessing.connection import PipeConnection
from typing import Tuple

import pytest
from PIL import Image

from src.gamelib.events import Event


class RecordedCallback:
    def __init__(self):
        self.called = False
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.called = True


@dataclass
class ExampleEvent(Event):
    string_field: str
    int_field: int


@pytest.fixture
def recorded_callback() -> RecordedCallback:
    return RecordedCallback()


@pytest.fixture
def example_event() -> Event:
    return ExampleEvent('1', 1)


@pytest.fixture
def image_file_maker(tmpdir) -> Callable[[Tuple[int, int]], pathlib.Path]:
    def _maker(size):
        path = pathlib.Path(tmpdir) / (str(time.time()) + '.png')
        img = Image.new('RGBA', size)
        img.save(path)
        return path
    return _maker


@pytest.fixture
def pipe_reader():
    def _reader(conn: PipeConnection, timeout=10):
        """Tries to return value read from pipe within timeout ms. Returns value read or None."""
        value = None
        for _ in range(timeout):
            if not conn.poll(0):
                time.sleep(1/1_000)
                continue
            value = conn.recv()
            break
        return value
    return _reader
