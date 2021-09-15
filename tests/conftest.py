import pathlib
import time
from dataclasses import dataclass
from multiprocessing.connection import PipeConnection
from typing import Tuple, Callable

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
    return ExampleEvent("1", 1)


@pytest.fixture
def image_file_maker(tmpdir) -> Callable[[Tuple[int, int]], pathlib.Path]:
    def _maker(size):
        path = pathlib.Path(tmpdir) / (str(time.time()) + ".png")
        img = Image.new("RGBA", size)
        img.save(path)
        return path

    return _maker


@pytest.fixture
def pipe_reader():
    def _reader(conn: PipeConnection, timeout=1000):
        """
        Tries to return value read from pipe within timeout ms. Returns value read or None.

        Default timeout potentially a little high for quick testing, but in practice it should
        return after only a few cycles on a passing test, and really only aught to take the full
        1s on failures. This should ensure that test results aren't flaky due to not having enough time
        to read the pipe.
        """
        value = None
        for _ in range(timeout):
            if not conn.poll(0):
                time.sleep(1 / 1_000)
                continue
            value = conn.recv()
            break
        return value

    return _reader
