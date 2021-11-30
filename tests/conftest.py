import pathlib
import time
from multiprocessing.connection import Connection
from typing import Tuple, Callable

import pytest
from PIL import Image

from gamelib import events
from gamelib.textures import ImageAsset


_tmp_dir_counter = 0


class RecordedCallback:
    def __init__(self):
        self.called = 0
        self.args = []
        self.kwargs = []

    def __call__(self, *args, **kwargs):
        self.args.append(args)
        self.kwargs.append(kwargs)
        self.called += 1

    @property
    def event(self):
        """Returns event from most recent call."""
        return self.args[-1][0]

    @property
    def events(self):
        """Returns all invoking events"""
        return [a[0] for a in self.args]

    def register(self, event_key):
        events.register(event_key, self)

    def await_called(self, num_times_called, timeout=5):
        ts = time.time()
        while time.time() < ts + timeout:
            if self.called == num_times_called:
                return
        raise TimeoutError(
            f"Target times called = {num_times_called}. "
            f"Current times called = {self.called}"
        )

    def wait_for_response(self, timeout=5, n=1):
        start = self.called
        ts = time.time()
        while time.time() < ts + timeout:
            if self.called >= start + n:
                return
        raise TimeoutError("No Response")


@pytest.fixture
def fake_ctx(mocker):
    return mocker.Mock()


@pytest.fixture
def recorded_callback() -> RecordedCallback:
    return RecordedCallback()


@pytest.fixture
def image_file_maker(tmpdir) -> Callable[[Tuple[int, int]], pathlib.Path]:
    def _maker(size):
        global _tmp_dir_counter
        path = pathlib.Path(tmpdir) / (str(_tmp_dir_counter) + ".png")
        _tmp_dir_counter += 1

        img = Image.new("RGBA", size)
        img.save(path)
        return path

    return _maker


@pytest.fixture
def asset_maker(image_file_maker):
    def inner(w, h):
        size = (w, h)
        asset = ImageAsset(str(time.time()), image_file_maker(size))
        asset.load()
        return asset

    return inner


@pytest.fixture
def pipe_reader():
    def _reader(conn: Connection, timeout=1, n=1, index=None):
        messages = []
        for _ in range(n):
            if not conn.poll(timeout):
                raise TimeoutError()
            incoming = conn.recv()
            if isinstance(incoming, Exception):
                raise incoming
            messages.append(incoming if index is None else incoming[index])
        return messages if n > 1 else messages[0]

    return _reader


@pytest.fixture
def filesystem_maker(tmpdir):
    def inner(*paths):
        global _tmp_dir_counter
        root = pathlib.Path(tmpdir) / str(_tmp_dir_counter)
        _tmp_dir_counter += 1

        paths = [
            root / p if isinstance(p, pathlib.Path) else root / pathlib.Path(p)
            for p in paths
        ]
        for path in paths:
            for p in path.parents:
                p.mkdir(exist_ok=True, parents=True)
            path.touch()
        return root

    yield inner


@pytest.fixture(autouse=True, scope="function")
def cleanup_event_handlers():
    events.clear_handlers()


@pytest.fixture(
    params=[
        ("int", 123),
        ("uint", 123),
        ("float", 123),
        ("double", 123),
        ("bool", True),
        ("bvec2", (True, False)),
        ("bvec3", (True, False, True)),
        ("bvec4", (True, False, True, False)),
        ("ivec2", (1, 2)),
        ("ivec3", (1, 2, 3)),
        ("ivec4", (1, 2, 3, 4)),
        ("uvec2", (1, 2)),
        ("uvec3", (1, 2, 3)),
        ("uvec4", (1, 2, 3, 4)),
        ("vec2", (1, 2)),
        ("vec3", (1, 2, 3)),
        ("vec4", (1, 2, 3, 4)),
        ("sampler2D", 2),
        ("mat2", [(1, 2), (3, 4)]),
        ("mat2x3", [(1, 2), (3, 4), (1, 2)]),
        ("mat2x4", [(1, 2), (3, 4), (3, 4), (5, 6)]),
        ("mat3x2", [(1, 2, 3), (3, 4, 5)]),
        ("mat3", [(1, 2, 3), (3, 4, 5), (5, 6, 7)]),
        ("mat3x4", [(1, 2, 3), (3, 4, 5), (5, 6, 7), (7, 8, 9)]),
        ("mat4x2", [(1, 2, 3, 4), (3, 4, 5, 6)]),
        ("mat4x3", [(1, 2, 3, 4), (3, 4, 5, 6), (5, 6, 7, 8)]),
        ("mat4", [(1, 2, 3, 4), (3, 4, 5, 6), (5, 6, 7, 8), (7, 8, 9, 0)]),
    ]
)
def glsl_dtype_and_input(request):
    # parameterized fixture returning a glsl dtype along with a valid
    #   python value that should fit into that dtype
    # note some dtypes are missing, as I'm currently only testing
    # against the minimum supported glsl version #version 330
    dtype, value = request.param
    yield dtype, value
