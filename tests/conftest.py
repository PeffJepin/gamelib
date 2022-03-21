import pathlib
import shutil
import time
import itertools
import tempfile
from multiprocessing.connection import Connection

import pytest
from PIL import Image

from gamelib.core import events
from gamelib.core import resources
from gamelib.rendering.textures import ImageAsset


_counter = itertools.count(0)
_tmp: pathlib.Path = None


@pytest.fixture(autouse=True, scope="session")
def setup_temp_directory():
    global _tmp
    with tempfile.TemporaryDirectory() as tmp:
        _tmp = pathlib.Path(tmp)
        yield
    _tmp = None


@pytest.fixture
def tempdir():
    new_dir = _tmp / str(next(_counter))
    new_dir.mkdir()
    yield new_dir
    shutil.rmtree(new_dir, ignore_errors=True)


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
        events.subscribe(event_key, self)

    def await_called(self, num_times_called, timeout=5):
        ts = time.time()
        while time.time() < ts + timeout:
            if self.called == num_times_called:
                return
        raise TimeoutError(
            f"Target times called = {num_times_called}. "
            f"Current times called = {self.called}"
        )

    def await_silence(self, timeout=0.1, _max_timeout=5):
        ts = time.time()
        while time.time() < ts + _max_timeout:
            try:
                self.wait_for_response(timeout=timeout)
            except TimeoutError:
                return
        raise TimeoutError(f"Still getting callbacks after {_max_timeout=}s")

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
def image_file_maker(tempdir):
    def inner(size, name=None):
        name = name or str(next(_counter))
        path = tempdir / (name + ".png")
        img = Image.new("RGBA", size)
        img.save(path)
        return path

    yield inner


@pytest.fixture
def asset_maker(image_file_maker):
    def inner(w, h):
        size = (w, h)
        name = str(next(_counter))
        asset = ImageAsset(name, image_file_maker(size, name=name))
        asset.load()
        return asset

    yield inner


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
def tmpdir_maker(tempdir):
    def inner(*paths):
        fs_root = tempdir / str(next(_counter))
        paths = [
            fs_root / p
            if isinstance(p, pathlib.Path)
            else fs_root / pathlib.Path(p)
            for p in paths
        ]
        for path in paths:
            for p in path.parents:
                p.mkdir(exist_ok=True, parents=True)
            path.touch()
        return fs_root

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


@pytest.fixture
def shaderdir(tempdir):
    directory = tempdir / "shaders"
    directory.mkdir()
    yield directory
    shutil.rmtree(directory)


@pytest.fixture
def write_shader_to_disk(shaderdir):
    def writer(filename, src):
        fn = filename if filename.endswith(".glsl") else filename + ".glsl"
        with open(shaderdir / fn, "w") as f:
            f.write(src)
        # update resource module after writing new files
        resources.set_content_roots(shaderdir)

    return writer


def assert_approx(iter1, iter2, rel=1e-6):
    for v1, v2 in zip(iter1, iter2):
        # I've seen a test failure here saying relative tolerance
        # can't be negative, but it's not reproducible.
        # If this continues starts failing with any consistency
        # something will have to change.
        assert v1 == pytest.approx(v2, rel=rel)


def compare_glsl(src1, src2):
    cleaned1 = "\n".join(
        line.strip()
        for line in src1.splitlines()
        if line.strip() not in ("", "\n")
    )
    cleaned2 = "\n".join(
        line.strip()
        for line in src2.splitlines()
        if line.strip() not in ("", "\n")
    )
    if cleaned1 != cleaned2:
        print("shader1 source")
        print(cleaned1)
        print()
        print("shader2 source")
        print(cleaned2)
    return cleaned1 == cleaned2
