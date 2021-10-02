import logging
import multiprocessing as mp
import pathlib
import time
from multiprocessing.connection import Connection
from typing import Tuple, Callable

import pytest
from PIL import Image

from src.gamelib.system import System
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

    def wait_for_response(self, timeout=1, n=1):
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
    def _reader(conn: Connection, timeout=1, n=1):
        messages = []
        for _ in range(n):
            if not conn.poll(timeout):
                return None
            else:
                incoming = conn.recv()
                if isinstance(incoming, Exception):
                    raise incoming
                messages.append(incoming)
        return messages if n > 1 else messages[0]

    return _reader


class SystemRunner(mp.Process):
    """
    Process that wraps a running system.
    """

    available_connections = []
    busy_connections = []
    all_processes = []

    STOP = "STOP"
    GET_STATUS = "GET STATUS"
    READY = "READY"
    BUSY = "BUSY"

    def __init__(self, conn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conn = conn

    def run(self):
        while True:
            try:
                self._main()
            except Exception as e:
                self.conn.send(e)

    def _main(self):
        if not self.conn.poll(0.05):
            return
        message = self.conn.recv()
        if message == self.GET_STATUS:
            return self.conn.send(self.READY)
        elif isinstance(message, tuple):
            sys_type, conn, max_entities = message
            self.conn.send("STARTING SYSTEM")
            sys_type._run(conn, self.conn, max_entities)


class PatchedSystem(System):
    """
    Subclassing this in test code avoids booting up a new process for every test
    """

    def __init__(self, conn, _runner_conn):
        super().__init__(conn)
        self._runner_conn = _runner_conn

    def _poll(self):
        super()._poll()
        if not self._runner_conn.poll(0):
            return
        message = self._runner_conn.recv()
        if message == SystemRunner.STOP:
            self._running = False
        elif message == SystemRunner.GET_STATUS:
            self._runner_conn.send(SystemRunner.BUSY)

    @classmethod
    def _run(cls, conn, _runner_conn, max_entities=1024):
        cls.MAX_ENTITIES = max_entities
        for attr in cls.array_attributes:
            attr.reallocate()
        inst = cls(conn, _runner_conn)
        inst._main()

    @classmethod
    def run_in_process(cls, max_entities):
        runner_connection = cls.__get_connection()
        a, b = mp.Pipe()
        runner_connection.send((cls, b, max_entities))
        assert runner_connection.poll(1)
        assert runner_connection.recv() == "STARTING SYSTEM"
        return a, (MockProcess(runner_connection))

    @classmethod
    def __get_connection(cls):
        if not SystemRunner.available_connections:
            a, b = mp.Pipe()
            runner = SystemRunner(b)
            runner.start()
            SystemRunner.all_processes.append(runner)
            SystemRunner.available_connections.append(a)
        conn = SystemRunner.available_connections.pop(0)
        SystemRunner.busy_connections.append(conn)
        return conn


class MockProcess:
    """
    Allows a PatchedSystem to return an object that can be
    joined as if it were running in a normal process.
    """

    def __init__(self, system_runner_connection):
        self.exitcode = None
        self.conn = system_runner_connection

    def join(self, timeout=1):
        self.conn.send(SystemRunner.STOP)
        self.conn.send(SystemRunner.GET_STATUS)
        ts = time.time()
        while time.time() < ts + timeout:
            if self.conn.poll(0.05):
                message = self.conn.recv()
                if message == SystemRunner.READY:
                    SystemRunner.available_connections.append(self.conn)
                    SystemRunner.busy_connections.remove(self.conn)
                    self.exitcode = 0
                    return
                if message == SystemRunner.BUSY:
                    self.conn.send(SystemRunner.GET_STATUS)
        raise TimeoutError("Could not get response from SystemRunner process.")

    def kill(self):
        self.join()


def pytest_sessionfinish(session, exitstatus):
    for p in SystemRunner.all_processes:
        p.kill()


@pytest.fixture(autouse=True, scope="session")
def setup_logging():
    logging.basicConfig(level=logging.DEBUG)
