import itertools
import logging
import multiprocessing as mp
import pathlib
import time
import traceback
from multiprocessing.connection import Connection
from typing import Tuple, Callable

import pytest
from PIL import Image

from src.gamelib import events
from src.gamelib.events import clear_handlers
from src.gamelib.sharedmem import SharedBlock
from src.gamelib.system import ProcessSystem, PublicAttribute, System
from src.gamelib.textures import Asset

counter = itertools.count(10_000)


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
                raise TimeoutError()
            else:
                incoming = conn.recv()
                if isinstance(incoming, Exception):
                    raise incoming
                messages.append(incoming)
        return messages if n > 1 else messages[0]

    return _reader


@pytest.fixture
def pipe_await_event():
    def _reader(conn, event_type, timeout=1):
        ts = time.time()
        while time.time() < ts + timeout:
            if not conn.poll(0):
                continue
            message = conn.recv()
            if isinstance(message, event_type):
                return message
            elif isinstance(message, tuple) and isinstance(message[0], event_type):
                return message[0]
        raise TimeoutError("Didn't find event_type while polling connection.")

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
                msg_with_traceback = f"{e}\n\n{traceback.format_exc()}"
                self.conn.send(type(e)(msg_with_traceback))

    def _main(self):
        if not self.conn.poll(0.05):
            return
        message = self.conn.recv()
        if message == self.GET_STATUS:
            return self.conn.send(self.READY)
        elif isinstance(message, tuple):
            events.clear_handlers()
            system, conn, max_entities, shm_block = message
            self.conn.send("STARTING SYSTEM")
            system._run(conn, max_entities, shm_block, _runner_conn=self.conn)

    @classmethod
    def get_connection(cls):
        if not cls.available_connections:
            a, b = mp.Pipe()
            runner = SystemRunner(b)
            runner.start()
            cls.all_processes.append(runner)
            cls.available_connections.append(a)
        conn = cls.available_connections.pop(0)
        cls.busy_connections.append(conn)
        return conn


class PatchedSystem(ProcessSystem):
    """
    Subclassing this in test code avoids booting up a new process for every test
    """

    def __init__(self, conn, _runner_conn):
        super().__init__(conn)
        self._runner_conn = _runner_conn

    def _poll(self):
        """Service pipe runner connection first."""
        if self._runner_conn.poll(0):
            message = self._runner_conn.recv()
            if message == SystemRunner.STOP:
                self._running = False
            elif message == SystemRunner.GET_STATUS:
                self._runner_conn.send(SystemRunner.BUSY)
        super()._poll()

    @classmethod
    def run_in_process(cls, max_entities, **kwargs):
        """Dispatch to an already running process."""
        runner_connection = SystemRunner.get_connection()
        local_conn, internal_conn = mp.Pipe()
        start_system_command = (
            cls,
            internal_conn,
            max_entities,
            PublicAttribute.SHARED_BLOCK,
        )
        runner_connection.send(start_system_command)

        assert runner_connection.poll(3)
        message = runner_connection.recv()
        if isinstance(message, Exception):
            raise message
        assert message == "STARTING SYSTEM"
        return local_conn, (MockProcess(runner_connection))

    @classmethod
    def make_test_shm_block(cls):
        return SharedBlock(
            cls.shared_specs, System.MAX_ENTITIES, name_extra=next(counter)
        )


class MockProcess:
    """
    Allows a PatchedSystem to return an object that can be
    joined as if it were running in a normal process.
    """

    def __init__(self, system_runner_connection):
        self.exitcode = None
        self.runner_conn = system_runner_connection

    def join(self, timeout=1):
        self.runner_conn.send(SystemRunner.STOP)
        self.runner_conn.send(SystemRunner.GET_STATUS)
        ts = time.time()
        while time.time() < ts + timeout:
            if self.runner_conn.poll(0.05):
                message = self.runner_conn.recv()
                if isinstance(message, Exception):
                    raise message
                if message == SystemRunner.READY:
                    SystemRunner.available_connections.append(self.runner_conn)
                    SystemRunner.busy_connections.remove(self.runner_conn)
                    self.exitcode = 0
                    return
                if message == SystemRunner.BUSY:
                    self.runner_conn.send(SystemRunner.GET_STATUS)
        raise TimeoutError("Could not get response from SystemRunner process.")

    def kill(self):
        self.join()


def pytest_sessionfinish(session, exitstatus):
    for p in SystemRunner.all_processes:
        p.kill()


@pytest.fixture(autouse=True, scope="session")
def setup_logging():
    logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(autouse=True, scope="function")
def cleanup_global_shm():
    if PublicAttribute.SHARED_BLOCK is not None:
        PublicAttribute.SHARED_BLOCK.unlink_shm()
        PublicAttribute.SHARED_BLOCK = None


@pytest.fixture(autouse=True, scope="function")
def cleanup_event_handlers():
    clear_handlers()
