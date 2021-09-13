import pathlib
from dataclasses import dataclass

import pytest

from src.gamelib.events import Event


@pytest.fixture
def recorded_callback():
    return RecordedCallback()


@pytest.fixture
def example_event():
    return ExampleEvent('1', 1)


def isolated_test_run():
    # run tests in a file without default project config
    null_ini_path = pathlib.Path(__file__).parent.parent / 'null.ini'
    pytest.main(["-c", str(null_ini_path)])


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


