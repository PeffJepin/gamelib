import pathlib

import pytest


@pytest.fixture
def recorded_callback():
    return RecordedCallback()


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


