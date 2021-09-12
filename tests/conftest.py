import pathlib

import pytest


def isolated_test_run():
    # run tests in a file without default project config
    null_ini_path = pathlib.Path(__file__).parent.parent / 'null.ini'
    pytest.main(["-c", str(null_ini_path)])


