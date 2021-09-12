from src.gamelib.system import System
from tests.conftest import isolated_test_run


class TestSystem:
    class ExampleSystem(System):
        def update(self):
            pass

    def test_can_communicate_with_child_process(self):
        pass


if __name__ == '__main__':
    isolated_test_run()
