import pytest
import multiprocessing as mp

from src.gamelib import ecs


def test_adding_a_static_global():
    key = "my key"
    ecs.add_static_global(key, 100)

    assert ecs.get_static_global(key) == 100


def test_registering_a_shared_array():
    key = "some key"
    array = mp.Array("i", 16)

    ecs.register_shared_array(key, array)

    assert ecs.get_shared_array(key) == array


def test_removing_a_shared_array():
    key = "some key"
    array = mp.Array("i", 16)

    ecs.register_shared_array(key, array)
    ecs.remove_shared_array(key)

    with pytest.raises(KeyError):
        ecs.get_shared_array(key)


def test_resetting_globals():
    key = "key"
    ecs.add_static_global(key, 123)

    ecs.reset_globals()

    with pytest.raises(KeyError):
        ecs.get_static_global(key)


def test_resetting_globals_with_some_initial_values():
    ecs.reset_globals({"key1": 123, "key2": 456})

    assert ecs.get_static_global("key1") == 123
    assert ecs.get_static_global("key2") == 456


def test_importing_and_exporting():
    array = mp.Array("i", 123)
    ecs.add_static_global("static key", 123)
    ecs.register_shared_array("shm", array)

    a, b = mp.Pipe()
    a.send(ecs.export_globals())
    dumped_and_loaded = b.recv()
    ecs.import_globals(dumped_and_loaded)

    assert 123 == ecs.get_static_global("static key")
    assert array == ecs.get_shared_array("shm")


def test_builtin_globals():
    ecs.reset_globals()

    for k in list(ecs.StaticGlobals):
        assert ecs.get_static_global(k)


@pytest.fixture(autouse=True)
def reset_globals():
    ecs.reset_globals()
