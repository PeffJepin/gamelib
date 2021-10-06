import itertools
from contextlib import contextmanager

import pytest

from src.gamelib import Update
from src.gamelib import environment
from src.gamelib.environment import UpdateComplete
from src.gamelib.events import MessageBus, eventhandler, Event
from src.gamelib.sharedmem import SharedBlock
from src.gamelib.system import (
    PublicAttribute,
    SystemUpdateComplete,
    ArrayAttribute,
    System,
)
from src.gamelib.textures import Asset, TextureAtlas
from ..conftest import PatchedSystem, RecordedCallback

counter = itertools.count(0)


class TestEnvironment:
    def test_assets_have_textures_after_load(self, create_test_env):
        with create_test_env(loaded=True) as env:
            for asset_label in env.ASSET_LABELS:
                asset = env.find_asset(asset_label)

                assert asset.texture is not None

    def test_assets_dont_have_textures_after_exit(self, create_test_env):
        with create_test_env(loaded=True) as env:
            env.exit()

            for asset_label in env.ASSET_LABELS:
                asset = env.find_asset(asset_label)

                assert asset.texture is None

    def test_does_not_handle_events_before_loading(self, create_test_env):
        mb = MessageBus()
        with create_test_env(mb) as env:
            mb.post_event(Event(), "ABC")

            assert 0 == env.abc_event_handled

    def test_handles_events_after_loading(self, create_test_env):
        mb = MessageBus()
        with create_test_env(mb, loaded=True) as env:
            mb.post_event(Event(), "ABC")

            assert 1 == env.abc_event_handled

    def test_does_not_handle_events_after_exiting(self, create_test_env):
        mb = MessageBus()
        with create_test_env(mb, loaded=True) as env:
            env.exit()
            mb.post_event(Event(), "ABC")

            assert 0 == env.abc_event_handled

    def test_shared_memory_is_initialized_after_load(self, create_test_env):
        with create_test_env(loaded=True) as env:
            assert all(Component1.public_attr[:] == 0)

    def test_shared_memory_is_released_after_exit(self, create_test_env):
        with create_test_env(loaded=True) as env:
            env.exit()
            with pytest.raises(Exception):
                should_raise_error = Component1.public_attr

    def test_systems_begin_handling_events_after_load(
        self, create_test_env, recorded_callback
    ):
        mb = MessageBus()
        mb.register(SystemUpdateComplete, recorded_callback)
        with create_test_env(mb, loaded=True) as env:
            mb.post_event(Update())
            recorded_callback.wait_for_response(n=2)

            assert 2 == recorded_callback.called

    def test_systems_stop_handling_events_after_exit(
        self, create_test_env, recorded_callback
    ):
        mb = MessageBus()
        mb.register(SystemUpdateComplete, recorded_callback)
        with create_test_env(mb, loaded=True) as env:
            env.exit()
            mb.post_event(Update())

            with pytest.raises(TimeoutError):
                recorded_callback.wait_for_response(timeout=0.1)

    def test_systems_shut_down_after_exit(self, create_test_env):
        mb = MessageBus()
        with create_test_env(mb, loaded=True) as env:
            assert env._running_systems is not None
            env.exit()
            assert env._running_systems is None

    def test_public_attr_correct_length_in_process(
        self, create_test_env, recorded_callback
    ):
        mb = MessageBus()
        mb.register(Response.QUERY_PUBLIC_ATTR_LENGTH, recorded_callback)
        with create_test_env(mb, loaded=True, max_entities=111) as env:
            mb.post_event(Event(), key="QUERY_PUBLIC_ATTR_LENGTH")
            mb.post_event(Update())
            recorded_callback.wait_for_response()
            assert 111 == recorded_callback.args[0].value

    def test_array_attr_correct_length_in_process(
        self, create_test_env, recorded_callback
    ):
        mb = MessageBus()
        mb.register(Response.QUERY_ARRAY_ATTR_LENGTH, recorded_callback)
        with create_test_env(mb, max_entities=12, loaded=True) as env:
            mb.post_event(Event(), key="QUERY_ARRAY_ATTR_LENGTH")
            mb.post_event(Update())
            recorded_callback.wait_for_response()

            assert 12 == recorded_callback.args[0].value

    def test_posts_event_complete_after_update_completes(
        self, create_test_env, recorded_callback
    ):
        mb = MessageBus()
        mb.register(UpdateComplete, recorded_callback)
        with create_test_env(mb, loaded=True) as env:
            mb.post_event(Update())
            # this will raise timeout error on test fail
            recorded_callback.wait_for_response()

    def test_public_attr_access_from_multiple_processes(self, create_test_env):
        mb = MessageBus()
        response_watcher = RecordedCallback()
        update_watcher = RecordedCallback()
        mb.register(Response.PUBLIC_ATTR_MULTIPLE_ACCESS, response_watcher)
        mb.register(UpdateComplete, update_watcher)

        with create_test_env(mb, loaded=True) as env:
            for i in range(10):
                mb.post_event(Event(), key="PUBLIC_ATTR_MULTIPLE_ACCESS")
                mb.post_event(Update())
                response_watcher.wait_for_response()

                assert i == Component1.public_attr[0]
                assert i == response_watcher.args[0].value

                if update_watcher.called <= i:
                    update_watcher.wait_for_response()
                env.update_public_attributes()


class System1(PatchedSystem):
    @eventhandler(Event.QUERY_PUBLIC_ATTR_LENGTH)
    def public_attr_length_response(self, event):
        res = Response(len(Component1.public_attr))
        self.post_event(res, key="QUERY_PUBLIC_ATTR_LENGTH")

    @eventhandler(Event.QUERY_ARRAY_ATTR_LENGTH)
    def array_attr_length_response(self, event):
        res = Response(len(Component1.array_attr))
        self.post_event(res, key="QUERY_ARRAY_ATTR_LENGTH")

    @eventhandler(Event.PUBLIC_ATTR_MULTIPLE_ACCESS)
    def multiple_access_increment(self, event):
        Component1.public_attr[0] += 1


class Response(Event):
    __slots__ = ["value"]


class Component1(System1.Component):
    public_attr = PublicAttribute(int)
    array_attr = ArrayAttribute(int)


class System2(PatchedSystem):
    @eventhandler(Event.PUBLIC_ATTR_MULTIPLE_ACCESS)
    def multiple_access_increment(self, event):
        res = Response(Component1.public_attr[0])
        self.post_event(res, key="PUBLIC_ATTR_MULTIPLE_ACCESS")


@pytest.fixture
def create_test_env(image_file_maker, fake_ctx):
    class ExampleEnvironment(environment.Environment):
        # injected for testing
        ASSET_LABELS = [f"asset{i + 1}" for i in range(6)]
        abc_event_handled = 0
        updates_completed = 0

        # normal implementation
        ASSETS = [
            Asset("asset1", image_file_maker((8, 8))),
            Asset("asset2", image_file_maker((8, 8))),
            Asset("asset3", image_file_maker((8, 8))),
            TextureAtlas(
                [
                    Asset("asset4", image_file_maker((8, 8))),
                    Asset("asset5", image_file_maker((8, 8))),
                    Asset("asset6", image_file_maker((8, 8))),
                ],
                allocation_step=8,
                max_size=(64, 64),
            ),
        ]

        SYSTEMS = [System1, System2]

        def _init_shm(self):
            shared_specs = sum((system.shared_specs for system in self.SYSTEMS), [])
            shared_block = SharedBlock(shared_specs, System.MAX_ENTITIES, name_extra="")
            System.set_shared_block(shared_block)

        @eventhandler(Event.ABC)
        def abc_handler(self, event):
            self.abc_event_handled += 1

        @eventhandler(SystemUpdateComplete)
        def update_complete_counter(self, event):
            self.updates_completed += 1

    @contextmanager
    def env_manager(mb=None, *, max_entities=100, loaded=False):
        env = ExampleEnvironment(mb or MessageBus())
        env._MAX_ENTITIES = max_entities
        try:
            if loaded:
                env.load(fake_ctx)
            yield env
        finally:
            env.exit()

    return env_manager


@pytest.fixture(autouse=True)
def auto_cleanup():
    for attr in System1.public_attributes:
        attr.is_open = False
    for attr in System2.public_attributes:
        attr.is_open = False
