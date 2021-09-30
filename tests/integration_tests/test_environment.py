import time
from contextlib import contextmanager

import pytest

from src.gamelib import Update
from src.gamelib import environment
from src.gamelib.events import MessageBus, eventhandler, BaseEvent
from src.gamelib.system import PublicAttribute, UpdateComplete, System
from src.gamelib.textures import Asset, TextureAtlas


class TestEnvironment:
    def test_assets_have_textures_after_load(self, fake_ctx, make_test_env):
        with make_test_env() as env:
            env.load(fake_ctx)

            for asset_label in env.ASSET_LABELS:
                asset = env.find_asset(asset_label)
                assert asset.texture is not None

    def test_assets_dont_have_textures_after_exit(self, fake_ctx, make_test_env):
        with make_test_env() as env:
            env.load(fake_ctx)
            env.exit()

            for asset_label in env.ASSET_LABELS:
                asset = env.find_asset(asset_label)
                assert asset.texture is None

    def test_does_not_handle_events_before_loading(self, make_test_env):
        mb = MessageBus()
        with make_test_env(mb) as env:
            mb.post_event(BaseEvent(), "ABC")
            assert 0 == env.abc_event_handled

    def test_handles_events_after_loading(self, fake_ctx, make_test_env):
        mb = MessageBus()
        with make_test_env(mb) as env:
            env.load(fake_ctx)
            mb.post_event(BaseEvent(), "ABC")

            assert 1 == env.abc_event_handled

    def test_does_not_handle_events_after_exiting(self, fake_ctx, make_test_env):
        mb = MessageBus()
        with make_test_env(mb) as env:
            env.load(fake_ctx)
            env.exit()
            mb.post_event(BaseEvent(), "ABC")
            assert 0 == env.abc_event_handled

    def test_shared_memory_is_initialized_after_load(self, fake_ctx, make_test_env):
        with make_test_env() as env:
            env.load(fake_ctx)
            assert all(Component1.public_attr[:] == 0)

    def test_shared_memory_is_released_after_exit(self, fake_ctx, make_test_env):
        with make_test_env() as env:
            env.load(fake_ctx)
            env.exit()
            with pytest.raises(FileNotFoundError):
                should_raise_error = Component1.public_attr

    def test_systems_begin_handling_events_after_load(self, make_test_env, fake_ctx):
        mb = MessageBus()
        with make_test_env(mb) as env:
            env.load(fake_ctx)
            mb.post_event(Update())

            # check that updates compelte == 2
            # try to break early, but wait up to ~ 1 second
            for _ in range(1000):
                time.sleep(0.001)
                if env.updates_completed == 2:
                    return
            assert False

    def test_systems_stop_handling_events_after_exit(self, make_test_env, fake_ctx):
        mb = MessageBus()
        with make_test_env(mb) as env:
            env.load(fake_ctx)
            env.exit()
            mb.post_event(Update())
            time.sleep(0.1)
            assert env.updates_completed == 0

    def test_systems_shut_down_after_exit(self, make_test_env, fake_ctx):
        mb = MessageBus()
        with make_test_env(mb) as env:
            env.load(fake_ctx)
            assert env._running_systems is not None
            env.exit()
            assert env._running_systems is None

    @pytest.fixture
    def fake_ctx(self, mocker):
        return mocker.Mock()


class System1(System):
    pass


class Component1(System1.Component):
    public_attr = PublicAttribute(int)


class System2(System):
    pass


@pytest.fixture
def make_test_env(image_file_maker):
    class ExampleEnvironment(environment.Environment):
        # injected for testing
        ASSET_LABELS = [f"asset{i + 1}" for i in range(6)]
        abc_event_handled = 0
        updates_completed = 0

        # example implementation
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

        @eventhandler(BaseEvent.ABC)
        def abc_handler(self, event):
            self.abc_event_handled += 1

        @eventhandler(UpdateComplete)
        def update_complete_counter(self, event):
            self.updates_completed += 1

    @contextmanager
    def self_contained_environment(mb=None):
        env = ExampleEnvironment(mb or MessageBus())
        try:
            yield env
        finally:
            env.exit()

    return self_contained_environment
