import itertools
from contextlib import contextmanager

import pytest

from src.gamelib import Update, events, EntityCreated, EntityDestroyed, Config
from src.gamelib import environment
from src.gamelib.environment import (
    UpdateComplete,
    EntityFactory,
)
from src.gamelib.events import eventhandler, Event
from src.gamelib.system import SystemUpdateComplete, System, ProcessSystem
from src.gamelib.component import ComponentCreated, ArrayAttribute, PublicAttribute
from src.gamelib.textures import Asset, TextureAtlas
from ..conftest import RecordedCallback

counter = itertools.count(0)


class TestEnvironment:
    def test_assets_have_textures_after_load(self, create_test_env):
        with create_test_env(loaded=True) as env:
            for asset_label in env.ASSET_LABELS:
                asset = env.find_asset(asset_label)

                assert asset.texture is not None

    def test_assets_do_not_have_textures_after_exit(self, create_test_env):
        with create_test_env(loaded=True) as env:
            env.exit()

            for asset_label in env.ASSET_LABELS:
                asset = env.find_asset(asset_label)

                assert asset.texture is None

    def test_does_not_handle_events_before_loading(self, create_test_env):
        with create_test_env() as env:
            events.post_event(Event(), "ABC")

            assert 0 == env.abc_event_handled

    def test_handles_events_after_loading(self, create_test_env):
        with create_test_env(loaded=True) as env:
            events.post_event(Event(), "ABC")

            assert 1 == env.abc_event_handled

    def test_does_not_handle_events_after_exiting(self, create_test_env):
        with create_test_env(loaded=True) as env:
            env.exit()
            events.post_event(Event(), "ABC")

            assert 0 == env.abc_event_handled

    def test_shared_memory_is_initialized_after_load(self, create_test_env):
        with create_test_env(loaded=True):
            assert all(Component1.public_attr[:] == 0)

    def test_shared_memory_is_released_after_exit(self, create_test_env):
        with create_test_env(loaded=True) as env:
            env.exit()
            with pytest.raises(Exception):
                Component1.public_attr += 1

    def test_systems_begin_handling_events_after_load(
        self, create_test_env, recorded_callback
    ):
        events.register(SystemUpdateComplete, recorded_callback)
        with create_test_env(loaded=True):
            events.post_event(Update())
            recorded_callback.wait_for_response(n=2)

            assert 2 == recorded_callback.called

    def test_systems_stop_handling_events_after_exit(
        self, create_test_env, recorded_callback
    ):
        events.register(SystemUpdateComplete, recorded_callback)
        with create_test_env(loaded=True) as env:
            env.exit()
            events.post_event(Update())

            with pytest.raises(TimeoutError):
                recorded_callback.wait_for_response(timeout=0.1)

    def test_systems_shut_down_after_exit(self, create_test_env):
        with create_test_env(loaded=True) as env:
            assert env._running_processes
            env.exit()
            assert not env._running_processes

    def test_public_attr_correct_length_in_process(
        self, create_test_env, recorded_callback
    ):
        events.register(Response.QUERY_PUBLIC_ATTR_LENGTH, recorded_callback)
        with create_test_env(loaded=True, max_entities=111):
            events.post_event(Event(), key="QUERY_PUBLIC_ATTR_LENGTH")
            events.post_event(Update())
            recorded_callback.wait_for_response()
            assert 111 == recorded_callback.event.value

    def test_array_attr_correct_length_in_process(
        self, create_test_env, recorded_callback
    ):
        events.register(Response.QUERY_ARRAY_ATTR_LENGTH, recorded_callback)
        with create_test_env(max_entities=12, loaded=True):
            events.post_event(Event(), key="QUERY_ARRAY_ATTR_LENGTH")
            events.post_event(Update())
            recorded_callback.wait_for_response()

            assert 12 == recorded_callback.event.value

    def test_posts_event_update_complete_after_processes_finish_updating(
        self, create_test_env, recorded_callback
    ):
        events.register(UpdateComplete, recorded_callback)
        with create_test_env(loaded=True):
            events.post_event(Update())
            # this will raise timeout error on test fail
            recorded_callback.wait_for_response()

    def test_public_attribute_buffers_update_automatically_after_update(self, create_test_env, recorded_callback):
        events.register(UpdateComplete, recorded_callback)
        with create_test_env(loaded=True):
            events.post_event(Event(), key="INCREMENT_PUBLIC_ATTR")
            events.post_event(Update())
            recorded_callback.wait_for_response()

            assert all(Component1.public_attr == 1)

    def test_public_attr_access_from_multiple_processes(self, create_test_env):
        response_watcher = RecordedCallback()
        update_watcher = RecordedCallback()
        events.register(Response.PUBLIC_ATTR_MULTIPLE_ACCESS, response_watcher)
        events.register(UpdateComplete, update_watcher)

        with create_test_env(loaded=True) as env:
            for i in range(10):
                events.post_event(Event(), key="PUBLIC_ATTR_MULTIPLE_ACCESS")
                events.post_event(Update())
                update_watcher.wait_for_response()

                assert i + 1 == Component1.public_attr[0]  # swap has happened automatically
                assert i == response_watcher.event.value  # swap had not happened yet within process

    def test_config_local_components_accurately_tracks_components_on_a_process(
        self, create_test_env, recorded_callback
    ):
        events.register(Response.TEST_CONFIG_LOCAL_COMPONENTS, recorded_callback)
        with create_test_env(loaded=True):
            events.post_event(Event(), key="TEST_CONFIG_LOCAL_COMPONENTS")
            events.post_event(Update())
            recorded_callback.await_called(2)

            assert Response([Component1]) in recorded_callback.events
            assert Response([Component2]) in recorded_callback.events
            assert Config.local_components == [LocalComponent]


class TestEntityFactory:
    @pytest.fixture
    def listener(self, recorded_callback):
        events.register(EntityCreated, recorded_callback)
        events.register(ComponentCreated, recorded_callback)
        return recorded_callback

    def test_posts_entity_created_event(self, listener):
        factory = EntityFactory()

        factory.create()

        assert isinstance(listener.event, EntityCreated)

    def test_entity_id_is_incremented_for_each_entity_created(self, listener):
        factory = EntityFactory()

        factory.create()
        assert EntityCreated(id=0) == listener.event

        factory.create()
        assert EntityCreated(id=1) == listener.event

    def test_raises_index_error_if_max_entities_are_exceeded(self):
        factory = EntityFactory(max_entities=10)

        for _ in range(10):
            factory.create()

        with pytest.raises(IndexError):
            factory.create()

    def test_posts_component_created_events(self, listener):
        factory = EntityFactory()

        factory.create((Component1, "9"), (Component1, "1"))

        assert (
            ComponentCreated(entity_id=0, type=Component1, args=("9",))
            in listener.events
        )
        assert (
            ComponentCreated(entity_id=0, type=Component1, args=("1",))
            in listener.events
        )

    def test_entity_ids_are_recycled_when_an_entity_is_destroyed(self, listener):
        factory = EntityFactory()

        factory.create()
        assert EntityCreated(id=0) == listener.event

        events.post_event(EntityDestroyed(id=0))
        factory.create()
        assert EntityCreated(id=0) == listener.event

    def test_recycling_avoids_overflowing_max_entities(self):
        factory = EntityFactory(max_entities=20)

        for _ in range(2):
            # create to capacity and destroy all a few times
            for _ in range(20):
                factory.create()

            for i in range(20):
                events.post_event(EntityDestroyed(id=i))

        for _ in range(20):
            # create exactly to capacity
            factory.create()

        with pytest.raises(IndexError):
            # overflow
            factory.create()


class System1(ProcessSystem):
    @eventhandler(Event.QUERY_PUBLIC_ATTR_LENGTH)
    def public_attr_length_response(self, _):
        res = Response(len(Component1.public_attr))
        self.raise_event(res, key="QUERY_PUBLIC_ATTR_LENGTH")

    @eventhandler(Event.QUERY_ARRAY_ATTR_LENGTH)
    def array_attr_length_response(self, _):
        res = Response(len(Component1.array_attr))
        self.raise_event(res, key="QUERY_ARRAY_ATTR_LENGTH")

    @eventhandler(Event.PUBLIC_ATTR_MULTIPLE_ACCESS)
    def multiple_access_increment(self, _):
        Component1.public_attr[0] += 1

    @eventhandler(Event.TEST_CONFIG_LOCAL_COMPONENTS)
    def local_system_check(self, _):
        self.raise_event(
            Response(Config.local_components), key="TEST_CONFIG_LOCAL_COMPONENTS"
        )

    @eventhandler(Event.INCREMENT_PUBLIC_ATTR)
    def public_attr_incrementer(self, _):
        Component1.public_attr += 1


class Response(Event):
    __slots__ = ["value"]


class Component1(System1.Component):
    public_attr = PublicAttribute(int)
    array_attr = ArrayAttribute(int)


class System2(ProcessSystem):
    @eventhandler(Event.PUBLIC_ATTR_MULTIPLE_ACCESS)
    def multiple_access_increment(self, _):
        res = Response(Component1.public_attr[0])
        self.raise_event(res, key="PUBLIC_ATTR_MULTIPLE_ACCESS")

    @eventhandler(Event.TEST_CONFIG_LOCAL_COMPONENTS)
    def local_system_check(self, _):
        self.raise_event(
            Response(Config.local_components), key="TEST_CONFIG_LOCAL_COMPONENTS"
        )


class Component2(System2.Component):
    pass


class LocalSystem(System):
    pass


class LocalComponent(LocalSystem.Component):
    pass


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

        SYSTEMS = [LocalSystem, System1, System2]

        @eventhandler(Event.ABC)
        def abc_handler(self, _):
            self.abc_event_handled += 1

        @eventhandler(SystemUpdateComplete)
        def update_complete_counter(self, _):
            self.updates_completed += 1

    @contextmanager
    def env_manager(max_entities=100, loaded=False):
        ExampleEnvironment._MAX_ENTITIES = max_entities
        env = ExampleEnvironment()
        try:
            if loaded:
                env.load(fake_ctx)
            yield env
        finally:
            env.exit()

    return env_manager
