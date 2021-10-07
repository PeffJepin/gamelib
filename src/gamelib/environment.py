from __future__ import annotations

import abc
from multiprocessing.connection import Connection
from typing import List, Type, Hashable, Dict

from . import SystemStop
from .events import MessageBus, find_eventhandlers, eventhandler, Event
from .sharedmem import SharedBlock
from .system import System, BaseComponent, SystemUpdateComplete
from .textures import Asset, TextureAtlas


class UpdateComplete(Event):
    pass


class EntityCreated(Event):
    __slots__ = ["id"]

    id: int


class EntityDestroyed(Event):
    __slots__ = ["id"]

    id: int


class ComponentCreated(Event):
    __slots__ = ["entity_id", "type", "args"]

    entity_id: int
    type: Type[BaseComponent]
    args: tuple


class Environment(abc.ABC):
    ASSETS: list

    SYSTEMS: List[Type[System]]
    _MAX_ENTITIES: int = 1024

    _system_connections: Dict[Type[System], Connection]
    _message_bus: MessageBus
    _shm_block: SharedBlock = None

    def __init__(self, message_bus):
        """
        Environment handles the lifecycle of Systems, Entities and Components.
        This includes maintaining required assets.

        An Environment is not 'loaded' on __init__ and must call load() before
        being used. Exit should be called when the Environment is no longer in use.

        Parameters
        ----------
        message_bus : MessageBus
        """
        self._message_bus = message_bus
        self._system_connections = dict()
        self._index_assets()
        self._loaded = False
        self._system_update_complete_counter = 0

    def load(self, ctx):
        """
        Loads resources needed by this Environments Systems and registers
        with the MessageBus.

        Parameters
        ----------
        ctx : moderngl.Context
            Rendering context to upload GFX assets to.
        """
        System.MAX_ENTITIES = self._MAX_ENTITIES
        self._message_bus.register_marked(self)
        self._load_assets(ctx)
        self._init_shm()
        self._start_systems()
        self._loaded = True

    def exit(self):
        """
        Cleans up resources this Environment is using and exits the MessageBus.
        """
        for system in self.SYSTEMS:
            system.teardown_shared_state()
        if not self._loaded:
            return
        self._release_assets()
        self._shutdown_systems()
        self._message_bus.unregister_marked(self)
        self._loaded = False

    def find_asset(self, label):
        """Returns reference to some Asset by Asset.label value."""
        return self._asset_lookup.get(label, None)

    def update_public_attributes(self):
        """
        This should be called from the main thread. Don't try to call it with an event
        handler, since the message bus runs on another thread.

        This shouldn't be a problem, as this function can be called from the top level
        at the same time the frame buffer is swapped.

        This restricts the game update loop to once per frame without some kind of change.
        This procedure would probably benefit a lot from an async implementation.
        """
        for system_type in self.SYSTEMS:
            for attr in system_type.public_attributes:
                attr.update()

    def _index_assets(self):
        self._asset_lookup = dict()
        for item in self.ASSETS:
            if isinstance(item, Asset):
                self._asset_lookup[item.label] = item
            elif isinstance(item, TextureAtlas):
                for asset in item:
                    self._asset_lookup[asset.label] = asset

    def _load_assets(self, ctx):
        for item in self.ASSETS:
            if isinstance(item, TextureAtlas):
                item.upload_texture(ctx)
            elif isinstance(item, Asset):
                item.upload_texture(ctx)

    def _release_assets(self):
        for item in self.ASSETS:
            if isinstance(item, TextureAtlas):
                item.release_texture()
            elif isinstance(item, Asset):
                item.release_texture()

    def _shutdown_systems(self):
        self._message_bus.post_event(SystemStop())
        for _, (process, conn) in self._running_systems.items():
            self._message_bus.stop_connection_service(conn)
            process.join()
        self._running_systems = None

    def _start_systems(self):
        self._running_systems = dict()
        for system in self.SYSTEMS:
            conn, process = system.run_in_process(self._MAX_ENTITIES)
            system_handler_types = find_eventhandlers(system).keys()
            self._message_bus.service_connection(conn, *system_handler_types)
            self._running_systems[system] = (process, conn)

    def _init_shm(self):
        System.MAX_ENTITIES = self._MAX_ENTITIES
        specs = sum((system.shared_specs for system in self.SYSTEMS), [])
        System.set_shared_block(SharedBlock(specs, self._MAX_ENTITIES))

    @eventhandler(SystemUpdateComplete)
    def _track_system_updates(self, event):
        self._system_update_complete_counter += 1
        if self._system_update_complete_counter != len(self.SYSTEMS):
            return
        self._system_update_complete_counter = 0
        self._message_bus.post_event(UpdateComplete())


class EntityFactory:
    def __init__(self, message_bus: MessageBus, max_entities=1024):
        self._mb = message_bus
        self._mb.register_marked(self)
        self._id_handout = list(range(max_entities))
        self._max_entities = max_entities

    def create(self, *components):
        entity_id = self._id_handout.pop(0)

        for comp_spec in components:
            type_, *args = comp_spec
            event = ComponentCreated(entity_id=entity_id, type=type_, args=tuple(args))
            self._mb.post_event(event)

        event = EntityCreated(entity_id)
        self._mb.post_event(event)

    @eventhandler(EntityDestroyed)
    def _recycle_entity_id(self, event: EntityDestroyed):
        idx = 0
        for i, id_ in enumerate(self._id_handout):
            if id_ > event.id:
                idx = i
                break
        self._id_handout.insert(idx, event.id)
