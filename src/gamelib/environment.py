from __future__ import annotations

import abc
from multiprocessing.connection import Connection
from typing import List, Type, Dict

from . import SystemStop, events, sharedmem, Config, EntityCreated, EntityDestroyed
from .events import eventhandler
from .system import System, SystemUpdateComplete, ProcessSystem
from .component import ComponentCreated
from .textures import Asset, TextureAtlas


class UpdateComplete(events.Event):
    pass


class Environment(abc.ABC):
    ASSETS: list
    SYSTEMS: List[Type[System]]
    _MAX_ENTITIES: int = 1024

    def __init__(self):
        """
        Environment handles the lifecycle of Systems, Entities and Components.
        This includes maintaining required assets.

        An Environment is not 'loaded' on __init__ and must call load() before
        being used. Exit should be called when the Environment is no longer in use.
        """
        self._index_assets()
        self._loaded = False
        self._system_update_complete_counter = 0
        self._running_processes = dict()
        self._local_systems = []

    def load(self, ctx):
        """
        Loads resources needed by this Environments Systems and registers
        with the MessageBus.

        Parameters
        ----------
        ctx : moderngl.Context
            Rendering context to upload GFX assets to.
        """
        Config.MAX_ENTITIES = self._MAX_ENTITIES
        events.register_marked(self)
        self._load_assets(ctx)
        self._init_shm()
        self._start_systems()
        self._loaded = True

    def exit(self):
        """
        Cleans up resources this Environment is using and exits the MessageBus.
        """
        self._loaded = False
        self._shutdown_systems()
        for system in self.SYSTEMS:
            for attr in system.public_attributes:
                attr.close_view()
        sharedmem.unlink()
        self._release_assets()
        events.unregister_marked(self)

    def find_asset(self, label):
        """Returns reference to some Asset by Asset.label value."""
        return self._asset_lookup.get(label, None)

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
        events.post_event(SystemStop())
        for _, (process, conn) in self._running_processes.items():
            events.stop_connection_service(conn)
            process.join()
        for system in self._local_systems:
            system.stop()
        self._running_processes.clear()
        self._local_systems.clear()

    def _start_systems(self):
        for system_type in self.SYSTEMS:
            if issubclass(system_type, ProcessSystem):
                conn, process = system_type.run_in_process()
                system_handler_types = events.find_eventhandlers(system_type).keys()
                events.service_connection(conn, *system_handler_types)
                self._running_processes[system_type] = (process, conn)
            else:
                self._local_systems.append(system_type())

    def _init_shm(self):
        specs = sum((system.shared_specs for system in self.SYSTEMS), [])
        sharedmem.allocate(specs)

    @eventhandler(SystemUpdateComplete)
    def _track_system_updates(self, _):
        self._system_update_complete_counter += 1
        if self._system_update_complete_counter != len(self._running_processes):
            return
        self._system_update_complete_counter = 0
        events.post_event(UpdateComplete())

    @eventhandler(UpdateComplete)
    def _update_public_attributes(self, _):
        if not self._loaded:
            return
        for system_type in self.SYSTEMS:
            for attr in system_type.public_attributes:
                attr.update_buffer()


class EntityFactory:
    def __init__(self, max_entities=1024):
        events.register_marked(self)
        self._id_handout = list(range(max_entities))
        self._max_entities = max_entities

    def create(self, *components):
        entity_id = self._id_handout.pop(0)

        for comp_spec in components:
            type_, *args = comp_spec
            event = ComponentCreated(entity_id=entity_id, type=type_, args=tuple(args))
            events.post_event(event)

        event = EntityCreated(entity_id)
        events.post_event(event)

    @eventhandler(EntityDestroyed)
    def _recycle_entity_id(self, event: EntityDestroyed):
        idx = 0
        for i, id_ in enumerate(self._id_handout):
            if id_ > event.id:
                idx = i
                break
        self._id_handout.insert(idx, event.id)
