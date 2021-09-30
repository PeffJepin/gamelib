from __future__ import annotations

import abc
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import List, Type, Hashable, Dict

from . import SystemStop
from .events import MessageBus, find_handlers
from .system import System
from .textures import Asset, TextureAtlas


class Environment(abc.ABC):
    ASSETS: list

    SYSTEMS: List[Type[System]]
    _MAX_ENTITIES: int = 1028

    _systems: List[System] = None
    _system_connections: Dict[Type[System], Connection]
    _message_bus: MessageBus

    def __init__(self, _message_bus):
        self._message_bus = _message_bus
        self._system_connections = dict()
        self._index_assets()
        self._loaded = False

    def _index_assets(self):
        self._asset_lookup = dict()
        for item in self.ASSETS:
            if isinstance(item, Asset):
                self._asset_lookup[item.label] = item
            elif isinstance(item, TextureAtlas):
                for asset in item:
                    self._asset_lookup[asset.label] = asset

    def load(self, ctx):
        System.MAX_ENTITIES = self._MAX_ENTITIES
        self._message_bus.register_marked_handlers(self)
        self._load_assets(ctx)
        self._start_systems()
        self._loaded = True

    def exit(self):
        if not self._loaded:
            return
        self._release_assets()
        self._shutdown_systems()
        self._message_bus.unregister_marked_handlers(self)
        self._loaded = False

    def find_asset(self, label):
        return self._asset_lookup.get(label, None)

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
        for SystemType, (process, conn) in self._running_systems.items():
            self._message_bus.stop_connection_service(conn)
            process.join()
            SystemType.teardown_shared_state()
        self._running_systems = None

    def _start_systems(self):
        self._running_systems = dict()
        for SystemType in self.SYSTEMS:
            SystemType.setup_shared_state()
            local_conn, process_conn = mp.Pipe()
            process = mp.Process(
                target=SystemType, args=(process_conn, self._MAX_ENTITIES)
            )
            process.start()
            system_handler_types = find_handlers(SystemType).keys()
            self._message_bus.service_connection(local_conn, system_handler_types)
            self._running_systems[SystemType] = (process, local_conn)


class Entity:
    id: int
    component_types: List[Type[BaseComponent]]
    tags: List[Hashable]


class BaseComponent:
    entity_id: int
