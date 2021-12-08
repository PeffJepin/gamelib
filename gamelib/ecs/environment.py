from __future__ import annotations

from typing import List, Type, NamedTuple

from gamelib import (
    events,
)
from gamelib.events import handler
from . import EntityDestroyed, StaticGlobals, reset_globals
from .component import Component
from .system import System, SystemRunner
from gamelib.textures import TextureAtlas


class UpdateComplete(NamedTuple):
    pass


class Environment:
    """An Environment wraps a group of Components and Systems
    and manages setting up / tearing down state when used
    as a context manager.

    The visual assets needed by this environment should be listed
    here so that by inspecting the Environment you can see which
    Assets should be loaded, but the Environment itself is not
    responsible for initializing the video memory.

    Systems can be listed as Local or Process Systems. As one would
    expect, local systems are run on this main process, while process
    Systems are run in their own processes.

    The attributes used by an environment can be defined on the type
    if Environment is subclassed, or passed into __init__ as parameters.
    Passing values into __init__ takes precedence over being defined on
    the Type object.
    """

    ASSETS: list
    COMPONENTS: List[Type[Component]]
    LOCAL_SYSTEMS: List[Type[System]] = []
    PROCESS_SYSTEMS: List[Type[System]] = []
    MAX_ENTITIES: int = 1024

    def __init__(
        self,
        assets=None,
        components=None,
        local_systems=None,
        process_systems=None,
        max_entities=None,
    ):
        """Parameters passed in here take precedence over those defined
        in the class body. They are optional.

        Parameters
        ----------
        assets : list[Asset]
        components : list[type[Component]]
            The components used by this Environment.
        local_systems : list[type[System]]
            The Systems to be run on the main process.
        process_systems : list[type[System]]
            The Systems to be run in their own processes.
        max_entities : int
            The Components underlying arrays will be allocated
            to this length.
        """

        # Use argument over default defined on the Type.
        self.ASSETS = assets or self.ASSETS
        self.COMPONENTS = components or self.COMPONENTS
        self.LOCAL_SYSTEMS = local_systems or self.LOCAL_SYSTEMS
        self.PROCESS_SYSTEMS = process_systems or self.PROCESS_SYSTEMS
        self.MAX_ENTITIES = max_entities or self.MAX_ENTITIES

        self._loaded = False
        self._index_assets()
        self._system_update_complete_counter = 0
        self._system_runners = []
        self._local_systems = []
        self._entity_pool = []

    def __enter__(self):
        """An Environment should be used as a context manager to
        set up all the required state for running Systems.

        Entering the context manager allocated shared memory,
        starts systems locally and in processes and starts the
        Environment responding to events.
        """
        reset_globals({StaticGlobals.MAX_ENTITIES: self.MAX_ENTITIES})
        self._entity_pool = list(range(self.MAX_ENTITIES))
        events.subscribe_obj(self)
        for component in self.COMPONENTS:
            component.allocate()
        self._start_systems()
        self._loaded = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Teardown what was setup in __enter__."""
        events.unsubscribe_obj(self)
        self._shutdown_systems()
        for component in self.COMPONENTS:
            component.free()
        self._loaded = False

    def create_entity(self, *components):
        """Write component data into shared arrays and unmask entity.

        Parameters
        ----------
        components : Iterable[Component]
            The Components associated with this entity.
            The entity id will be assigned automatically.
            Entities are recycled after being destroyed.
        """
        entity = self._entity_pool.pop(0)
        for component in components:
            component.bind_to_entity(entity)
        return entity

    def find_asset(self, label):
        """Returns reference to some Asset by Asset.label value."""
        return self._asset_lookup.get(label, None)

    def _index_assets(self):
        self._asset_lookup = dict()
        for item in self.ASSETS:
            if isinstance(item, TextureAtlas):
                for asset in item:
                    self._asset_lookup[asset.label] = asset
            self._asset_lookup[item.label] = item

    def _start_systems(self):
        for system in self.LOCAL_SYSTEMS:
            self._local_systems.append(system())
        for system in self.PROCESS_SYSTEMS:
            runner = SystemRunner(system)
            runner.start()
            self._system_runners.append(runner)

    def _shutdown_systems(self):
        for runner in self._system_runners:
            runner.join()
        for system in self._local_systems:
            system.stop()
        self._system_runners.clear()
        self._local_systems.clear()

    @handler(EntityDestroyed)
    def _recycle_entity(self, event: EntityDestroyed):
        entity = event.id
        for component in self.COMPONENTS:
            component.destroy(entity)
        for i, pooled_entity in enumerate(self._entity_pool):
            if entity < pooled_entity:
                self._entity_pool.insert(i, entity)
                return
