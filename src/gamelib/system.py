from __future__ import annotations

import multiprocessing as mp
import traceback
from collections import defaultdict
from multiprocessing.connection import Connection
from typing import Type, List

from . import Update, SystemStop, events, sharedmem, Config, EntityDestroyed
from .component import ComponentCreated, BaseComponent
from .events import eventhandler, Event


class _SystemMeta(type):
    """
    Used as the metaclass for System.

    This creates a BaseComponent 'Component' in the class namespace.
    cls.Component is unique for each subclass of System.
    """

    Component: Type[BaseComponent]

    def __new__(mcs, *args, **kwargs):
        cls: _SystemMeta = super().__new__(mcs, *args, **kwargs)
        # cls.Component.SYSTEM = cls
        return cls

    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        namespace: dict = super().__prepare__(name, bases, **kwargs)
        namespace["Component"] = type("SystemBaseComponent", (BaseComponent,), {})
        return namespace

    @property
    def public_attributes(cls):
        """
        Gets a list of all PublicAttribute instances owned by subclasses of cls.Component

        Returns
        -------
        atts : List[src.gamelib.component.PublicAttribute]
        """
        return [
            attr
            for comp_type in cls.Component.__subclasses__()
            for attr in comp_type.get_public_attributes()
        ]

    @property
    def shared_specs(cls):
        """
        Get a list of all the shm ArraySpecs this system implements.

        Returns
        -------
        specs : Iterable[ArraySpec]
        """
        return [spec for attr in cls.public_attributes for spec in attr.shared_specs]

    @property
    def array_attributes(cls):
        """
        Gets a list of all ArrayAttribute instances defined on subclasses of cls.Component

        Returns
        -------
        attrs : List[src.gamelib.component.ArrayAttribute]
        """
        return [
            attr
            for comp_type in cls.Component.__subclasses__()
            for attr in comp_type.get_array_attributes()
        ]


class System(metaclass=_SystemMeta):
    """
    A System will be run in a multiprocessing.Process and communicates
    with the parent process through a multiprocessing.Pipe.

    All subclasses of System will have their own unique Component Type attributes.

    All Components of a System should inherit from these base types creating an explicit
    link between a System and its Components.

    For example given this system:
        class Physics(System):
            ...
        All components used by this system should derive from Physics.Component
    """

    Component: Type[BaseComponent]
    _running: bool

    def __init__(self):
        Config.local_components.extend(self.Component.__subclasses__())
        events.register_marked(self)
        self._component_lookup = defaultdict(dict)

    def update(self):
        """
        Stub for subclass defined behavior.

        The Update event will first invoke this method, then post all events
        pooled in the event queue, and finally send a SystemUpdateComplete back
        to the main process.
        """

    def stop(self):
        for component in self.Component.__subclasses__():
            Config.local_components.remove(component)
        self._running = False
        events.unregister_marked(self)

    def get_component(self, type_, entity_id):
        try:
            return self._component_lookup[entity_id][type_]
        except KeyError:
            return None

    @eventhandler(EntityDestroyed)
    def _destroy_related_components(self, event: EntityDestroyed):
        for component in self._component_lookup[event.id].values():
            component.destroy()
        self._component_lookup[event.id].clear()

    @eventhandler(ComponentCreated)
    def _create_component(self, event: ComponentCreated):
        component = event.type(event.entity_id, *event.args)
        self._component_lookup[event.entity_id][event.type] = component

    @eventhandler(Update)
    def _update_handler(self, event):
        self.update()


class ProcessSystem(System):
    def __init__(self, conn):
        self._conn = conn
        self._running = False
        self._event_queue = []
        super().__init__()

    @classmethod
    def run_in_process(cls, **kwargs):
        """
        Method that should be called from the main process to actually start the system.

        Config.MAX_ENTITIES should be set before calling if not default.
        If using PublicAttributes sharedmem should be allocated first.

        Returns
        -------
        local : Connection
            multiprocessing.Pipe
        process: Process
        """
        local, foreign = mp.Pipe()
        process = mp.Process(
            target=cls._run,
            args=(foreign, Config.MAX_ENTITIES),
            kwargs=kwargs,
        )
        process.start()
        return local, process

    @classmethod
    def _run(cls, conn, max_entities):
        """Internal process entry point. Sets some global state and clears forked state."""
        Config.MAX_ENTITIES = max_entities
        Config.local_components = []
        events.clear_handlers()
        for attr in cls.array_attributes:
            attr.reallocate()
        inst = cls(conn)
        inst._main()
        cls._teardown_shared_state()

    @classmethod
    def _teardown_shared_state(cls):
        """Local only teardown."""
        for attr in cls.public_attributes:
            attr.close_view()
        sharedmem.close()

    def raise_event(self, event, key=None):
        """Sends an event back to the main process."""
        self._conn.send((event, key))

    def _main(self):
        self._running = True
        while self._running:
            self._poll()

    def _poll(self):
        if not self._conn.poll(0):
            return
        message = self._conn.recv()
        try:
            (event, key) = message
            if isinstance(event, Update) or isinstance(event, SystemStop):
                events.post_event(event, key=key)
            else:
                self._event_queue.append((event, key))
        except Exception as e:
            msg_with_traceback = f"{e}\n\n{traceback.format_exc()}"
            self._conn.send(type(e)(msg_with_traceback))

    @eventhandler(SystemStop)
    def _stop(self, _):
        self.stop()

    @eventhandler(Update)
    def _update_handler(self, _):
        for (event, key) in self._event_queue:
            events.post_event(event, key=key)
        self._event_queue = []
        self.update()
        self.raise_event(SystemUpdateComplete(type(self)))


class SystemUpdateComplete(Event):
    """
    Posted to the main process after a system finishes handling an Update event.
    """

    __slots__ = ["system_type"]

    system_type: Type[System]
