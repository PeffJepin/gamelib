from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Type

import numpy as np

from . import Update, SystemStop
from .events import MessageBus, eventhandler, Event
from .sharedmem import DoubleBufferedArray


class BaseComponent:
    SYSTEM: Type[System]

    def __init__(self, entity_id):
        self.entity_id = entity_id


class _SystemMeta(type):
    """
    Used as the metaclass for System.

    This creates a BaseComponent 'Component' in the class namespace.
    cls.Component is unique for each subclass of System.
    """

    Component: Type[BaseComponent]

    def __new__(mcs, *args, **kwargs):
        cls: _SystemMeta = super().__new__(mcs, *args, **kwargs)
        cls.Component.SYSTEM = cls
        return cls

    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        namespace: dict = super().__prepare__(name, bases, **kwargs)
        namespace["Component"] = type("SystemBaseComponent", (BaseComponent,), {})
        return namespace

    @property
    def public_attributes(cls):
        """
        Gets a list of all PublicAttribute instances defined on subclasses of cls.Component

        Returns
        -------
        attrs : list[PublicAttribute]
        """
        attrs = []
        for comp_type in cls.Component.__subclasses__():
            discovered = [
                v for k, v in vars(comp_type).items() if isinstance(v, PublicAttribute)
            ]
            attrs.extend(discovered)
        return attrs

    @property
    def array_attributes(cls):
        """
        Gets a list of all ArrayAttribute instances defined on subclasses of cls.Component

        Returns
        -------
        attrs : list[ArrayAttribute]
        """
        attrs = []
        for comp_type in cls.Component.__subclasses__():
            discovered = [
                v
                for k, v in vars(comp_type).items()
                if isinstance(v, ArrayAttribute) and not isinstance(v, PublicAttribute)
            ]
            attrs.extend(discovered)
        return attrs


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

    MAX_ENTITIES: int = 1024
    Component: Type[BaseComponent]

    def __init__(self, conn):
        """
        Parameters
        ----------
        conn : Connection
            The Connection object from a multiprocessing.Pipe() used for communication.
        """
        self._message_bus = MessageBus()
        self._message_bus.register_marked(self)
        self._conn = conn
        self._running = True
        self._event_queue = []

    @classmethod
    def _run(cls, conn, max_entities=1024):
        """Internal target to run system."""
        cls.MAX_ENTITIES = max_entities
        for attr in cls.array_attributes:
            attr.reallocate()
        inst = cls(conn)
        inst._main()

    @classmethod
    def setup_shared_state(cls):
        """
        Setup that should happen before System.__init__.
        Should only be called once, so the main process should call this.
        """
        for attr in cls.public_attributes:
            attr.allocate_shm()

    @classmethod
    def teardown_shared_state(cls):
        """Teardown state setup in System.setup_shared_state."""
        for attr in cls.public_attributes:
            attr.close_shm()

    @classmethod
    def run_in_process(cls, max_entities):
        """
        Method that should be called externally to actually start the system.

        Parameters
        ----------
        max_entities : int
            Lets the system know how much memory to allocate for numpy arrays.
        """
        cls.setup_shared_state()
        local, foreign = mp.Pipe()
        process = mp.Process(target=cls._run, args=(foreign, max_entities))
        process.start()
        return local, process

    def update(self):
        """
        Stub for subclass defined behavior.

        The Update event will first invoke this method, then post all events
        pooled in the event queue, and finally send a SystemUpdateComplete back
        to the main process.
        """

    def post_event(self, event, key=None):
        """Sends an event back to the main process."""
        self._conn.send((event, key))

    def _main(self):
        while self._running:
            self._poll()
        self.teardown_shared_state()

    def _poll(self):
        if not self._conn.poll(0):
            return
        message = self._conn.recv()
        try:
            (event, key) = message
            if isinstance(event, Update) or isinstance(event, SystemStop):
                self._message_bus.post_event(event, key=key)
            else:
                self._event_queue.append((event, key))
        except Exception as e:
            self._conn.send(e)

    @eventhandler(Update)
    def _update(self, event):
        self.update()
        for (event, key) in self._event_queue:
            self._message_bus.post_event(event, key=key)
        self._event_queue = []
        self.post_event(SystemUpdateComplete(type(self)))

    @eventhandler(SystemStop)
    def _stop(self, event):
        self._running = False


class SystemUpdateComplete(Event):
    """
    Posted to the main process after a system finishes handling an Update event.
    """

    __slots__ = ["system_type"]

    system_type: Type[System]


class ArrayAttribute:
    _owner: BaseComponent
    _name: str

    def __init__(self, dtype, length=1):
        """
        A Descriptor that manages a numpy array. This should be used on a System.Component.

        Parameters
        ----------
        dtype : np.dtype
        length : int
            The underlying array will be 1 dimensional with this length
        """
        self._array = np.zeros((length,), dtype)
        self.dtype = dtype

    def reallocate(self):
        """Reallocates the underlying array. Does not preserve data."""
        self._array = np.zeros((self.length,), self.dtype)

    def __set_name__(self, owner, name):
        if not issubclass(owner, BaseComponent):
            raise TypeError(
                f"{self.__class__.__name__} is meant to be used on BaseComponent subclasses."
            )
        self._owner = owner
        self._name = name
        self.reallocate()

    def __get__(self, instance, owner):
        """
        Returns
        -------
        value : numpy.ndarray | int | float | str
            If invoked on an instance returns an entry from the array using entity_id as index.
            Otherwise if invoked from the Type object it returns the entire array.
        """
        if instance is None:
            return self._array
        index = instance.entity_id
        return self._array[index]

    def __set__(self, obj, value):
        index = obj.entity_id
        self._array[index] = value

    @property
    def length(self):
        return self._owner.SYSTEM.MAX_ENTITIES


class PublicAttribute:
    def __init__(self, dtype):
        """
        A Descriptor for a shared public attribute.

        Normally an array attribute is localized to its own process,
        data stored in one of these attributes can be accessed from any
        process.

        THERE ARE NO LOCKING MECHANISMS IN PLACE

        The array will be double buffered, with writes going to
        one array and reads to the other. Once all systems have
        signaled that they are done Updating the main process
        will copy the write buffer into the read buffer.

        Parameters
        ----------
        dtype : np.dtype
        """
        self.dtype = dtype
        self._array: DoubleBufferedArray = None

    def __set_name__(self, owner, name):
        if not issubclass(owner, BaseComponent):
            raise TypeError(
                f"{self.__class__.__name__} is meant to be used on BaseComponent subclasses."
            )
        self._owner = owner
        self._name = name

    def __get__(self, instance, owner):
        if self._array is None:
            self._connect_shm()
        if instance is None:
            return self._array
        index = instance.entity_id
        return self._array[index]

    def __set__(self, obj, value):
        index = obj.entity_id
        self._array[index] = value

    def allocate_shm(self):
        """
        Allocates the shm file. Attempted access before allocation will
        raise FileNotFoundError.

        This only needs to be called once across all processes. As such, the
        main process should probably be responsible for calling this.
        """
        self._array = DoubleBufferedArray.create(
            self._shm_id, (self.length,), self.dtype
        )

    def close_shm(self):
        """
        Totally unlinks the shm file.

        TODO: Current implementation works for both windows and POSIX, but it would
            probably be better to just handle the cases separately in the future.
        """
        if self._array is not None:
            self._array.unlink()
            self._array = None

    def update(self):
        """
        Copies the write buffer into the read buffer.

        Note that there are no synchronization primitives guarding this process.
        """
        if self._array is None:
            self._connect_shm()
        self._array.flip()

    @property
    def length(self):
        return self._owner.SYSTEM.MAX_ENTITIES

    @property
    def _shm_id(self):
        return f"{self._owner.__name__}__{self._name}"

    def __repr__(self):
        return f"<PublicAttribute({self._shm_id})>"

    def _connect_shm(self):
        self._array = DoubleBufferedArray(self._shm_id, (self.length,), self.dtype)
