from __future__ import annotations

import abc
from functools import partial
from typing import Type

from numpy import ma


from . import events, Config
from .sharedmem import DoubleBufferedArray


class ArrayAttribute:
    _owner: Type[BaseComponent]
    _name: str
    _array: ma.MaskedArray

    def __init__(self, dtype):
        """
        A Descriptor that manages a numpy array. This should be used on a System.Component.

        Parameters
        ----------
        dtype : type[int] | type[float] | np.dtype
        """
        self._dtype = dtype

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
        value : ma.MaskedArray | int | float | str
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

    def reallocate(self):
        """Reallocates the underlying array. Does not preserve data."""
        self._array = ma.zeros((self.length,), self._dtype)
        self._array[:] = ma.masked

    @property
    def length(self):
        return Config.MAX_ENTITIES


class PublicAttribute:
    _dbl_buffer: DoubleBufferedArray | None = None

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
        dtype : type[int] | type[float] | np.dtype
        """
        self.is_open = False
        self._dtype = dtype

    def __set_name__(self, owner, name):
        if not issubclass(owner, BaseComponent) and owner != BaseComponent:
            raise TypeError(
                f"{self.__class__.__name__} is meant to be used on BaseComponent subclasses."
            )
        self._owner = owner
        self._name = name

    def __get__(self, instance, owner):
        """
        Get reference to the DoubleBufferedArray. Will attempt to connect the array to shm
        if it is not already connected.

        Returns
        -------
        value : Numeric | DoubleBufferedArray
            A value from the array at index = entity_id if accessed from an instance.
            The whole array if accessed from the component type.

        Raises
        ------
        FileNotFoundError:
            If memory has not been allocated by the time of access.
        """
        if not self.is_open:
            self._open()
        if instance is None:
            return self._dbl_buffer
        index = instance.entity_id
        return self._dbl_buffer[index]

    def __set__(self, obj, value):
        """
        Set an entry in this array to value indexed on obj.entity_id

        Raises
        ------
        FileNotFoundError:
            If shm has not been allocated by time of use.
        """
        if not self.is_open:
            self._open()
        index = obj.entity_id
        self._dbl_buffer[index] = value

    def __repr__(self):
        return f"<PublicAttribute({self._owner.__name__}.{self._name}, open={self.is_open})>"

    def update_buffer(self):
        """
        Copies the write buffer into the read buffer.

        Note that there are no synchronization primitives guarding this process.
        """
        self._dbl_buffer.flip()

    def close_view(self):
        self.is_open = False
        self._dbl_buffer = None

    @property
    def shared_specs(self):
        """
        Get the underlying shared memory specification.

        Returns
        -------
        specs : tuple[ArraySpec]
        """
        if not self.is_open:
            return DoubleBufferedArray(
                self._shm_name, self._dtype, Config.MAX_ENTITIES
            ).specs
        return self._dbl_buffer.specs

    @property
    def _shm_name(self):
        return f"{self._owner.__class__.__name__}__{self._name}"

    def _open(self):
        self._dbl_buffer = DoubleBufferedArray(
            self._shm_name, self._dtype, Config.MAX_ENTITIES
        )
        self._dbl_buffer.connect()
        self.is_open = True


class ComponentCreated(events.Event):
    __slots__ = ["entity_id", "type", "args"]

    entity_id: int
    type: Type[BaseComponent]
    args: tuple


class BaseComponent(abc.ABC):
    _array_attributes: dict
    _public_attributes: dict

    def __init__(self, entity_id):
        self.entity_id = entity_id

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        cls._array_attributes = {
            k: v for k, v in vars(cls).items() if isinstance(v, ArrayAttribute)
        }
        cls._public_attributes = {
            k: v for k, v in vars(cls).items() if isinstance(v, PublicAttribute)
        }

    def destroy(self):
        for name in self._array_attributes.keys():
            setattr(self, name, ma.masked)

    @classmethod
    def get_array_attributes(cls):
        return cls._array_attributes.values()

    @classmethod
    def get_public_attributes(cls):
        return cls._public_attributes.values()
