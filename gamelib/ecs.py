# TODO: Some way to get a proxy to a component attribute such that another
#   module could keep reference to the proxy and call it on demand to get
#   to the internal data. Since the internal array is subject to reallocation,
#   simply getting a reference to the underlying array is insufficient.

# TODO: An option to allocate the internal arrays using the structured dtype
#   instead of individual arrays. Using a decorator approach like dataclass
#   uses might offer a more suitable API for optional features like this.

# TODO: A module docstrings with examples after this module is fleshed out a
#   bit more.

# TODO: Eventually I will want the option to allocate the arrays using
#   shared memory. Since they are regularly reallocated this will probably
#   require a new shared memory module so other processes can easily find the
#   internal array after it has been reallocated.

import itertools
import threading

from typing import Iterable

import numpy as np


_STARTING_LENGTH = 10


class DynamicArrayManager:
    def __init__(self, **dtypes):
        self._internal_length = _STARTING_LENGTH
        self._arrays = {
            name: np.empty(self._internal_length, dtype)
            for name, dtype in dtypes.items()
        }
        self._ids = np.empty(self._internal_length, int)
        self._index_lookup = np.empty(self._internal_length, int)
        self._ids[:] = -1
        self._index_lookup[:] = -1
        self._length = 0
        self._counter = itertools.count(0)
        self._recycled_ids = []

    def __getattr__(self, attr):
        if attr in self._arrays:
            return self._arrays[attr][: self._length]
        if attr == "ids":
            return self._ids[: self._length]

    def __getitem__(self, id):
        try:
            index = self._index_lookup[id]
        except IndexError:
            return None

        if index == -1:
            return None
        return _DynamicArrayEntry(id, self)

    def __delitem__(self, id):
        index = self._index_lookup[id]
        if index == -1:
            return

        final_index = self._length - 1
        if index != final_index:
            final_id = self._ids[final_index]
            # swap good data (end of the array) into the deleted index
            for array in self._arrays.values():
                array[index] = array[final_index]
            # correct the id and index lookup arrays to reflect the swap
            self._ids[index] = final_id
            self._index_lookup[final_id] = index

        self._recycled_ids.append(id)
        self._length -= 1
        # mask off the the final entry where there is now stale data
        self._index_lookup[id] = -1
        self._consider_shrinking()

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        return (getattr(self, field) for field in self._arrays)

    @property
    def masked_length(self):
        return self._length

    def new_entry(self, *args, **kwargs):
        if self._recycled_ids:
            id = self._recycled_ids.pop(0)
        else:
            id = next(self._counter)

        if self._length >= self._internal_length:
            self._reallocate_arrays(self._length * 1.5)
        if id >= len(self._index_lookup):
            new_lookup = _reallocate_array(self._index_lookup, id + 1, fill=-1)
            self._index_lookup = new_lookup

        index = self._length
        self._length += 1
        self._ids[index] = id
        self._index_lookup[id] = index
        if args:
            for arr, val in zip(self._arrays.values(), args):
                arr[index] = val
        elif kwargs:
            for name, val in kwargs.items():
                self._arrays[name][index] = val

        return _DynamicArrayEntry(id, self)

    def get_index(self, id):
        try:
            index = self._index_lookup[id]
            if index == -1:
                return None
            return index
        except IndexError:
            return None

    def get_raw_arrays(self):
        return self._arrays

    def get_indices(self, ids):
        return self._index_lookup[ids]

    def clear(self):
        self._reallocate_arrays(_STARTING_LENGTH)
        self._index_lookup = np.empty(_STARTING_LENGTH, int)
        self._index_lookup[:] = -1
        self._internal_length = _STARTING_LENGTH
        self._length = 0
        self._counter = itertools.count(0)
        self._recycled_ids = []

    def _reallocate_arrays(self, new_length):
        new_length = max(int(new_length), _STARTING_LENGTH)
        for name, array in self._arrays.items():
            self._arrays[name] = _reallocate_array(array, new_length, fill=-1)
        self._ids = _reallocate_array(self._ids, new_length, fill=-1)
        self._internal_length = new_length

    def _consider_shrinking(self):
        # consider shrinking id lookup
        if len(self._index_lookup) >= self._length * 1.8 and self._length > 0:
            largest_id = np.argwhere(self._index_lookup != -1)[-1]
            if largest_id <= self._length * 1.4:
                self._index_lookup = _reallocate_array(
                    self._index_lookup, largest_id + 1, -1
                )
                self._counter = itertools.count(largest_id + 1)
                self._recycled_ids = list(
                    filter(lambda id: id < largest_id, self._recycled_ids)
                )

        # consider shrinking component data arrays
        if self._length <= self._internal_length * 0.65:
            self._reallocate_arrays(self._length)


class _DynamicArrayEntry:
    _initialized = False

    def __init__(self, id, manager):
        self._manager = manager
        self._id = id
        self._initialized = True

    @property
    def id(self):
        return self._id

    def __getattr__(self, attr):
        arrays = self._manager.get_raw_arrays()
        index = self._manager.get_index(self._id)
        if index is None:
            return None
        if attr not in arrays:
            raise AttributeError(f"{attr} not in {arrays.keys()!r}")
        return arrays[attr][index]

    def __setattr__(self, attr, value):
        if not self._initialized:
            super().__setattr__(attr, value)
        else:
            arrays = self._manager.get_raw_arrays()
            index = self._manager.get_index(self._id)
            if index is None:
                return
            if attr not in arrays:
                raise AttributeError(f"{attr} not in {arrays.keys()!r}")
            arrays[attr][index] = value

    def __iter__(self):
        index = self._manager.get_index(self._id)
        if index is None:
            return ()
        return iter(a[index] for a in self._manager.get_raw_arrays().values())

    def __eq__(self, other):
        if not isinstance(other, Iterable):
            return False
        return all(v1 == v2 for v1, v2 in zip(self, other))

    def __repr__(self):
        names = self._manager.get_raw_arrays().keys()
        strdesc = ", ".join(f"{name}={val}" for name, val in zip(names, self))
        return f"<ArrayEntry({strdesc})>"


class _ComponentType(type):
    """Metaclass for Component.

    Responsible for managing access to the underlying ndarrays when
    accessed through the Type object.
    """

    def __getattr__(cls, name):
        """Get the underlying array if name is an annotated field."""

        if name in cls._fields:
            return getattr(cls._arrays, name)
        raise AttributeError(f"{cls!r} has no attribute {name!r}")

    def __setattr__(cls, name, value):
        """Don't allow values to be bound to annotated attribute names on
        the Type object."""

        if cls._initialized and name in cls._fields:
            # don't allow setting annotated fields as class attributes.
            return
        super().__setattr__(name, value)

    @property
    def length(cls):
        return cls._arrays.masked_length

    def __enter__(cls):
        """Acquire a lock on the internal arrays."""

        cls._lock.acquire()

    def __exit__(cls, *args, **kwargs):
        """Release the internal lock."""

        cls._lock.release()


class Component(metaclass=_ComponentType):
    """Component is a base class used for laying out data in contiguous
    memory. Attributes should be annotated like a dataclass and appropriate
    internal arrays will be managed with the annotated dtype."""

    _initialized = False

    def __init_subclass__(cls, **kwargs):
        """Initialize the subclass based on what has been annotated."""

        cls._initialized = False
        cls._lock = threading.RLock()
        cls._fields = cls.__dict__.get("__annotations__", {})
        if not cls._fields:
            raise AttributeError("No attributes have been annotated.")
        cls._arrays = DynamicArrayManager(**cls._fields)
        cls._structured_dtype = np.dtype([
            (name, dtype) if isinstance(dtype, np.dtype) 
            else (name, np.dtype(dtype))
            for name, dtype in cls._fields.items()
        ])
        cls.itemsize = cls._structured_dtype.itemsize
        cls._initialized = True

    def __new__(cls, *args, _id=None, **kwargs):
        """Create some new Component data. If id is given this will load
        access into existing data instead.

        Parameters
        ----------
        *args : Any
            __init__ args
        id : int, optional
            If given, an existing component should be loaded, otherwise
            a new id should be generated and a new component will be created.
        **kwargs : Any
            __init__ kwargs

        Returns
        -------
        Component | None:
            If id is given and isn't found to be an existing component, None
            will be returned instead of a Component instance.
        """

        if _id is not None:
            # check if this id exists
            if cls._arrays[_id] is None:
                return None
            entry = cls._arrays[_id]
        else:
            entry = cls._arrays.new_entry()

        instance = super().__new__(cls)
        instance.id = entry.id
        instance._array_entry = entry
        return instance

    def __init__(self, *args, **kwargs):
        """Set the initial values of a component. *args or **kwargs are
        mutually exclusive.

        Parameters
        ----------
        *args : Any
            Args will map to annotated attributes in the order they are given.
        **kwargs : Any
            Keys will map to annotated attribute names.
        """

        if args:
            for name, arg in zip(self._fields, args):
                setattr(self._array_entry, name, arg)
        else:
            for name, value in kwargs.items():
                if name in self._fields:
                    setattr(self._array_entry, name, value)

    def __repr__(self):
        values = ", ".join(
            f"{name}={getattr(self, name)}" for name in self._fields
        )
        return f"<{self.__class__.__name__}(id={self.id}, {values})>"

    def __eq__(self, other):
        """A component compares for equality based on the values of it's
        annotated attributes."""

        if type(self) == type(other):
            return self.values == other.values
        else:
            return self.values == other

    def __setattr__(self, name, value):
        """Setting an annotated attribute should set the value inside the
        appropriate internal array, rather than binding the given value to
        the instance as would normally happen."""

        if name in self._fields:
            setattr(self._array_entry, name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        """Getting an annotated attribute from the context of a Component
        instance should index into the internal array to retrieve the
        appropriate value."""

        if name in self._fields:
            return getattr(self._array_entry, name)
        else:
            raise AttributeError(f"{self!r} has no attribute {name!r}")

    def __enter__(self):
        """Use the instance as a context manager to lock the internal array."""

        self._lock.acquire()

    def __exit__(self, *args, **kwargs):
        """Release the lock."""

        self._lock.release()

    @property
    def values(self):
        """Get the values for this instance's annotated attributes."""

        return tuple(getattr(self, name) for name in self._fields)

    @classmethod
    def get(cls, id):
        """Gets an existing instance of this Component given an id.

        Parameters
        ----------
        id : int

        Returns
        -------
        Component | None:
            Depending on if a component with this id is accounted for.
        """

        return cls(_id=id)

    @classmethod
    def get_raw_arrays(cls):
        """Gets the raw internal arrays (unmasked).

        Returns
        -------
        np.ndarray:
            The resulting array will have a structured dtype which is an
            aggregate of all the annotated attributes of this component.
        """

        combined = np.empty(len(cls._arrays), cls._structured_dtype)
        for name in cls._fields:
            combined[name] = getattr(cls._arrays, name)
        return combined

    @classmethod
    def destroy(cls, target):
        """Destroys a component given either an instance of `cls` or an
        integer id.

        Parameters
        ----------
        target : Component | int
            This can either be an instance of this type of component or the
            id of the component to be deleted.
        """

        id = target.id if isinstance(target, cls) else target
        del cls._arrays[id]

    @classmethod
    def clear(cls):
        """Resets the component to initial state."""

        for id in cls._arrays.ids:
            cls.destroy(id)
        cls._arrays.clear()

    @classmethod
    def indices_from_ids(cls, ids):
        return cls._arrays.get_indices(ids)


class _EntityMeta(type):
    def __getattr__(cls, name):
        return _EntityMask(cls, cls._fields[name])

    def get_component_ids(cls, component_type):
        attribute_name = ""
        for name, type in cls._fields.items():
            if type is component_type:
                attribute_name = name
                break
        if not attribute_name:
            raise ValueError(f"Couldn't find a field for {component_type!r}.")
        return getattr(cls._arrays, attribute_name)


class _EntityMask:
    def __init__(self, entity_type, component_type):
        self._entity = entity_type
        self._component = component_type

    def __getattr__(self, name):
        ids = self._entity.get_component_ids(self._component)
        indices = self._component.indices_from_ids(ids)
        return getattr(self._component, name)[indices]


class Entity(metaclass=_EntityMeta):
    def __init_subclass__(cls, **kwargs):
        cls._fields = cls.__dict__.get("__annotations__", {})
        if not cls._fields:
            raise AttributeError("No attributes have been annotated.")
        id_fields = dict()
        for name, dtype in cls._fields.items():
            if not issubclass(dtype, Component):
                raise AttributeError(
                    "Expected {dtype!r} to be a Component subclass."
                )
            id_fields[name] = int
        cls._arrays = DynamicArrayManager(**id_fields)

    def __new__(cls, *args, _id=None, **kwargs):
        if _id is None:
            if args:
                component_ids = tuple(arg.id for arg in args)
                array_entry = cls._arrays.new_entry(*component_ids)
            elif kwargs:
                component_ids = {
                    name: comp.id for name, comp in kwargs.items()
                }
                array_entry = cls._arrays.new_entry(**component_ids)
            else:
                raise ValueError("Either *args or **kwargs must be given.")
        else:
            array_entry = cls._arrays[_id]

        if array_entry is None:
            return None

        instance = super().__new__(cls)
        instance._array_entry = array_entry
        instance._id = array_entry.id
        return instance

    def __getattr__(self, name):
        if self._arrays[self._array_entry.id] is None:
            return None
        if name in self._fields:
            comp_type = self._fields[name]
            return comp_type.get(getattr(self._array_entry, name))

        raise AttributeError()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return all(comp1 == comp2 for comp1, comp2 in zip(self, other))

    def __iter__(self):
        return iter(getattr(self, name) for name in self._fields)

    @property
    def id(self):
        return self._id

    @classmethod
    def get(cls, id):
        return cls(_id=id)

    @classmethod
    def destroy(cls, target):
        if isinstance(target, cls):
            id = target.id
            instance = target
        else:
            id = target
            instance = cls.get(id)
        for comp in instance:
            comp.destroy(comp.id)
        del cls._arrays[id]

    @classmethod
    def clear(cls):
        for id in cls._arrays.ids:
            cls.destroy(id)
        cls._arrays.clear()


def _reallocate_array(array, new_length, fill=0):
    """Allocates a new array with new_length and copies old data back into
    the array. Empty space created will be filled with fill value."""

    new_length = max(int(new_length), _STARTING_LENGTH)
    new_array = np.empty(new_length, array.dtype)
    new_array[:] = fill
    if len(array) <= new_length:
        new_array[: len(array)] = array
    else:
        new_array[:] = array[:new_length]
    return new_array
