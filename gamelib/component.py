from __future__ import annotations

import itertools
import threading
import numpy as np


_STARTING_LENGTH = 10


class _ComponentType(type):
    def __getattr__(cls, name):
        if name in cls._fields:
            return getattr(cls, "_" + name)[: cls.length]
        raise AttributeError(f"{cls!r} has no attribute {name!r}")

    def __setattr__(cls, name, value):
        if cls._initialized and name in cls._fields:
            # don't allow setting annotated fields as class attributes.
            return
        super().__setattr__(name, value)

    def __enter__(cls):
        cls._lock.acquire()

    def __exit__(cls, *args, **kwargs):
        cls._lock.release()


class Component(metaclass=_ComponentType):
    # once initialized the public attributes wont allow you to bind values
    # to them. this allows syntax like Component.x += 10 to increment the
    # internal component array without binding the result value to Component.x.
    _initialized = False

    def __init_subclass__(cls, **kwargs):
        # __init_subclass__ can also be used to reset a component
        # so it needs to be reset to False here.
        cls._initialized = False
        cls._lock = threading.RLock()

        # find annotated fields
        cls._fields = cls.__dict__.get("__annotations__", {})
        if not cls._fields:
            raise AttributeError("No attributes have been annotated.")

        # keep an array of ids alongside the annotated fields
        cls.ids = np.empty(_STARTING_LENGTH, int)
        cls.ids[:] = -1

        # keep a record of where to find data for a particular component id
        cls.id_lookup = np.zeros(_STARTING_LENGTH, int)
        cls.id_lookup[:] = -1

        # maintain a standard way of assigning ids
        cls._max_length = _STARTING_LENGTH
        cls._counter = itertools.count(0)
        cls._recycled_ids = []
        cls.length = 0

        # create the underlying component arrays and form an aggregate
        # structured dtype allowing the individual arrays created from
        # attribute annotation to be viewed as a single structured array
        structure = []
        for name, dtype in cls._fields.items():
            if not isinstance(dtype, np.dtype):
                dtype = np.dtype(dtype)
            array = np.zeros(_STARTING_LENGTH, dtype)
            setattr(cls, "_" + name, array)
            structure.append((name, dtype))
        cls._structured_dtype = np.dtype(structure)

        # number of bytes a single component instance requires
        cls.itemsize = cls._structured_dtype.itemsize

        # effectively freezes the annotated attributes, disallowing
        # the public-facing attributes to be set with new values.
        cls._initialized = True

    def __new__(cls, *args, id=None, **kwargs):
        # if id is given don't create a new component.. check if the requested
        # component exists already and set up the instance to display that
        # component data, if no component exists with this id return None.
        if id is not None:
            # check if this id exists
            if cls.id_lookup[id] == -1:
                return None

        # else if no id is given then a new component is being created..
        # in this case retrieve an id for the new component and set all
        # appropriate record keeping values for the new component.
        else:
            id = cls._get_new_id()
            if id >= cls._max_length:
                cls.reallocate(cls._max_length * 1.5)
            cls.id_lookup[id] = cls.length
            cls.ids[cls.length] = id
            cls.length += 1

        # setup the instance with the correct id before returning and allowing
        # __init__ to proceed. Since setting an instance's annotated attribute
        # value sets a value into an internal array, the instance must have
        # a valid id before __init__ is called.
        instance = super().__new__(cls)
        instance.id = id
        return instance

    def __init__(self, *args, **kwargs):
        # args or kwargs but not both can be used to set field values
        if args:
            for name, arg in zip(self._fields, args):
                setattr(self, name, arg)
        else:
            for name, value in kwargs.items():
                setattr(self, name, value)

    def __repr__(self):
        values = ", ".join(
            f"{name}={getattr(self, name)}" for name in self._fields
        )
        return f"<{self.__class__.__name__}({values})>"

    def __eq__(self, other):
        if type(self) == type(other):
            return self.values == other.values
        else:
            return self.values == other

    def __setattr__(self, name, value):
        # instances set a value in the larger array
        if name in self._fields:
            index = self.id_lookup[self.id]
            getattr(self, "_" + name)[index] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        # instances get their value from the larger array
        if name in self._fields:
            index = self.id_lookup[self.id]
            value = getattr(self, "_" + name)[index] if index >= 0 else None
            return value
        else:
            raise AttributeError(f"{self!r} has no attribute {name!r}")

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, *args, **kwargs):
        self._lock.release()

    @property
    def values(self):
        return tuple(getattr(self, name) for name in self._fields)

    @classmethod
    def get(cls, id):
        return cls(id=id)

    @classmethod
    def get_raw_arrays(cls):
        print(cls._max_length)
        combined = np.empty(cls._max_length, cls._structured_dtype)
        for name in cls._fields:
            combined[name] = getattr(cls, "_" + name)
        return combined

    @classmethod
    def destroy(cls, target):
        # target can be a component instance or an integer id
        id = target if isinstance(target, int) else target.id
        index = cls.id_lookup[id]

        # component doesn't actually exist
        if index == -1:
            return

        # swap the component to be deleted to the end of the array
        # effectively removing it after the length is decremented and
        # it's id is recycled.
        last_index = cls.length - 1
        if index != last_index:
            id_of_last_component = cls.ids[last_index]
            cls.id_lookup[id_of_last_component] = index
            arrays = cls.get_raw_arrays()
            for name in cls._fields:
                arrays[name][index] = arrays[name][last_index]

        # clean-up the old id and officially decrement the length
        cls.length -= 1
        cls._recycled_ids.append(id)
        cls.id_lookup[id] = -1

        # check if the arrays should shrink
        cls._consider_shrinking()

    @classmethod
    def _consider_shrinking(cls):
        # consider shrinking id lookup
        if len(cls.id_lookup) >= cls.length * 1.8 and cls.length > 0:
            greatest_id_in_use = np.argwhere(cls.id_lookup != -1)[-1]
            if greatest_id_in_use <= cls.length * 1.4:
                cls.id_lookup = _reallocate_array(
                    cls.id_lookup, greatest_id_in_use + 1, -1
                )
                cls._counter = itertools.count(greatest_id_in_use + 1)
                cls._recycled_ids = list(
                    filter(
                        lambda id: id < greatest_id_in_use, cls._recycled_ids
                    )
                )

        # consider shrinking component data arrays
        if cls.length <= cls._max_length * 0.65:
            cls.reallocate(cls.length)

    @classmethod
    def clear(cls):
        cls.__init_subclass__()

    @classmethod
    def _get_new_id(cls):
        if cls._recycled_ids:
            id = cls._recycled_ids.pop(0)
        else:
            id = next(cls._counter)
        if id >= len(cls.id_lookup):
            cls.id_lookup = _reallocate_array(cls.id_lookup, id * 1.5, fill=-1)
        return id

    @classmethod
    def reallocate(cls, new_length):
        new_length = max(int(new_length), _STARTING_LENGTH)
        for name in cls._fields:
            name = "_" + name
            array = _reallocate_array(getattr(cls, name), new_length)
            setattr(cls, name, array)
        cls.ids = _reallocate_array(cls.ids, new_length, fill=-1)
        cls._max_length = new_length


def _reallocate_array(array, new_length, fill=0):
    new_length = max(int(new_length), _STARTING_LENGTH)
    new_array = np.empty(new_length, array.dtype)
    new_array[:] = fill
    if len(array) <= new_length:
        new_array[: len(array)] = array
    else:
        new_array[:] = array[:new_length]
    return new_array
