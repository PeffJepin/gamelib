from __future__ import annotations

import enum
import os

from src.gamelib import events

_POSIX = os.name == 'posix'
_globals: _EcsGlobals | None = None


def reset_globals(statics=None):
    """ Get a fresh global object instance.

    Can be initialized with some static values.

    Parameters
    ----------
    statics : dict
        internal statics can be passed in with the reset call.
    """
    global _globals
    _globals = _EcsGlobals()

    if not statics:
        return
    for k, value in statics.items():
        add_static_global(k, value)


def export_globals():
    """ Expose the globals object. This should be used to get a reference to it
    to send it to a child process.

    Does nothing on posix, windows has to do this because of process spawning.
    Posix systems will inherit globals from fork.

    Returns
    -------
    globals : _EcsGlobals
    """
    return None if _POSIX else _globals


def import_globals(instance):
    """ Import an instance of the globals object from the main process.

    Does nothing on posix, windows has to do this because of process spawning.
    Posix systems will inherit globals from fork.

    Parameters
    ----------
    instance : _EcsGlobals
    """
    if _POSIX:
        return
    global _globals
    _globals = instance


def add_static_global(key, value):
    """ Add a static variable to the globals shared between processes.

    Parameters
    ----------
    key : Any
        must be picklable
    value : Any
         must be picklable
    """
    _globals.statics[key] = value


def get_static_global(key):
    """ Retrieve a static variable from the globals shared between processes.

        Parameters
        ----------
        key : Any

        Returns
        -------
        static : Any

        Raises
        ------
        KeyError:
            If you haven't added a static with this key or
            globals weren't sent to a child process.
    """
    return _globals.statics[key]


def register_shared_array(key, array):
    """ Register a shared memory array with the globals object so windows
    processes can find it.

    Parameters
    ----------
    key : Any
    array : mp.Array
    """
    _globals.shm[key] = array


def get_shared_array(key):
    """ Get a reference to a previously registered multiprocessing.Array

    Parameters
    ----------
    key : Any

    Returns
    -------
    array : mp.Array

    Raises
    ------
    KeyError:
        If you haven't registered an array with this key or
        the globals weren't sent to a child process.
    """
    return _globals.shm[key]


class _EcsGlobals:
    def __init__(self, max_entities=1024):
        self.statics = {
            StaticGlobals.MAX_ENTITIES: max_entities
        }
        self.shm = dict()


class StaticGlobals(enum.Enum):
    MAX_ENTITIES = enum.auto()


# init globals
reset_globals()


class EntityDestroyed(events.Event):
    __slots__ = ["id"]

    id: int
