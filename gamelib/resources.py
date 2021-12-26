# TODO: A more general approach to discovering resources such that different
#  types of resources can be added with a few lines of code.
# TODO: Implement some BaseResource type of class so it can be made quite
#  uniform in what to expect this module to return when asking it to locate
#  a resource.

import pathlib
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from itertools import groupby
from typing import List, DefaultDict, Dict


@dataclass
class ResourceDirectories:
    shaders: List[pathlib.Path] = field(default_factory=list)
    assets: List[pathlib.Path] = field(default_factory=list)

    def clear(self):
        self.shaders.clear()
        self.assets.clear()


_RESOURCE_ROOTS = (pathlib.Path.cwd(),)
_SHADER_EXTS = (".vert", ".frag", ".tesc", ".tese", ".geom")
_ASSET_EXTS = (".jpg", ".png")

_discovered_resource_directories = ResourceDirectories()
_shader_srcs: DefaultDict[str, list] = defaultdict(list)
_asset_paths: Dict[str, pathlib.Path] = {}


def clear_cache():
    """Clear all cached resources that have been discovered."""

    _discovered_resource_directories.clear()
    _shader_srcs.clear()
    _asset_paths.clear()


def set_resource_roots(*paths):
    """Sets the root path that will be searched to find resources. When this
    method is called the cache will be cleared.

    Parameters
    ----------
    *paths : pathlib.Path
    """

    global _RESOURCE_ROOTS
    _RESOURCE_ROOTS = paths
    clear_cache()
    discover_directories()


def add_resource_roots(*paths):
    """Similar to `set_resource_roots` but doesn't overwrite what's currently
    in use, and doesn't clear the cache.

    Parameters
    ----------
    *paths : pathlib.Path
    """

    global _RESOURCE_ROOTS
    for path in paths:
        discover_directories(path)
    _RESOURCE_ROOTS = (*_RESOURCE_ROOTS, *paths)


def discover_directories(*paths):
    """Crawls through either given paths or paths that have been added to the
    resource roots for directories named "assets" or "shaders" and caches them.

    Parameters
    ----------
    *paths : pathlib.Path, optional

    Returns
    -------
    ResourceDirectories:
        Returns the resource directories cache.
    """

    paths = paths or _RESOURCE_ROOTS

    def _walk_dir(dir_path):
        if not dir_path.is_dir():
            return
        if dir_path.name == "shaders":
            _discovered_resource_directories.shaders.append(dir_path)
        elif dir_path.name == "assets":
            _discovered_resource_directories.assets.append(dir_path)
        else:
            for p in dir_path.iterdir():
                _walk_dir(p)

    for path in paths:
        _walk_dir(path)

    return _discovered_resource_directories


def discover_shader_sources(*paths):
    """Looks through previously discovered shader directories to find files
    with glsl shader extensions:
        .vert
        .frag
        .tese
        .tesc
        .geom
    If paths are given they will be included in the search.

    Parameters
    ----------
    paths : pathlib.Path, optional

    Returns
    -------
    dict[str, list]
        The keys are shader names and values are paths to the files.

        Given dir:
            shaders/
                water.vert
                water.frag
        key is "water"
        value is a list containing paths to those two files.
    """

    if paths:
        discover_directories(*paths)

    if not _discovered_resource_directories.shaders:
        discover_directories()

    def _walk_dir(dir_path):
        src = []
        for path in dir_path.iterdir():
            if any((path.name.endswith(ext) for ext in _SHADER_EXTS)):
                src.append(path)
            elif path.is_dir():
                _walk_dir(path)
        for name, group in groupby(src, lambda p: p.name.split(".")[0]):
            _shader_srcs[name].extend(group)

    for d in _discovered_resource_directories.shaders:
        _walk_dir(d)

    return _shader_srcs.copy()


def discover_asset_files(*paths):
    """Looks for files in previously discovered assets
    directories with the extensions:
        .jpg
        .png
    If given, paths will be included in the search.

    Parameters
    ----------
    paths : pathlib.Path, optional

    Returns
    -------
    dict[str, pathlib.Path]
        Keys are the filename. water.png = "water"
        values are pathlib.Paths to the file
    """

    if paths:
        discover_directories(*paths)

    if not _discovered_resource_directories.assets:
        discover_directories()

    def _walk_dir(dir_path):
        for path in dir_path.iterdir():
            if any((path.name.endswith(ext) for ext in _ASSET_EXTS)):
                _asset_paths[path.name.split(".")[0]] = path
            elif path.is_dir():
                _walk_dir(path)

    for d in _discovered_resource_directories.assets:
        _walk_dir(d)

    return _asset_paths.copy()


def find_shader(name):
    """Tries to get a list of paths to shader source files with
    the given name.

    Parameters
    ----------
    name : str
        for shader made up of files water.vert, water.frag
        name would be "water"

    Returns
    -------
    list[pathlib.Path]

    Raises
    ------
    KeyError:
        When the files could not be found for given name.
    """
    if name not in _shader_srcs:
        discover_shader_sources()
        if name not in _shader_srcs:
            raise KeyError(
                f"Shader with {name=} not found in "
                f"{list(_shader_srcs.keys())}."
            )
    return _shader_srcs[name]


def find_asset(name):
    """Tries to get the path to an asset resource file with the given name.

    Parameters
    ----------
    name : str
        water.jpg name would be "water"

    Returns
    -------
    pathlib.Path

    Raises
    ------
    KeyError:
        When the file could not be found.
    """

    try:
        return _asset_paths[name]
    except KeyError:
        discover_asset_files()
        return _asset_paths[name]
