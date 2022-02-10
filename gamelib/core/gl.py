"""This module is largely just a namespace for OpenGL constants.

The dtype variables map to numpy structured dtypes representing the OpenGL
data model.
"""

import numpy as np

# noinspection PyUnresolvedReferences
from moderngl.error import Error

# noinspection PyUnresolvedReferences
from moderngl import (
    TRIANGLES,
    TRIANGLE_FAN,
    TRIANGLE_STRIP,
    TRIANGLES_ADJACENCY,
    TRIANGLE_STRIP_ADJACENCY,
    POINTS,
    LINES,
    LINE_STRIP,
    LINE_STRIP_ADJACENCY,
    LINE_LOOP,
    LINES_ADJACENCY,
    PATCHES,
)

_int = int
_float = float
_bool = bool

int = np.dtype("i4")
uint = np.dtype("u4")
float = np.dtype("f4")
double = np.dtype("f8")
bool = np.dtype("bool")
byte = np.dtype("byte")
bvec2 = np.dtype((bool, 2))
bvec3 = np.dtype((bool, 3))
bvec4 = np.dtype((bool, 4))
ivec2 = np.dtype((int, 2))
ivec3 = np.dtype((int, 3))
ivec4 = np.dtype((int, 4))
uvec2 = np.dtype((uint, 2))
uvec3 = np.dtype((uint, 3))
uvec4 = np.dtype((uint, 4))
dvec2 = np.dtype((double, 2))
dvec3 = np.dtype((double, 3))
dvec4 = np.dtype((double, 4))
vec2 = np.dtype((float, 2))
vec3 = np.dtype((float, 3))
vec4 = np.dtype((float, 4))
sampler2D = np.dtype("i4")
mat2 = np.dtype((float, (2, 2)))
mat2x3 = np.dtype((float, (3, 2)))
mat2x4 = np.dtype((float, (4, 2)))
mat3x2 = np.dtype((float, (2, 3)))
mat3 = np.dtype((float, (3, 3)))
mat3x4 = np.dtype((float, (4, 3)))
mat4x2 = np.dtype((float, (2, 4)))
mat4x3 = np.dtype((float, (3, 4)))
mat4 = np.dtype((float, (4, 4)))
dmat2 = np.dtype((double, (2, 2)))
dmat2x3 = np.dtype((double, (3, 2)))
dmat2x4 = np.dtype((double, (4, 2)))
dmat3x2 = np.dtype((double, (2, 3)))
dmat3 = np.dtype((double, (3, 3)))
dmat3x4 = np.dtype((double, (4, 3)))
dmat4x2 = np.dtype((double, (2, 4)))
dmat4x3 = np.dtype((double, (3, 4)))
dmat4 = np.dtype((double, (4, 4)))


def coerce_array(array, gl_type, copy=False):
    """Tries to coerce the given array into the dtype and shape of
    the given glsl type.

    Parameters
    ----------
    array : np.ndarray
    gl_type : str | np.dtype

    Returns
    -------
    np.ndarray
    """

    if isinstance(gl_type, str):
        try:
            dtype = eval(gl_type)
            assert isinstance(dtype, np.dtype)
        except (NameError, AssertionError):
            dtype = np.dtype(gl_type)
    else:
        dtype = gl_type
    assert isinstance(dtype, np.dtype)

    if array.dtype != dtype:
        if dtype.subdtype is not None:
            base_dtype, shape = dtype.subdtype
            array = array.astype(base_dtype, copy=copy).reshape((-1, *shape))
        else:
            array = array.astype(dtype, copy=copy)

    return array
