import moderngl
import moderngl_window
import numpy as np

from moderngl.error import Error as GLError
from moderngl_window import settings

window: moderngl_window.BaseWindow
context: moderngl.Context

_int = int
_float = float
_bool = bool

int = np.dtype("i4")
uint = np.dtype("u4")
float = np.dtype("f4")
double = np.dtype("f8")
bool = np.dtype("bool")
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
sampler2D = int
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


def coerce_array(array, dtype):
    if isinstance(dtype, str):
        try:
            dtype = eval(dtype)
            assert isinstance(dtype, np.dtype)
        except (AssertionError, NameError):
            dtype = np.dtype(dtype)

    if array.dtype != dtype:
        if dtype.subdtype is not None:
            base_dtype, shape = dtype.subdtype
            array = array.astype(base_dtype).reshape((-1, *shape))
        else:
            array = array.astype(dtype)
    return array


def init_window(**config):
    """Creates the window and stores a reference to it.

    Parameters
    ----------
    **config : Any
        see moderngl-window reference for options.

    Returns
    -------
    window : moderngl_window.BaseWindow

    Notes
    -----
    https://moderngl-window.readthedocs.io/en/latest/guide/window_guide.html
    """
    global window
    global context

    if "class" not in config:
        settings.WINDOW["class"] = "moderngl_window.context.pygame2.Window"
    for k, v in config.items():
        settings.WINDOW[k] = v
    window = moderngl_window.create_window_from_settings()
    context = window.ctx
    return window


def init_standalone():
    """Initialize and store global reference to a headless OpenGL context.

    Returns
    -------
    ctx : moderngl.Context
    """
    global context
    context = moderngl.create_standalone_context()
    return context
