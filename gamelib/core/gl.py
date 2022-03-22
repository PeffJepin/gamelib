"""This module is largely just a namespace for OpenGL constants.

The dtype variables map to numpy structured dtypes representing the OpenGL
data model.
"""

import numpy as np
from moderngl.error import Error
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
from moderngl.program import Program as GLShader

import gamelib
from gamelib import utils

ensure_opengl = utils.Ensure(
    lambda: gamelib.get_context() is not None,
    "An active OpenGL Context is required to call this function, "
    "Perhaps you haven't yet called gamelib.init().",
)


@ensure_opengl
def make_shader_glo(
    vert, tesc=None, tese=None, geom=None, frag=None, varyings=()
):
    return gamelib.get_context().program(
        vertex_shader=vert,
        fragment_shader=frag,
        geometry_shader=geom,
        tess_control_shader=tesc,
        tess_evaluation_shader=tese,
        varyings=varyings,
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

GLSL_DTYPE_STRINGS = {
    "int",
    "uint",
    "float",
    "double",
    "bool",
    "bvec2",
    "bvec3",
    "bvec4",
    "ivec2",
    "ivec3",
    "ivec4",
    "vec2",
    "vec3",
    "vec4",
    "uvec2",
    "uvec3",
    "uvec4",
    "dvec2",
    "dvec3",
    "dvec4",
    "sampler2d",
    "mat2",
    "mat3",
    "mat4",
    "mat2x3",
    "mat2x4",
    "mat3x2",
    "mat3x4",
    "mat4x2",
    "mat4x3",
    "dmat2",
    "dmat3",
    "dmat4",
    "dmat2x3",
    "dmat2x4",
    "dmat3x2",
    "dmat3x4",
    "dmat4x2",
    "dmat4x3",
    "void",
}

GLSL_BUILTIN_FUNCTION_STRINGS = {
    "abs",
    "acos",
    "acosh",
    "all",
    "any",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "atomicAdd",
    "atomicAnd",
    "atomicCompSwap",
    "atomicCounter",
    "atomicCounterDecrement",
    "atomicCounterIncrement",
    "atomicExchange",
    "atomicMax",
    "atomicMin",
    "atomicOr",
    "atomicXor",
    "barrier",
    "bitCount",
    "bitfieldExtract",
    "bitfieldInsert",
    "bitfieldReverse",
    "ceil",
    "clamp",
    "cos",
    "cosh",
    "cross",
    "degrees",
    "determinant",
    "dFdx",
    "dFdxCoarse",
    "dFdxFine",
    "dFdy",
    "dFdyCoarse",
    "dFdyFine",
    "distance",
    "dot",
    "EmitStreamVertex",
    "EmitVertex",
    "EndPrimitive",
    "EndStreamPrimitive",
    "equal",
    "exp",
    "exp2",
    "faceforward",
    "findLSB",
    "findMSB",
    "floatBitsToInt",
    "floatBitsToUint",
    "floor",
    "fma",
    "fract",
    "frexp",
    "fwidth",
    "fwidthCoarse",
    "fwidthFine",
    "gl_ClipDistance",
    "gl_CullDistance",
    "gl_FragCoord",
    "gl_FragDepth",
    "gl_FrontFacing",
    "gl_GlobalInvocationID",
    "gl_HelperInvocation",
    "gl_InstanceID",
    "gl_InvocationID",
    "gl_Layer",
    "gl_LocalInvocationID",
    "gl_LocalInvocationIndex",
    "gl_NumSamples",
    "gl_NumWorkGroups",
    "gl_PatchVerticesIn",
    "gl_PointCoord",
    "gl_PointSize",
    "gl_Position",
    "gl_PrimitiveID",
    "gl_PrimitiveIDIn",
    "gl_SampleID",
    "gl_SampleMask",
    "gl_SampleMaskIn",
    "gl_SamplePosition",
    "gl_TessCoord",
    "gl_TessLevelInner",
    "gl_TessLevelOuter",
    "gl_VertexID",
    "gl_ViewportIndex",
    "gl_WorkGroupID",
    "gl_WorkGroupSize",
    "greaterThan",
    "greaterThanEqual",
    "groupMemoryBarrier",
    "imageAtomicAdd",
    "imageAtomicAnd",
    "imageAtomicCompSwap",
    "imageAtomicExchange",
    "imageAtomicMax",
    "imageAtomicMin",
    "imageAtomicOr",
    "imageAtomicXor",
    "imageLoad",
    "imageSamples",
    "imageSize",
    "imageStore",
    "imulExtended",
    "intBitsToFloat",
    "interpolateAtCentroid",
    "interpolateAtOffset",
    "interpolateAtSample",
    "inverse",
    "inversesqrt",
    "isinf",
    "isnan",
    "ldexp",
    "length",
    "lessThan",
    "lessThanEqual",
    "log",
    "log2",
    "matrixCompMult",
    "max",
    "memoryBarrier",
    "memoryBarrierAtomicCounter",
    "memoryBarrierBuffer",
    "memoryBarrierImage",
    "memoryBarrierShared",
    "min",
    "mix",
    "mod",
    "modf",
    "noise",
    "noise1",
    "noise2",
    "noise3",
    "noise4",
    "normalize",
    "not",
    "notEqual",
    "outerProduct",
    "packDouble2x32",
    "packHalf2x16",
    "packSnorm2x16",
    "packSnorm4x8",
    "packUnorm",
    "packUnorm2x16",
    "packUnorm4x8",
    "pow",
    "radians",
    "reflect",
    "refract",
    "removedTypes",
    "round",
    "roundEven",
    "sign",
    "sin",
    "sinh",
    "smoothstep",
    "sqrt",
    "step",
    "tan",
    "tanh",
    "texelFetch",
    "texelFetchOffset",
    "texture",
    "textureGather",
    "textureGatherOffset",
    "textureGatherOffsets",
    "textureGrad",
    "textureGradOffset",
    "textureLod",
    "textureLodOffset",
    "textureOffset",
    "textureProj",
    "textureProjGrad",
    "textureProjGradOffset",
    "textureProjLod",
    "textureProjLodOffset",
    "textureProjOffset",
    "textureQueryLevels",
    "textureQueryLod",
    "textureSamples",
    "textureSize",
    "transpose",
    "trunc",
    "uaddCarry",
    "uintBitsToFloat",
    "umulExtended",
    "unpackDouble2x32",
    "unpackHalf2x16",
    "unpackSnorm2x16",
    "unpackSnorm4x8",
    "unpackUnorm",
    "unpackUnorm2x16",
    "unpackUnorm4x8",
    "usubBorrow",
}


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
