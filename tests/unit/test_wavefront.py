# fmt: off
from gamelib import geometry
from gamelib.core import gl

import pytest
import numpy as np


@pytest.fixture
def tmpfile(tempdir):
    path = tempdir / "test.obj"
    path.touch()
    return path


def test_minimal_obj_file(tmpfile):
    source = """
v 0 0 0
v 0 -1.11 0
v 1.11 0 0
f 1 2 3
"""
    with open(tmpfile, 'w') as f:
        f.write(source)

    vertices = np.array(
        [(0,    0,    0),
         (0,   -1.11, 0),
         (1.11, 0,    0)], 
        dtype=gl.vec3
    )
    triangles = np.array((0, 1, 2), gl.uvec3)
    parsed = geometry.load_model(tmpfile)

    assert np.all(parsed.vertices == vertices)
    assert np.all(parsed.indices == triangles)


def test_with_vertex_normals_base_case(tmpfile):
    source = """
v 0 0 0
v 0 1.11 0
v 1.11 1.11 0
v 1.11 0 0
vn 0 0 1
vn 0 0 3
vn 1 0 0
vn 0 -1 0
f 1//1 2//2 3//3
f 1//1 3//3 4//4
"""
    with open(tmpfile, 'w') as f:
        f.write(source)

    vertices = np.array(
        [(0,    0,    0), 
         (0,    1.11, 0), 
         (1.11, 1.11, 0), 
         (1.11, 0,    0)], 
        dtype=gl.vec3
    )

    normals = np.array(
        [(0,  0, 1), 
         (0,  0, 1), 
         (1,  0, 0), 
         (0, -1, 0)], 
        dtype=gl.vec3
    )

    triangles = np.array(
        [(0, 1, 2), 
         (0, 2, 3)],
        dtype=gl.uvec3
    )

    parsed = geometry.load_model(tmpfile)
    assert np.all(parsed.vertices == vertices)
    assert np.all(parsed.normals == normals)
    assert np.all(parsed.indices == triangles)


def test_vertex_data_out_of_order(tmpfile):
    source = """
v 0 0 0
v 0 1.11 0
v 1.11 1.11 0
v 1.11 0 0
vn 0 0 1
vn 0 0 -3.111
vn 1 0 0
vn 0 1 0
f 1//3 2//4 3//1
f 1//3 3//1 4//2
"""
    with open(tmpfile, 'w') as f:
        f.write(source)

    vertices = np.array(
        [(0,    0,    0), 
         (0,    1.11, 0), 
         (1.11, 1.11, 0), 
         (1.11, 0,    0)], 
        dtype=gl.vec3
    )

    normals = np.array(
        [(1, 0,  0), 
         (0, 1,  0), 
         (0, 0,  1), 
         (0, 0, -1)], 
        dtype=gl.vec3
    )

    triangles = np.array(
        [(0, 1, 2), 
         (0, 2, 3)], 
        dtype=gl.uvec3
    )

    parsed = geometry.load_model(tmpfile)
    assert np.all(parsed.vertices == vertices)
    assert np.all(parsed.normals == normals)
    assert np.all(parsed.indices == triangles)


def test_faces_that_arent_triangles(tmpfile):
    source = """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""
    with open(tmpfile, 'w') as f:
        f.write(source)

    parsed = geometry.load_model(tmpfile)
    assert len(parsed.indices) == 2
