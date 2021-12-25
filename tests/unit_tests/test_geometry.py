from gamelib import geometry


def test_grid_mesh():
    mesh1 = geometry.GridMesh(lod=1)
    mesh2 = geometry.GridMesh(lod=2)

    assert mesh1.vertices.size == 4 * 3
    assert mesh1.indices.size == 6

    assert mesh2.vertices.size == 9 * 3
    assert mesh2.indices.size == 24
