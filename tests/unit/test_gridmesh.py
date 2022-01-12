from gamelib.geometry import GridMesh


def test_grid_mesh():
    mesh1 = GridMesh(lod=1)
    mesh2 = GridMesh(lod=2)

    assert mesh1.vertices.size == 4 * 3
    assert mesh1.triangles.size == 6

    assert mesh2.vertices.size == 9 * 3
    assert mesh2.triangles.size == 24
