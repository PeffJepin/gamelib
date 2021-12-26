import pytest
from gamelib import resources


@pytest.fixture
def _blank_filesystem(filesystem_maker):
    return filesystem_maker()


@pytest.fixture(autouse=True)
def cleanup(_blank_filesystem):
    resources.set_resource_roots(_blank_filesystem)


def test_discovering_shader_directories(filesystem_maker):
    root = filesystem_maker(
        "dir1/subdir/shaders/shader1.vert",
        "dir1/subdir/shaders/shader1.frag",
        "dir1/subdir/shaders/shader1.tesc",
        "dir1/subdir/shaders/shader1.tese",
        "dir1/subdir/shaders/shader1.geom",
        "dir1/subdir/shaders/shader1.glsl",
        "shaders/shader2.vert",
        "shaders/shader2.frag",
        "shaders/shader3.vert",
        "shaders/shader3.frag",
    )
    directories = resources.discover_directories(root).shaders

    assert len(directories) == 2
    assert root / "dir1" / "subdir" / "shaders" in directories
    assert root / "shaders" in directories


def test_discovering_shader_files(filesystem_maker):
    root = filesystem_maker(
        "dir1/subdir/shaders/shader1.vert",
        "dir1/subdir/shaders/shader1.frag",
        "dir1/subdir/shaders/shader1.tesc",
        "dir1/subdir/shaders/shader1.tese",
        "dir1/subdir/shaders/shader1.geom",
        "shaders/shader2.vert",
        "shaders/shader2.frag",
        "shaders/shader3.vert",
        "shaders/shader3.frag",
    )
    sources = resources.discover_shader_sources(root)

    assert len(sources) == 3
    assert len(sources["shader1"]) == 5
    assert len(sources["shader2"]) == 2
    assert len(sources["shader3"]) == 2


def test_discovering_asset_directories(filesystem_maker):
    root = filesystem_maker(
        "notassets/subdir/assets/file1.jpg",
        "notassets/subdir/assets/file2.png",
        "notassets/subdir/assets/nested/file3.jpg",
        "assets/file4.txt",
    )
    dirs = resources.discover_directories(root).assets

    assert len(dirs) == 2
    assert root / "notassets" / "subdir" / "assets" in dirs
    assert root / "assets" in dirs


def test_discovering_asset_files(filesystem_maker):
    root = filesystem_maker(
        "notassets/subdir/assets/file1.jpg",
        "notassets/subdir/assets/file2.png",
        "notassets/subdir/assets/nested/file3.jpg",
        "assets/file4.txt",
    )
    files = resources.discover_asset_files(root)
    dir_containing_files = root / "notassets" / "subdir" / "assets"

    file1 = dir_containing_files / "file1.jpg"
    file2 = dir_containing_files / "file2.png"
    file3 = dir_containing_files / "nested" / "file3.jpg"

    assert file1 == files["file1"]
    assert file2 == files["file2"]
    assert file3 == files["file3"]
    assert len(files) == 3


def test_set_resource_root(filesystem_maker):
    root = filesystem_maker("assets/test_filename.png")

    with pytest.raises(KeyError):
        resources.find_asset("test_filename")

    resources.set_resource_roots(root)
    assert resources.find_asset("test_filename") is not None


def test_set_resource_root_clears_cache(filesystem_maker):
    fs1 = filesystem_maker("assets/file1.png")
    fs2 = filesystem_maker("assets/file2.png")
    resources.set_resource_roots(fs1)

    assert resources.find_asset("file1") is not None
    with pytest.raises(KeyError):
        resources.find_asset("file2")

    resources.set_resource_roots(fs2)
    assert resources.find_asset("file2") is not None
    with pytest.raises(KeyError):
        resources.find_asset("file1")


def test_add_resources_roots(filesystem_maker):
    fs1 = filesystem_maker("assets/file1.png")
    fs2 = filesystem_maker("assets/file2.png")
    fs3 = filesystem_maker("assets/file3.png")
    resources.add_resource_roots(fs1, fs2, fs3)

    assert resources.find_asset("file1") is not None
    assert resources.find_asset("file2") is not None
    assert resources.find_asset("file3") is not None


def test_find_asset(filesystem_maker):
    root = filesystem_maker("assets/test_file.png")
    resources.set_resource_roots(root)

    assert root / "assets" / "test_file.png" == resources.find_asset(
        "test_file"
    )


def test_find_shader(filesystem_maker):
    root = filesystem_maker("shaders/example.vert", "shaders/example.frag")
    resources.set_resource_roots(root)

    src = resources.find_shader("example")
    assert root / "shaders" / "example.vert" in src
    assert root / "shaders" / "example.frag" in src
