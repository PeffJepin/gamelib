import pytest
from gamelib.core import resources


@pytest.fixture
def _blank_filesystem(tmpdir_maker):
    return tmpdir_maker("nothing_here")


@pytest.fixture(autouse=True)
def cleanup(_blank_filesystem):
    resources.set_content_roots(_blank_filesystem)


@pytest.fixture(
    params=(
        "some_shader.glsl",
        "some_image.png",
        "some_image.jpg",
        "some_geometry.obj",
    )
)
def supported_filename(request):
    return request.param


def test_set_content_roots(tmpdir_maker):
    root = tmpdir_maker("assets/test_filename.png")

    with pytest.raises(KeyError):
        resources.get_image_file("test_filename")

    resources.set_content_roots(root)
    assert resources.get_image_file("test_filename") is not None


def test_set_content_roots_clears_cache(tmpdir_maker):
    fs1 = tmpdir_maker("assets/file1.png")
    fs2 = tmpdir_maker("assets/file2.png")
    resources.set_content_roots(fs1)

    assert resources.get_image_file("file1") is not None
    with pytest.raises(KeyError):
        resources.get_image_file("file2")

    resources.set_content_roots(fs2)
    assert resources.get_image_file("file2") is not None
    with pytest.raises(KeyError):
        resources.get_image_file("file1")


def test_add_content_roots(tmpdir_maker):
    fs1 = tmpdir_maker("assets/file1.png")
    fs2 = tmpdir_maker("assets/file2.png")
    fs3 = tmpdir_maker("assets/file3.png")
    resources.add_content_roots(fs1, fs2, fs3)

    assert resources.get_image_file("file1") is not None
    assert resources.get_image_file("file2") is not None
    assert resources.get_image_file("file3") is not None


def test_get_file_base_case(tmpdir_maker, supported_filename):
    unsupported_filename = "fjkaldf.fadjkfla"
    root = tmpdir_maker(supported_filename, unsupported_filename)
    resources.set_content_roots(root)

    discovered_file = resources.get_file(supported_filename)
    assert discovered_file.name == supported_filename

    with pytest.raises(Exception):
        resources.get_file(unsupported_filename)


def test_get_file_nested_case(tmpdir_maker, supported_filename):
    nested = f"subdir/{supported_filename}"
    root = tmpdir_maker(supported_filename, nested)
    resources.set_content_roots(root)

    # should return whatever it finds first... assume this is not determinate
    assert resources.get_file(supported_filename) is not None

    # parent directory can be specified to choose files with the same name
    not_nested = f"{root.name}/{supported_filename}"
    assert resources.get_file(nested).parent.name == "subdir"
    assert resources.get_file(not_nested).parent.name == root.name

    # only supports one parent directory specification
    with pytest.raises(Exception):
        double_nested = f"{root.name}/{nested}"
        resources.get_file(double_nested)


def test_adding_a_supported_extension_base_case(tmpdir_maker):
    unsupported = "bad_file.fafadf"
    extended_support1 = "good_file.abcde"
    extended_support2 = "good_file.custom_filetype"
    root = tmpdir_maker(unsupported, extended_support1, extended_support2)

    resources.add_supported_extensions(".abcde")
    resources.add_supported_extensions("custom_filetype")  # . optional
    resources.set_content_roots(root)

    assert resources.get_file(extended_support1).name == extended_support1
    assert resources.get_file(extended_support2).name == extended_support2
    with pytest.raises(Exception):
        resources.get_file(unsupported)


def test_adding_a_supported_extension_rechecks_roots(tmpdir_maker):
    filename = "filename.my_file_type"
    root = tmpdir_maker(filename)
    resources.set_content_roots(root)

    with pytest.raises(Exception):
        resources.get_file(filename)
    resources.add_supported_extensions(".my_file_type")

    assert resources.get_file(filename).name == filename


def test_get_shader_file(tmpdir_maker):
    root = tmpdir_maker(
        "shader.glsl",
    )
    resources.set_content_roots(root)
    path = resources.get_shader_file("shader")

    assert path == root / "shader.glsl"


def test_get_image_file(tmpdir_maker):
    root = tmpdir_maker("test_file.png")
    resources.set_content_roots(root)

    assert root / "test_file.png" == resources.get_image_file("test_file")


def test_get_model_file(tmpdir_maker):
    root = tmpdir_maker("cube.obj")
    resources.set_content_roots(root)

    assert root / "cube.obj" == resources.get_model_file("cube")
