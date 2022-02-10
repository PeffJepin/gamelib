import pytest

from gamelib.rendering import shaders
from gamelib.core import resources

from ..conftest import compare_glsl


@pytest.fixture(autouse=True, scope="function")
def shader_dir(tempdir):
    return tempdir


@pytest.fixture
def glsl(shader_dir):
    file = shader_dir / "test.glsl"
    file.touch()
    resources.set_content_roots(shader_dir)
    yield file


@pytest.fixture
def include_glsl(shader_dir):
    file = shader_dir / "test_include.glsl"
    file.touch()
    resources.set_content_roots(shader_dir)
    yield file


@pytest.fixture
def vert(shader_dir):
    file = shader_dir / "test.vert"
    file.touch()
    resources.set_content_roots(shader_dir)
    yield file


@pytest.fixture
def tesc(shader_dir):
    file = shader_dir / "test.tesc"
    file.touch()
    resources.set_content_roots(shader_dir)
    yield file


@pytest.fixture
def tese(shader_dir):
    file = shader_dir / "test.tese"
    file.touch()
    resources.set_content_roots(shader_dir)
    yield file


@pytest.fixture
def geom(shader_dir):
    file = shader_dir / "test.geom"
    file.touch()
    resources.set_content_roots(shader_dir)
    yield file


@pytest.fixture
def frag(shader_dir):
    file = shader_dir / "test.frag"
    file.touch()
    resources.set_content_roots(shader_dir)
    yield file


def test_loading_single_file_shader(glsl):
    version = "#version 330"
    common = "jlkfjvlkdsjg;l"
    vert = "void main() {1}"
    tesc = "void main() {2}"
    tese = "void main() {3}"
    geom = "void main() {4}"
    frag = "void main() {5}"
    with open(glsl, "w") as f:
        f.write(
            f"""
            {version}
            {common}
            #vert
            {vert}
            #tesc
            {tesc}
            #tese
            {tese}
            #geom
            {geom}
            #frag
            {frag}
        """
        )

    parsed = shaders.Shader.read_file("test").code
    for actual, stage in zip(
        (parsed.vert, parsed.tesc, parsed.tese, parsed.geom, parsed.frag),
        (vert, tesc, tese, geom, frag),
    ):
        expected = f"""
            {version}
            {common}
            {stage}
        """
        assert compare_glsl(actual, expected)


def test_loading_separate_file_shader(vert, tesc, tese, geom, frag):
    vert_s = "#version 330\nvoid main() {1}"
    tesc_s = "#version 330\nvoid main() {2}"
    tese_s = "#version 330\nvoid main() {3}"
    geom_s = "#version 330\nvoid main() {4}"
    frag_s = "#version 330\nvoid main() {5}"

    for file, src in zip(
        (vert, tesc, tese, geom, frag),
        (vert_s, tesc_s, tese_s, geom_s, frag_s),
    ):
        with open(file, "w") as f:
            f.write(src)

    parsed = shaders.Shader.read_file("test").code
    for actual, src in zip(
        (parsed.vert, parsed.tesc, parsed.tese, parsed.geom, parsed.frag),
        (vert_s, tesc_s, tese_s, geom_s, frag_s),
    ):
        assert compare_glsl(actual, src)


def test_include_directive(vert, include_glsl):
    incl_src = """
        afkgjlkfja
        fjkladjfkadf
        ajdklfjadjflkadd
    """
    vert_src1 = """
        #version 400
        #include <test_include.glsl>
        void main() {123}
    """
    vert_src2 = """
        #version 400
        #include <test_include>
        void main() {123}
    """
    expected = f"""
        #version 400
        {incl_src}
        void main() {{123}}
    """
    with open(include_glsl, "w") as f:
        f.write(incl_src)

    with open(vert, "w") as f:
        f.write(vert_src1)
    parsed = shaders.Shader.read_file("test").code
    assert compare_glsl(parsed.vert, expected)

    with open(vert, "w") as f:
        f.write(vert_src2)
    parsed = shaders.Shader.read_file("test").code
    assert compare_glsl(parsed.vert, expected)


def test_source_tracks_where_it_came_from(glsl):
    version = "#version 330"
    vert = "void main() {1}"
    with open(glsl, "w") as f:
        f.write(
            f"""
            {version}
            #vert
            {vert}
            """
        )

    file_shader = shaders.Shader.read_file(glsl.name)
    assert file_shader.files == [glsl]

    str_shader = shaders.Shader.read_string(
        """
        #version 330
        #vert
        void main() {}
        """
    )
    assert str_shader.files is None
