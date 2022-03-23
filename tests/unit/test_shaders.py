import functools

import pytest

from gamelib.rendering import shaders
from gamelib.core import gl

from ..conftest import compare_glsl

Shader = functools.partial(shaders.Shader, init_gl=False)


def test_parsing_single_string_shader():
    src = r"""
#version 330

#vert
void main()
{
    int i = 1;
}

#tesc
void main()
{
    int i = 2;
}

#tese
void main()
{
    int i = 3;
}

#geom
void main()
{
    int i = 4;
}

#frag
void main()
{
    int i = 5;
}
"""
    shader = Shader(src=src)
    expected_template = """
#version 330

void main()
{{
    int i = {};
}}
"""

    assert compare_glsl(shader.code.vert, expected_template.format(1))
    assert compare_glsl(shader.code.tesc, expected_template.format(2))
    assert compare_glsl(shader.code.tese, expected_template.format(3))
    assert compare_glsl(shader.code.geom, expected_template.format(4))
    assert compare_glsl(shader.code.frag, expected_template.format(5))


def test_parsing_vertex_shader_data():
    src = """
#version 330

#vert
in vec3 v_pos;
in float scale;
out vec3 world_pos;
void main() {}

#frag
in vec3 world_pos;
out vec4 frag;
void main() {}
"""

    shader = Shader(src=src)
    expected_vertex_attributes = (
        shaders.GLSLAttribute("v_pos", gl.vec3, 1),
        shaders.GLSLAttribute("scale", gl.float, 1),
    )
    expected_vertex_output = shaders.GLSLVertexOutput("world_pos", gl.vec3, 1)

    for tok in expected_vertex_attributes:
        assert shader.meta.attributes[tok.name] == tok
    assert len(shader.meta.attributes) == len(expected_vertex_attributes)

    assert shader.meta.vertex_outputs["world_pos"] == expected_vertex_output
    assert len(shader.meta.vertex_outputs) == 1


def test_parsing_uniforms():
    src = """
#version 330
uniform float u_float;

#vert
in vec3 v_pos;
out vec3 world_pos;
uniform vec4 u_vec4;
void main(){}

#frag
in vec3 world_pos;
out vec4 frag;
uniform vec4 u_vec4;
uniform mat4 u_mat4[2];
void main() {}
"""

    shader = Shader(src=src)
    expected_uniforms = (
        shaders.GLSLUniform("u_float", gl.float, 1),
        shaders.GLSLUniform("u_vec4", gl.vec4, 1),
        shaders.GLSLUniform("u_mat4", gl.mat4, 2),
    )

    for tok in expected_uniforms:
        assert shader.meta.uniforms[tok.name] == tok
    assert len(shader.meta.uniforms) == 3


def test_error_when_no_source_is_given():
    with pytest.raises(ValueError) as excinfo:
        shader = Shader()

    assert "no source specified" in str(excinfo.value).lower()


def test_error_when_name_and_src_are_both_given():
    with pytest.raises(ValueError) as excinfo:
        shader = Shader("filename", src="my source string")

    assert "mutually exclusive" in str(excinfo.value).lower()


@pytest.mark.parametrize(
    "sig,desc",
    (
        # fmt: off
        ("func1()", shaders.GLSLFunctionDefinition("func1", (), ())),
        ("func2(int i, int j)", shaders.GLSLFunctionDefinition("func2", ("int i", "int j"), (None, None))),
        ("func3(int i, int j=1)", shaders.GLSLFunctionDefinition("func3", ("int i", "int j"), (None, "1"))),
        ("func4(vec2 p=vec2(1, 3), int i=1)", shaders.GLSLFunctionDefinition("func4", ("vec2 p", "int i"), ("vec2(1, 3)", "1"))),
        # fmt: on
    ),
)
def test_function_parsing(sig, desc):
    src = (
        """
#version 330
#vert
void %s {}
void main() {}
    """
        % sig
    )

    assert Shader(src=src).meta.functions[desc.name] == desc


def test_kwargs_after_positional():
    src = """#version 330
#vert
void my_func(int i, int j=2, int k) {}
void main() {}
    """

    with pytest.raises(SyntaxError) as excinfo:
        Shader(src=src)
    error_message = str(excinfo.value)

    assert "default values" in error_message
    assert "line 3" in error_message


def test_missing_positional_args():
    src = """#version 330
#vert
void my_func(int i, int j=2) {}
void main() {
    my_func(j=1);
}
    """

    with pytest.raises(SyntaxError) as excinfo:
        Shader(src=src)
    error_message = str(excinfo.value).lower()

    assert "missing" in error_message
    assert "line 5" in error_message


def test_unknown_kwarg():
    src = """#version 330
#vert
void my_func(int i, int j=2) {}
void main() {
    my_func(h=1);
}
    """

    with pytest.raises(SyntaxError) as excinfo:
        Shader(src=src)
    error_message = str(excinfo.value).lower()

    assert "unknown" in error_message
    assert "line 5" in error_message


def test_function_parsing_with_multiline_syntax():
    src = """
#version 330
#vert
void my_func(int i,
             vec2 p=vec2(1, 1),
             int j=2
             ) {}
void main() {}
    """

    shader = Shader(src=src)
    expected_desc = shaders.GLSLFunctionDefinition(
        "my_func", ("int i", "vec2 p", "int j"), (None, "vec2(1, 1)", "2")
    )
    assert shader.meta.functions["my_func"] == expected_desc


def test_functions_are_replaced_with_proper_glsl_syntax():
    src = """
#version 330
#vert
void func1()
{
}
void func2(int i, vec2 p=vec2(2, 2), int j=3)
{
}

void main()
{
    func1();
    func2(2);
    func2(2, vec2(3, 3), 4);
    func2(i=2, p=vec2(3, 3), j=4);
    func2(i=2, j=4);
}
    """
    expected = """
#version 330
void func1()
{
}
void func2(int i, vec2 p, int j)
{
}

void main()
{
    func1();
    func2(2, vec2(2, 2), 3);
    func2(2, vec2(3, 3), 4);
    func2(2, vec2(3, 3), 4);
    func2(2, vec2(2, 2), 4);
}
    """

    shader = Shader(src=src)

    assert compare_glsl(shader.code.vert, expected)
