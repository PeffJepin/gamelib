from gamelib.rendering import glslutils
from gamelib import gl

from ..conftest import compare_glsl


def test_parsing_single_string_shader():
    version = "#version 330"
    common_top = "fjakdlfjkad"
    common_middle = "zsdfafjzfzv"
    common_bottom = "jflkajfladjf"
    vert = """
        void main()
        {
            int i = 1;
        }
    """
    tesc = """
        void main()
        {
            int i = 2;
        }
    """
    tese = """
        void main(){
            int i = 3;
        }
    """
    geom = """
        void main()
        { int i = 4; }
    """
    frag = """
        void main() { int i = 5; }
    """

    source_string = f"""
    {version}
    {common_top}
    #vert
    {vert}
    #frag
    {frag}
    {common_middle}
    #tesc
    {tesc}
    #tese
    {tese}
    #geom
    {geom}
    {common_bottom}
    """

    common = "\n".join((common_top, common_middle, common_bottom))
    parsed = glslutils.ShaderData.read_string(source_string).code

    for parsed_stage, stage_src in zip(
        (parsed.vert, parsed.tesc, parsed.tese, parsed.geom, parsed.frag),
        (vert, tesc, tese, geom, frag),
    ):
        expected = f"""
        {version}
        {common}
        {stage_src}
        """
        assert compare_glsl(expected, parsed_stage)


def test_parsing_individual_string_shaders():
    vert = """
        #version 330
        void main()
        {
            int i = 1;
        }
    """
    tesc = """
        #version 330
        void main()
        {
            int i = 2;
        }
    """
    tese = """
        #version 330
        void main(){
            int i = 3;
        }
    """
    geom = """
        #version 330
        void main()
        { int i = 4; }
    """
    frag = """
        #version 330
        void main() { int i = 5; }
    """

    code = glslutils.ShaderData.read_strings(vert, tesc, tese, geom, frag).code
    assert compare_glsl(code.vert, vert)
    assert compare_glsl(code.tesc, tesc)
    assert compare_glsl(code.tese, tese)
    assert compare_glsl(code.geom, geom)
    assert compare_glsl(code.frag, frag)


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

    meta = glslutils.ShaderData.read_string(src).meta

    assert glslutils.TokenDesc("v_pos", gl.vec3, 1) in meta.attributes.values()
    assert (
        glslutils.TokenDesc("scale", gl.float, 1) in meta.attributes.values()
    )
    assert len(meta.attributes) == 2
    assert (
        glslutils.TokenDesc("world_pos", gl.vec3, 1)
        in meta.vertex_outputs.values()
    )
    assert len(meta.vertex_outputs) == 1


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
    meta = glslutils.ShaderData.read_string(src).meta

    assert (
        glslutils.TokenDesc("u_float", gl.float, 1) in meta.uniforms.values()
    )
    assert glslutils.TokenDesc("u_vec4", gl.vec4, 1) in meta.uniforms.values()
    assert glslutils.TokenDesc("u_mat4", gl.mat4, 2) in meta.uniforms.values()
    assert len(meta.uniforms) == 3


def test_shader_source_code_hash():
    src = """
        #version 330
        #vert
        in vec3 v_pos;
        void main(){}
    """
    shader1 = glslutils.ShaderData.read_string(src)
    shader2 = glslutils.ShaderData.read_string(src)
    shader3 = glslutils.ShaderData.read_string(src + "some difference")

    assert hash(shader1) == hash(shader2)
    assert hash(shader1) != hash(shader3)
