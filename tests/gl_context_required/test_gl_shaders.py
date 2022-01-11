import pathlib

import numpy as np
import pytest
import gamelib

from gamelib.rendering import shaders
from gamelib import gl


@pytest.fixture(autouse=True, scope="module")
def init_ctx():
    gamelib.init(headless=True)
    yield gamelib.get_context()
    gamelib.exit()


class FakeLock:
    def __init__(self):
        self.times_used = 0

    def __enter__(self):
        self.times_used += 1

    def __exit__(self, *args):
        pass


class TestIndexBuffer:
    def test_indices_from_num_entities(self):
        buffer = shaders.OrderedIndexBuffer(
            order=(0, 1, 2, 0, 2, 3), num_entities=2
        )

        expected = np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7])
        assert np.all(buffer.read().astype(int) == expected)

    def test_changing_number_of_entities(self):
        buffer = shaders.OrderedIndexBuffer(
            order=(0, 1, 2, 0, 2, 3), num_entities=2
        )
        buffer.num_entities = 3

        expected = np.array(
            [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11]
        )
        assert np.all(buffer.read().astype(int) == expected)


class TestAutoBuffer:
    def test_basic_read_write(self):
        array = np.zeros(10)
        buffer = shaders.AutoBuffer(array)

        assert all(buffer.read() == array)

    def test_read_does_not_read_unused_space(self):
        array = np.zeros(5, dtype=gl.int)
        buffer = shaders.AutoBuffer(array, max_elements=10)

        assert np.all(buffer.read() == array)

        array = np.arange(3, dtype=gl.int)
        buffer.write(array)
        assert np.all(buffer.read() == array)

    def test_updates_from_given_array(self):
        array = np.zeros(10)
        buffer = shaders.AutoBuffer(array)

        array += 100
        buffer.update()

        assert all(buffer.read() == array)

    def test_writing_a_numpy_array(self):
        array = np.zeros(10)
        buffer = shaders.AutoBuffer(array)

        new_data = np.arange(10)
        buffer.write(new_data)

        assert all(buffer.read() == new_data)

    def test_writing_bytes(self):
        array = np.zeros(10)
        buffer = shaders.AutoBuffer(array)

        new_data = np.arange(10).tobytes()
        buffer.write(new_data)

        assert buffer.read(bytes=True) == new_data

    def test_dtype_coercion(self):
        array = np.array([1, 2, 3])
        buffer = shaders.AutoBuffer(array, dtype="f4")

        assert array.astype("f4").tobytes() == buffer.read(bytes=True)

    def test_size(self):
        array = np.arange(10, dtype=gl.byte)

        assert shaders.AutoBuffer(array).size == 10
        assert shaders.AutoBuffer(array, dtype=gl.float).size == 40

    def test_size_override_with_max_elements(self):
        array = np.arange(10, dtype=gl.byte)

        assert shaders.AutoBuffer(array, max_elements=20).size == 20

    @pytest.mark.parametrize(
        "gl_type, expected",
        (
            (gl.float, 12),
            (gl.vec2, 6),
            (gl.vec3, 4),
            (gl.vec4, 3),
            (gl.mat3x2, 2),
            (gl.mat3x4, 1),
        ),
    )
    def test_length_base_case(self, gl_type, expected):
        array = np.arange(12)

        buffer = shaders.AutoBuffer(array, gl_type)
        assert len(buffer) == expected

        buffer = shaders.AutoBuffer(array, gl_type, max_elements=20)
        assert len(buffer) == expected

    def test_length_keeps_track_of_elements_in_buffer(self):
        buffer = shaders.AutoBuffer(dtype=gl.int, max_elements=20)

        for i in range(20):
            array = np.arange(i + 1)
            buffer.write(array)
            assert len(buffer) == i + 1

    def test_setting_the_source_array(self):
        buffer = shaders.AutoBuffer(dtype=gl.int, max_elements=20)
        glo = buffer.gl

        array = np.arange(10)
        buffer.use_array(array)
        assert np.all(buffer.read() == array)

        array += 123
        buffer.update()
        assert np.all(buffer.read() == array)

        assert buffer.gl is glo

    def test_setting_the_source_array_too_large(self):
        buffer = shaders.AutoBuffer(dtype=gl.float, max_elements=20)
        array = np.arange(21)

        with pytest.raises(MemoryError):
            buffer.use_array(array)

    def test_memory_error_when_array_is_too_big(self):
        buffer = shaders.AutoBuffer(dtype=gl.float, max_elements=15)
        array = np.arange(16)

        with pytest.raises(MemoryError):
            buffer.write(array)

        with pytest.raises(MemoryError):
            buffer.write(array.astype(gl.float).tobytes())

    def test_using_a_context_manager_for_source_array_access(self):
        array = np.zeros((10,), int)
        lock = FakeLock()

        assert lock.times_used == 0

        # used in initial write
        buffer = shaders.AutoBuffer(array, lock=lock)
        assert lock.times_used == 1

        # used on update
        buffer.update()
        assert lock.times_used == 2

        # used on write
        buffer.write(array)
        assert lock.times_used == 3

        # not used on read - reading from gpu doesn't effect source array
        buffer.read()
        assert lock.times_used == 3


class TestShaderProgram:
    simple_vertex_source = """
        #version 330
        out int value;
        void main() {
            value = gl_VertexID;
        }
    """

    @staticmethod
    def assert_simple_transform(ctx, moderngl_program):
        # using moderngl api to make sure ShaderProgram
        # creates a valid gl shader program
        vao = ctx.vertex_array(moderngl_program, [])
        buffer = ctx.buffer(reserve=10 * 4)
        vao.transform(buffer, vertices=10)

        expected = np.arange(10, dtype="i4").tobytes()
        assert buffer.read() == expected

    def test_loading_a_shader_with_source_string(self):
        program = shaders.ShaderProgram(
            vertex_shader=self.simple_vertex_source,
            varyings=["value"],
        )
        self.assert_simple_transform(program.gl.ctx, program.gl)

    def test_loading_a_shader_with_source_file(self, tmpdir):
        path = pathlib.Path(tmpdir) / "temporary_vertex_shader_source"
        with open(path, "w") as f:
            f.write(self.simple_vertex_source)

        program = shaders.ShaderProgram(vertex_shader=path, varyings=["value"])
        self.assert_simple_transform(program.gl.ctx, program.gl)

    def test_transform_given_vertices(self):
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                
                out int value;
                
                void main() 
                {
                    value = 123;
                }
            """,
            varyings=["value"],
        )

        result = program.transform(vertices=4)
        expected = np.array([123] * 4, gl.int)
        assert all(result == expected)

    def test_transform_multiple_outputs_given_vertices(self):
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                
                out int test_int_output;
                out vec2 test_vec2_output;
                
                void main()
                {
                    test_int_output = 321;
                    test_vec2_output = vec2(123, 456);
                }
            """,
            varyings=["test_int_output", "test_vec2_output"],
        )

        result = program.transform(vertices=10)
        expected_ints = np.array([321] * 10, gl.int)
        expected_vec2s = np.array([(123, 456)] * 10, gl.vec2)
        assert result["test_int_output"].tobytes() == expected_ints.tobytes()
        assert result["test_vec2_output"].tobytes() == expected_vec2s.tobytes()

    def test_transform_must_calculate_result_buffer_size(self):
        in_values = np.array([(1, 2), (3, 4), (5, 6)])
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                
                in vec2 test_inputs;
                out float x;
                out float y;
                
                void main()
                {
                    x = test_inputs.x;
                    y = test_inputs.y;
                } 
            """,
            varyings=["x", "y"],
            buffers={"test_inputs": in_values},
        )

        result = program.transform()
        expected = in_values.astype(gl.float)
        assert result.tobytes() == expected.tobytes()

    def test_inspection(self):
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                in vec2 v_pos;
                void main() 
                {
                    gl_Position = vec4(v_pos, 0, 1);
                } 
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D atlas;
                out vec4 frag;
                void main() 
                {
                    frag = texture(atlas, vec2(0, 0));
                }
            """,
        )
        assert "v_pos" in program.vertex_attributes
        assert len(program.vertex_attributes) == 1
        assert "atlas" in program.uniforms
        assert len(program.uniforms) == 1

    def test_automatic_uniform_sourcing(self, glsl_dtype_and_input):
        gl_type, input_value = glsl_dtype_and_input
        if gl_type == "sampler2D" or gl_type.startswith("b"):
            # not applicable
            return

        uniform = np.array(input_value)
        program = shaders.ShaderProgram(
            vertex_shader=f"""
                #version 330
                
                uniform {gl_type} test_input;
                out {gl_type} test_output;
                
                void main() 
                {{
                    test_output = test_input; 
                }}
            """,
            varyings=["test_output"],
            uniforms={"test_input": uniform},
        )

        expected = gl.coerce_array(uniform, gl_type)
        assert program.transform(1).tobytes() == expected.tobytes()

        uniform += 1
        expected = gl.coerce_array(uniform, gl_type)
        assert program.transform(1).tobytes() == expected.tobytes()

    def test_uniform_array_support(self, glsl_dtype_and_input):
        gl_type, input_value = glsl_dtype_and_input
        if gl_type == "sampler2D" or gl_type.startswith("b"):
            # not applicable
            return

        arr1 = np.array(input_value)
        arr2 = arr1 + 7
        uniform = np.vstack((arr1, arr2))
        program = shaders.ShaderProgram(
            vertex_shader=f"""
                #version 330

                uniform {gl_type} test_input[2];
                out {gl_type} test_output;

                void main() 
                {{
                    test_output = test_input[gl_VertexID]; 
                }}
            """,
            varyings=["test_output"],
            uniforms={"test_input": uniform},
        )

        expected = gl.coerce_array(uniform, gl_type)
        assert program.transform(vertices=2).tobytes() == expected.tobytes()

    def test_use_uniforms_no_current_uniforms(self):
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330

                uniform int input1;
                uniform int input2;
                out int output_value;

                void main() 
                {
                    output_value = input1 + input2;
                }
            """,
            varyings=["output_value"],
        )
        uni1, uni2 = np.array([0], gl.int), np.array([100], gl.int)
        program.use_uniforms(input1=uni1, input2=uni2)
        assert program.transform(1) == uni1 + uni2

        uni1 += 12
        uni2 += 11
        assert program.transform(1) == uni1 + uni2

    def test_use_uniforms_with_existing_uniforms_in_place(self):
        uni1, uni2 = np.array([0], gl.int), np.array([100], gl.int)
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330

                uniform int input1;
                uniform int input2;
                out int output_value;

                void main() 
                {
                    output_value = input1 + input2;
                }
            """,
            varyings=["output_value"],
            uniforms={"input1": uni1, "input2": uni2},
        )
        assert program.transform(1) == uni1 + uni2

        uni3, uni4 = np.array([9], gl.int), np.array([120], gl.int)
        program.use_uniforms(input1=uni3, input2=uni4)
        assert program.transform(1) == uni3 + uni4

        uni3 += 12
        uni4 += 11
        assert program.transform(1) == uni3 + uni4

    def test_buffers_auto_coercing_their_data_type(self, glsl_dtype_and_input):
        gl_type, valid_input_value = glsl_dtype_and_input
        if gl_type in ("sampler2D", "double") or gl_type.startswith("b"):
            # skip a few dtypes that don't apply
            # rather than make a ton of test fixtures
            return

        array = np.array(valid_input_value)
        buffer = np.vstack((array, array, array, array))
        program = shaders.ShaderProgram(
            vertex_shader=f"""
                #version 330
                in {gl_type} inval;
                out {gl_type} outval;
                
                void main()
                {{
                    outval = inval;
                }}
            """,
            varyings=["outval"],
            buffers={"inval": buffer},
        )
        assert np.all(program.transform() == gl.coerce_array(buffer, gl_type))

    def test_use_buffers_no_current_buffers(self):
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                
                in int input1;
                in int input2;
                out int output_value;
                
                void main() 
                {
                    output_value = input1 + input2;
                }
            """,
            varyings=["output_value"],
        )
        array1 = np.arange(6, dtype=gl.int)
        array2 = np.arange(6, 12, dtype=gl.int)

        program.use_buffers(input1=array1, input2=array2)
        assert np.all(program.transform() == array1 + array2)

        array1 += 10
        array2 += 20
        assert np.all(program.transform() == array1 + array2)

    def test_use_buffer_buffers_already_there(self):
        array1 = np.arange(6, dtype=gl.int)
        array2 = np.arange(6, 12, dtype=gl.int)
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330

                in int input1;
                in int input2;
                out int output_value;

                void main() 
                {
                    output_value = input1 + input2;
                }
            """,
            varyings=["output_value"],
            buffers={"input1": array1, "input2": array2},
        )
        assert np.all(program.transform() == array1 + array2)

        array3 = np.arange(100, dtype=gl.int)
        array4 = np.arange(100, dtype=gl.int)
        program.use_buffers(input1=array3, input2=array4)
        assert np.all(program.transform() == array3 + array4)

        array3 += 100
        array4 += 33
        assert np.all(program.transform() == array3 + array4)

    def test_max_entities(self):
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                 
                in int input1;
                out int output1;
                
                void main()
                {
                    output1 = input1;
                }
            """,
            varyings=["output1"],
            max_entities=10,
        )

        for i in range(10):
            array = np.arange(i + 1)
            program.use_buffers(input1=array)
            assert np.all(program.transform() == array)

        with pytest.raises(MemoryError):
            program.use_buffers(input1=np.arange(11))

    def test_num_entities_governed_by_smallest_buffer(self):
        array1 = np.arange(12)
        array2 = np.arange(12)
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                
                in int input1;
                in vec2 input2;
                out int output1;
                out vec2 output2;
                
                void main()
                {
                    output1 = input1;
                    output2 = input2;
                }
            """,
            varyings=["output1", "output2"],
            buffers={"input1": array1, "input2": array2},
        )
        assert program.num_elements == 6

        program.use_buffers(input2=np.arange(20))
        assert program.num_elements == 10

        program.use_buffers(input2=np.arange(100))
        assert program.num_elements == 12

    def test_max_entities_prioritizes_index_buffer_if_given(self):
        index_array = np.arange(8)
        array = np.arange(10)
        program = shaders.ShaderProgram(
            vertex_shader="""
                #version 330
                
                in int input1;
                out int output1;
                
                void main()
                {
                    output1 = input1;
                }
            """,
            varyings=["output1"],
            buffers={"input1": array},
            index_buffer=index_array,
        )

        assert program.num_elements == 8
        program.use_buffers(input1=np.arange(4))
        assert program.num_elements == 8
