#version 330

in vec3 v_pos;

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;

out vec3 world_pos;

void main()
{
    vec4 world_space_transform = model * vec4(v_pos, 1.0);
    world_pos = world_space_transform.xyz;
    gl_Position = proj * view * world_space_transform;
}
