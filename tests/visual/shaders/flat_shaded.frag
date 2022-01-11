#version 330

in vec3 world_pos;
out vec4 frag;

vec3 light_color = vec3(0.9, 0.6, 0.3);
vec3 light_pos = vec3(-400.0, -125.0, 1000.0);
float ambient_strength = 0.15;

void main()
{
    vec3 dx = dFdx(world_pos);
    vec3 dy = dFdy(world_pos);
    vec3 face_normal = normalize(cross(dx, dy));
    vec3 light_dir = normalize(light_pos - world_pos);

    float diffuse_strength = max(dot(face_normal, light_dir), 0.0);
    vec3 diffuse = diffuse_strength * light_color;
    vec3 ambient = light_color * ambient_strength;

    frag = vec4(ambient + diffuse, 1.0);
}
