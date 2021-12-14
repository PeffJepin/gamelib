import pathlib
import traceback
import importlib.util
import time

import gamelib
import moderngl
from gamelib import resources
from gamelib import rendering

START_TAG = "# BEGIN SCRIPT MAINTAINED REGION"
END_TAG = "# END SCRIPT MAINTAINED REGION"
IMPORTS = ("# IMPORTS", "import numpy as np",)

CWD = pathlib.Path.cwd()
current_source_files = ()
current_shader_name = ""
current_source_dir = CWD
current_shader: rendering.ShaderProgram = None
conf_path = lambda: current_source_dir / f"{current_shader_name}.conf.py"
conf = None
registry = dict()
should_render = False


class ConfigError(Exception):
    pass


def format_conf_source(body):
    imports = "\n".join(IMPORTS)
    components = (START_TAG, imports, body, END_TAG, "")
    return "\n".join(components)


def pick_a_shader():
    global current_source_files
    global current_shader_name
    global current_source_dir

    shader_sources = resources.discover_shader_sources(CWD)
    shader_names = list(shader_sources.keys())
    for i, name in enumerate(shader_names):
        print(f"{i} - {name}")
    prompt = "Pick a shader: "
    while (chosen_idx := int(input(prompt))) not in range(len(shader_names)):
        # keep prompting until valid input
        pass
    current_shader_name = shader_names[chosen_idx]
    current_source_files = shader_sources[current_shader_name]
    current_source_dir = current_source_files[0].parent
    print(f"rendering with {current_shader_name!r} from {current_source_dir}")


def load_config():
    global conf
    assert current_shader_name is not None
    path = conf_path()

    if not path.exists():
        with open(path, "w") as f:
            f.write(format_conf_source(""))

    update_config_source()
    spec = importlib.util.spec_from_file_location("conf", path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)


def attach_config_buffers():
    for name, attr in current_shader.vertex_attributes.items():
        data = getattr(conf, name).astype(attr.shape)
        current_shader.use_buffer(name, data)


def update_config_source():
    path = conf_path()
    with open(path, "r") as f:
        lines = f.readlines()

    # find the markers for script maintained data
    start_idx = end_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == START_TAG:
            start_idx = i
        elif line.strip() == END_TAG:
            end_idx = i
    if start_idx == -1 or end_idx == -1:
        raise ConfigError("Couldn't find placeholder tags.")

    # separate user source from script source
    starting_lines = lines[:start_idx]
    ending_lines = lines[end_idx:]
    ending_lines.pop(0)

    # parse script source
    deletion_tags = []
    existing_setup = {}
    for raw_line in lines[start_idx + 1 : end_idx]:
        line = raw_line.strip()
        if line.startswith("# DELETED"):
            # decrement existing deletion tags, remove at n = 0
            n = int(line[10])
            n -= 1
            if n > 0:
                deletion_tags.append(f"# DELETED({n}{line[11:]}")
        else:
            # parse existing values for setup variables
            for i, c in enumerate(line):
                if c == "=":
                    name = line[:i].strip()
                    value = line[i + 1 :].strip()
                    existing_setup[name] = value
                    break

    # make lines for shader variables
    attrs = current_shader.vertex_attributes
    uniforms = current_shader.uniforms
    new_lines = []
    default = "np.array([])"
    new_lines.append("# BUFFERS")
    for name in attrs.keys():
        value = existing_setup.pop(name) if name in existing_setup else default
        new_lines.append(f"{name} = {value}")
    new_lines.append("# UNIFORMS")
    for name in uniforms.keys():
        value = existing_setup.pop(name) if name in existing_setup else default
        new_lines.append(f"{name} = {value}")

    # compose script maintained source code
    orphaned_lines = [
        f"# DELETED(3) {k} = {v}" for k, v in existing_setup.items()
    ]
    all_lines = new_lines + orphaned_lines + deletion_tags
    body = "\n".join(all_lines)

    # write the new config file
    with open(path, "w") as f:
        f.writelines(starting_lines)
        f.write(format_conf_source(body))
        f.writelines(ending_lines)
    registry[path] = path.stat().st_mtime


def _find_existing_value(name, existing_lines):
    # find line like: "name = value"
    # and return "value"
    for line in existing_lines:
        if line.startswith(name):
            for i, c in enumerate(line):
                if c == "=":
                    return line[i + 1 :]
    return ""


def init_shader():
    global current_shader
    global should_render

    for path in current_source_files:
        registry[path] = path.stat().st_mtime

    try:
        current_shader = rendering.ShaderProgram(name=current_shader_name)
        load_config()
        attach_config_buffers()
        should_render = True
    except (moderngl.error.Error, ConfigError):
        traceback.print_exc()
        should_render = False


def check_for_updates():
    for path in current_source_dir.iterdir():
        if not path.name.startswith(current_shader_name):
            continue
        if path not in registry:
            return update_shader()
        if path.stat().st_mtime != registry[path]:
            return update_shader()


def update_shader():
    global current_shader
    global current_source_files

    all_sources = resources.discover_shader_sources(current_source_dir)
    current_source_files = all_sources[current_shader_name]
    init_shader()


def render():
    global should_render

    if not should_render:
        return

    try:
        current_shader.render()
    except moderngl.error.Error:
        traceback.print_exc()
        should_render = False


def run():
    pick_a_shader()
    window = gamelib.init()
    init_shader()

    previous_time = time.time()
    while not window.is_closing:
        window.clear()
        if time.time() - previous_time < 0.5:
            check_for_updates()
            previous_time = time.time()
        render()
        window.swap_buffers()


if __name__ == "__main__":
    run()
