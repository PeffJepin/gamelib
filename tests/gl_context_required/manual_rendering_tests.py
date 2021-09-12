from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from src.gamelib.atlas import TextureAtlas, Asset


def render_asset_to_PIL_Image(ctx, asset, atlas):
    program = ctx.program(
        vertex_shader="""
            #version 330

            in vec2 device_coord;
            in vec2 tex_coord;

            out vec2 vertex_tex_coord;

            void main() {
                gl_Position = vec4(device_coord, 0, 1);
                vertex_tex_coord = tex_coord;
            }
        """,
        fragment_shader="""
            #version 330

            in vec2 vertex_tex_coord;

            out vec4 frag_color;

            uniform sampler2D test_texture;

            void main() {
                frag_color = texture(test_texture, vertex_tex_coord);   
            }
        """,
    )
    program["test_texture"] = 0
    atlas.texture.use(0)

    vertices = np.array([-1, -1, -1, 1, 1, 1, 1, -1])
    vbo = ctx.buffer(vertices.astype("f4").tobytes())

    tex_coords = np.array(
        [
            asset.left,
            asset.bottom,
            asset.left,
            asset.top,
            asset.right,
            asset.top,
            asset.right,
            asset.bottom,
        ]
    )
    tex_buf = ctx.buffer(tex_coords.astype("f4").tobytes())

    indices = np.array([0, 1, 2, 0, 2, 3])
    ibo = ctx.buffer(indices.astype("u1").tobytes())

    vao = ctx.vertex_array(
        program,
        [(vbo, "2f", "device_coord"), (tex_buf, "2f", "tex_coord")],
        index_buffer=ibo,
        index_element_size=1,
    )
    fbo = ctx.simple_framebuffer(asset.size)
    fbo.use()
    fbo.clear()
    vao.render()
    return Image.frombytes(
        "RGBA", fbo.size, fbo.read(components=4), "raw", "RGBA", 0, -1
    )


def test_texture_atlas():
    """
    Show side by side comparison of an image loaded from disk vs rendered by the gpu
    Also show an image of the entire atlas at the end.
    """
    ctx = moderngl.create_standalone_context()
    img_dir = Path(__file__).parent / "testing_images"
    src_dict = {i: img_dir / fn for i, fn in enumerate(img_dir.iterdir())}
    atlas = TextureAtlas(ctx, src_dict, max_size=(512, 512), allocation_step=32)

    for asset in atlas:
        rendered = render_asset_to_PIL_Image(ctx, asset, atlas)
        loaded = Image.open(asset.path)
        w, h = loaded.size
        side_by_side = Image.new("RGBA", (w * 2, h))
        side_by_side.paste(loaded, (0, 0))
        side_by_side.paste(rendered, (w, 0))
        side_by_side.show(title="Loaded From File <---> Rendered On GPU")

    mock_atlas_asset = Asset((0, 0, 1, 1), Path("dummy_path"), atlas.size)
    entire_atlas = render_asset_to_PIL_Image(ctx, mock_atlas_asset, atlas)
    entire_atlas.show()


if __name__ == "__main__":
    test_texture_atlas()
