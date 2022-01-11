#!/usr/bin/env python3

from pathlib import Path

import moderngl
import numpy as np
import pygame
from PIL import Image

from gamelib.rendering.textures import (
    TextureAtlas,
    ImageAsset,
    SimpleRowAllocator,
    PILWriter,
    TextAsset,
)

IMGS = Path(__file__).parent / "testing_images"


def render_asset_to_PIL(ctx, asset):
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
    asset.texture.gl.use(0)

    vertices = np.array([-1, -1, -1, 1, 1, 1, 1, -1])
    vbo = ctx.buffer(vertices.astype("f4").tobytes())

    tex_coords = np.array(
        [
            asset.texture.left,
            asset.texture.bottom,
            asset.texture.left,
            asset.texture.top,
            asset.texture.right,
            asset.texture.top,
            asset.texture.right,
            asset.texture.bottom,
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
    fbo = ctx.simple_framebuffer(asset.texture.size)
    fbo.use()
    fbo.clear()
    vao.render()
    return Image.frombytes(
        "RGBA", fbo.size, fbo.read(components=4), "raw", "RGBA", 0, -1
    )


def test_image_asset():
    asset = ImageAsset("gradiant", IMGS / "grad1.png")
    ctx = moderngl.create_standalone_context()
    asset.upload_texture(ctx)
    image = render_asset_to_PIL(ctx, asset)
    image.show("gradiant rendered from asset")


def test_text_asset():
    pygame.init()
    asset = TextAsset("SEND HELP", font_size=200, color=(255, 0, 0, 255))
    ctx = moderngl.create_standalone_context()
    asset.upload_texture(ctx)
    image = render_asset_to_PIL(ctx, asset)
    image.show()


def test_texture_atlas(assets):
    """
    Show side by side comparison of an image loaded from disk vs rendered by
    the gpu. Also show an image of the entire atlas at the end.
    """
    ctx = moderngl.create_standalone_context()
    allocator = SimpleRowAllocator(max_size=(512, 512), allocation_step=32)
    atlas = TextureAtlas("my_atlas", assets, allocator, PILWriter())
    atlas.upload_texture(ctx)

    for asset in atlas:
        if isinstance(asset, ImageAsset):
            rendered = render_asset_to_PIL(ctx, asset)
            loaded = Image.open(asset.path)
            w, h = loaded.size
            side_by_side = Image.new("RGBA", (w * 2, h))
            side_by_side.paste(loaded, (0, 0))
            side_by_side.paste(rendered, (w, 0))
            side_by_side.show("Loaded From File <---> Rendered On GPU")
        else:
            rendered = render_asset_to_PIL(ctx, asset)
            rendered.show()

    entire_atlas = render_asset_to_PIL(ctx, atlas)
    entire_atlas.show()


if __name__ == "__main__":
    test_image_asset()
    test_text_asset()

    image_assets = [
        ImageAsset(i, IMGS / fn) for i, fn in enumerate(IMGS.iterdir())
    ]
    text_assets = [
        TextAsset("Woah", 64),
        TextAsset("Hello, World", 24),
        TextAsset("I had better be red!", 24, (255, 0, 0, 255)),
    ]
    test_texture_atlas(image_assets + text_assets)
