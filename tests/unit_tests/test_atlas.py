import pathlib
import tempfile
import time

import pytest
from PIL import Image

from src.gamelib.atlas import TextureAtlas, PILAllocator, PILWriter

TMP = pathlib.Path(tempfile.gettempdir())


class TestTextureAtlas:
    def test_can_be_indexed_for_assets_by_label(self, ctx):
        src = {"label1": make_image_file((4, 4)), "label2": make_image_file((8, 8))}
        atlas = TextureAtlas(ctx, src, max_size=(12, 12), allocation_step=8)

        asset = atlas["label2"]
        assert asset.path == src["label2"] and asset.size == (8, 8)

    @pytest.fixture
    def ctx(self, mocker):
        # test gpu interaction separately
        return mocker.Mock()


class TestPILWriter:
    def test_crops_image_to_appropriate_size(self):
        allocator = PILAllocator(max_size=(64, 64), allocation_step=8)
        writer = PILWriter(allocator)
        src_files = {i: make_image_file((8, 8)) for i in range(4)}

        _, dims, _ = writer.stitch_texture(src_files)
        assert dims == (32, 8)

    def test_returns_assets_that_reference_each_src_image(self):
        allocator = PILAllocator(max_size=(64, 64), allocation_step=8)
        writer = PILWriter(allocator)
        src_files = {i: make_image_file((8, 8)) for i in range(4)}
        _, _, assets = writer.stitch_texture(src_files)

        for i in range(4):
            assert assets[i].path == src_files[i]


class TestPILAllocator:
    def test_allocates_in_existing_row_if_possible(self):
        allocator = PILAllocator(max_size=(32, 32), allocation_step=4)
        im4 = make_image((4, 4))

        first_allocation = allocator.allocate(im4)
        second_allocation = allocator.allocate(im4)

        assert second_allocation[1] == first_allocation[1]

    def test_allocates_in_new_row_if_existing_row_is_full(self):
        allocator = PILAllocator(max_size=(16, 16), allocation_step=4)
        im4 = make_image((4, 4))

        for _ in range(4):
            allocator.allocate(im4)
        fifth_allocation = allocator.allocate(im4)

        assert fifth_allocation == (0, 4)  # new row

    def test_allocates_in_rows_limited_in_size_by_step(self):
        allocator = PILAllocator(max_size=(128, 128), allocation_step=8)
        im8 = make_image((8, 8))
        im10 = make_image((10, 10))
        im12 = make_image((12, 12))
        im30 = make_image((18, 18))

        h1 = 8
        _, y1 = allocator.allocate(im8)
        _, y2 = allocator.allocate(im8)
        assert y1 == y2 == 0

        h2 = 16
        _, y3 = allocator.allocate(im10)
        _, y4 = allocator.allocate(im12)
        assert y3 == y4 == h1

        h3 = 32
        _, y5 = allocator.allocate(im30)
        assert y5 == h1 + h2

    def test_raises_memory_error_with_one_large_image(self):
        allocator = PILAllocator(max_size=(32, 32), allocation_step=4)
        im64 = make_image((64, 64))

        with pytest.raises(MemoryError):
            allocator.allocate(im64)

    def test_raises_memory_error_with_many_small_images(self):
        allocator = PILAllocator(max_size=(32, 32), allocation_step=4)
        im8 = make_image((8, 8))

        for _ in range(16):
            allocator.allocate(im8)

        with pytest.raises(MemoryError):
            allocator.allocate(im8)


def make_image_file(size) -> pathlib.Path:
    path = TMP / (str(time.time()) + ".png")
    im = Image.new("RGBA", size)
    im.save(path)
    return path


def make_image(size) -> Image:
    return Image.new("RGBA", size)


if __name__ == "__main__":
    pytest.main()
