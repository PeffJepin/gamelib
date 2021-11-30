import pytest

from gamelib.textures import (
    ImageAsset,
    TextureAtlas,
    SimpleRowAllocator,
    PILWriter,
)


class TestTextureAtlas:
    def test_can_be_indexed_for_assets_by_label(
        self, mocker, image_file_maker
    ):
        assets = [
            ImageAsset("label1", image_file_maker((4, 4))),
            ImageAsset("label2", image_file_maker((8, 8))),
        ]
        atlas = TextureAtlas(
            "asset1", assets, allocator=mocker.Mock(), writer=mocker.Mock()
        )

        assert atlas["label2"] is assets[1]

    def test_upload_assigns_textures_for_children(self, asset_maker, fake_ctx):
        assets = [asset_maker(4, 4) for _ in range(4)]
        atlas = TextureAtlas("atlas1", assets)
        atlas.upload_texture(fake_ctx)

        for asset in assets:
            assert asset.texture is not None

    def test_release_removes_textures_from_children(
        self, asset_maker, fake_ctx
    ):
        assets = [asset_maker(4, 4) for _ in range(4)]
        allocator = SimpleRowAllocator(max_size=(12, 12), allocation_step=4)
        writer = PILWriter()
        atlas = TextureAtlas("atlas1", assets, allocator, writer)
        atlas.upload_texture(fake_ctx)
        atlas.release_texture()

        for asset in assets:
            assert asset.texture is None

    @pytest.fixture
    def fake_ctx(self, mocker):
        return mocker.Mock()


class TestSimpleRowAllocator:
    def test_allocates_in_existing_row_if_possible(self):
        allocator = SimpleRowAllocator(max_size=(32, 32), allocation_step=4)

        first_allocation = allocator.allocate((4, 4))
        second_allocation = allocator.allocate((4, 4))

        assert second_allocation[1] == first_allocation[1]

    def test_allocates_in_new_row_if_existing_row_is_full(self):
        allocator = SimpleRowAllocator(max_size=(16, 16), allocation_step=4)

        for _ in range(4):
            allocator.allocate((4, 4))
        fifth_allocation = allocator.allocate((4, 4))

        assert fifth_allocation == (0, 4)  # new row

    def test_allocates_in_rows_limited_in_size_by_step(self):
        allocator = SimpleRowAllocator(max_size=(128, 128), allocation_step=8)

        h1 = 8
        _, y1 = allocator.allocate((8, 8))
        _, y2 = allocator.allocate((8, 8))
        assert y1 == y2 == 0

        h2 = 16
        _, y3 = allocator.allocate((10, 10))
        _, y4 = allocator.allocate((12, 12))
        assert y3 == y4 == h1

        h3 = 32
        _, y5 = allocator.allocate((30, 30))
        assert y5 == h1 + h2

    def test_raises_memory_error_with_too_wide_image(self):
        allocator = SimpleRowAllocator(max_size=(32, 32), allocation_step=4)

        with pytest.raises(MemoryError):
            allocator.allocate((100, 10))

    def test_raises_memory_error_with_too_tall_image(self):
        allocator = SimpleRowAllocator(max_size=(32, 32), allocation_step=4)

        with pytest.raises(MemoryError):
            allocator.allocate((10, 100))

    def test_raises_memory_error_with_many_small_images(self):
        allocator = SimpleRowAllocator(max_size=(32, 32), allocation_step=4)

        for _ in range(16):
            allocator.allocate((8, 8))

        with pytest.raises(MemoryError):
            allocator.allocate((8, 8))

    def test_pack_assets(self, asset_maker):
        allocator = SimpleRowAllocator(max_size=(16, 16), allocation_step=8)
        assets = [asset_maker(8, 8) for _ in range(4)]

        expected = (
            {
                assets[0]: (0, 0),
                assets[1]: (8, 0),
                assets[2]: (0, 8),
                assets[3]: (8, 8),
            },
            (16, 16),
        )

        assert expected == allocator.pack_assets(assets)
