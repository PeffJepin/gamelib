from __future__ import annotations

import pathlib
import sys
from abc import abstractmethod, ABC
from typing import Tuple

import moderngl
from PIL import Image


class Asset:
    """
    Maintains some image asset.

    Responsible for loading and cleaning up an OpenGL texture object.
    """

    def __init__(self, label, path_to_src):
        """
        Initialize the Asset object.

        Will record the image size on init but image data will need to be uploaded
        to the GPU before this Asset will be available for rendering.

        Parameters
        ----------
        label : Hashable
            some identifier that could be used as a key
        path_to_src : pathlib.Path
            The path where the source image can be found.
        """
        self.label = label
        self.path = path_to_src
        self.texture = None
        im = Image.open(path_to_src)
        self._size = im.size

    @property
    def size(self):
        return self._size

    def upload_texture(self, ctx) -> None:
        """
        Convenience method to easily upload any image to GPU, though
        in practice using a TextureAtlas may be a superior approach.

        Parameters
        ----------
        ctx : moderngl.Context

        Raises
        ------
        ValueError
            If texture has already been uploaded.
            Uploading a texture to multiple contexts not supported.
        """
        if self.texture is not None:
            raise ValueError(
                "Expected Asset texture to be None. Existing texture must first be released."
            )
        im = Image.open(self.path).transpose(Image.FLIP_TOP_BOTTOM)
        gl_texture = ctx.texture(self._size, 4, im.tobytes())
        self.texture = TextureReference(gl_texture)

    def release_texture(self):
        """
        Releases the texture from GPU memory.

        Note that this shouldn't be called for an Atlased texture,
        instead the TextureAtlas should be released all at once.
        """
        if not self.texture:
            return
        self.texture.gl.release()
        self.texture = None

    def __repr__(self):
        return f"<Asset({self.path})>"


class TextureReference:
    """
    Reference to image data kept on the GPU.

    Atlased images can easily be referenced by many of these Textures
    all pointing to the same gl_texture_object with appropriate uv coords.
    """

    def __init__(self, gl_texture_object, uv=(0, 0, 1, 1)):
        """
        Parameters
        ----------
        gl_texture_object : moderngl.Texture
        uv : tuple[float, float, float, float]
            left, bottom, width, height in 0-1 normalized space
        """
        self.gl = gl_texture_object
        self.uv = uv
        self._x, self._y, self._w, self._h = uv

    @property
    def left(self):
        return self._x

    @property
    def right(self):
        return self._x + self._w

    @property
    def top(self):
        return self._y + self._h

    @property
    def bottom(self):
        return self._y


class TextureAtlas:
    """
    A collection of visual assets combined into a single texture and stored in graphics memory.

    Example
    -------
    Creating a TextureAtlas.

    >>> assets = [Asset('goblin', pathlib.Path('goblin.png')), Asset('orc', pathlib.Path('orc.png'))]
    >>> atlas = TextureAtlas(assets)

    Referencing Assets in the TextureAtlas.

    >>> goblin_asset = atlas['goblin']

    Assets are not automatically uploaded to GPU.

    >>> goblin_asset.texture
    None
    >>> atlas.gl
    None

    Assets are on the GPU after upload_texture call.

    >>> gl_ctx = moderngl.create_context()
    >>> atlas.upload_texture(gl_ctx)
    >>> goblin_asset.texture
    <TextureReference()>

    Assets are no longer on the GPU after release_texture call.

    >>> atlas.release_texture()
    >>> goblin_asset.texture
    None
    """

    def __init__(self, assets, max_size=(2048, 2048), allocation_step=16, components=4):
        """
        Parameters
        ----------
        assets : list[Asset]
        max_size : Tuple[int, int]
            Constraint on final texture size.
        allocation_step : int
            Constraint on row height.
        components : int
            The number of texture components, default 4 for RGBA
        """
        self.gl = None
        self._allocations = SimpleRowAllocator(max_size, allocation_step).pack_assets(
            assets
        )
        self._asset_lookup = {asset.label: asset for asset in assets}
        self._writer = PILWriter(max_size)
        self._components = components
        self._video_memory = None
        self._size = None

    def upload_texture(self, ctx: moderngl.Context):
        """
        Blits all the images into a single image and transfers the data to GPU memory.

        Parameters
        ----------
        ctx : moderngl.Context
            The OpenGL context this texture will be uploaded with.

        """
        atlas_image = self._writer.stitch_assets(self._allocations)
        self.gl = ctx.texture(atlas_image.size, self._components)
        self._create_texture_references(atlas_image.size)
        bytes_ = atlas_image.tobytes()
        self._video_memory = sys.getsizeof(bytes_)
        self._size = atlas_image.size
        self.gl.write(bytes_)

    def release_texture(self):
        """
        Releases the texture memory on the GPU and cleans up the
        TextureReference objects on the Assets.
        """
        if self.gl is None:
            return
        self.gl.release()
        for asset in self._asset_lookup.values():
            asset.texture = None
        self._size = None

    def _create_texture_references(self, atlas_size):
        assert self.gl is not None
        for asset, pos in self._allocations.items():
            x, y = pos[0] / atlas_size[0], pos[1] / atlas_size[1]
            w, h = asset.size[0] / atlas_size[0], asset.size[1] / atlas_size[1]
            uv = (x, y, w, h)
            asset.texture = TextureReference(self.gl, uv)

    @property
    def size(self):
        return self._size

    def __iter__(self):
        return iter(self._asset_lookup.values())

    def __getitem__(self, key) -> Asset:
        return self._asset_lookup[key]

    def __len__(self):
        return len(self._asset_lookup)

    def __repr__(self):
        gpu_memory = round(self._video_memory / 1_000_000, 4)
        return f"<TextureAtlas(num_assets={len(self)}, size={self._size}, {gpu_memory=} MB)>"


class AtlasWriter(ABC):
    """Writes a collection of images into a larger single image and keeps some metadata."""

    @abstractmethod
    def stitch_assets(self, allocations):
        """
        Stitch a group of image files into a single larger texture.

        Parameters
        ----------
        allocations : dict[Asset, tuple[int, int]]


        Returns
        -------
        atlas_image : Image
        """


class AtlasAllocator(ABC):
    """Allocates space to paste smaller images into a single larger texture."""

    max_size: Tuple[int, int]

    @abstractmethod
    def pack_assets(self, assets):
        """
        Pack assets into some geometry, preferably utilizing space efficiently.

        Parameters
        ----------
        assets : list[Asset]
            List of Assets to be packed into this atlas.

        Returns
        -------
        allocations : dict[Asset, tuple[int, int]]
            A Dictionary mapping assets to their px coordinates in the atlas.

        Raises
        ------
        MemoryError
            If max_size can't be respected.
        """


class PILWriter(AtlasWriter):
    def __init__(self, max_size, mode="RGBA"):
        self.max_size = max_size
        self.mode = mode

    def stitch_assets(self, allocations):
        atlas_image = Image.new(self.mode, self.max_size)
        max_x = 0
        max_y = 0

        for asset, pos in allocations.items():
            current_image = Image.open(asset.path).transpose(Image.FLIP_TOP_BOTTOM)
            atlas_image.paste(current_image, pos)
            max_x = max(max_x, pos[0] + asset.size[0])
            max_y = max(max_y, pos[1] + asset.size[1])

        return atlas_image.crop((0, 0, max_x, max_y))


class SimpleRowAllocator(AtlasAllocator):
    def __init__(self, max_size: Tuple[int, int], allocation_step: int):
        self._next_row_height = 0
        self._max_size = max_size
        self._step = allocation_step
        self._max_w, self._max_h = max_size
        self._rows = dict()

    def pack_assets(self, assets):
        return {asset: self.allocate(asset.size) for asset in assets}

    def allocate(self, image_size):
        height = image_size[1]
        amount_over_interval = height % self._step
        if amount_over_interval:
            height += self._step - amount_over_interval

        # try to allocate to existing row first
        if row_pointer := self._rows.get(height):
            if allocation := self._get_allocation(image_size, row_pointer, height):
                return allocation

        # create a new row and allocate there
        row_pointer = self._begin_new_row(height)
        if allocation := self._get_allocation(image_size, row_pointer, height):
            return allocation

        raise MemoryError(f"Unable to allocate to a newly created row. {image_size=}")

    def _get_allocation(self, image_size, row_pointer, height):
        row_x, row_y = row_pointer
        new_x = row_x + image_size[0]
        if new_x <= self._max_w:
            self._rows[height] = (new_x, row_y)
            return row_x, row_y
        else:
            del self._rows[height]

    def _begin_new_row(self, height):
        row_pointer = (0, self._next_row_height)
        self._rows[height] = row_pointer
        self._next_row_height += height
        if self._next_row_height > self._max_h:
            raise MemoryError(
                "Ran out of memory, can not begin new row for allocation."
            )
        return row_pointer
