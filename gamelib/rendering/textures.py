import abc

from typing import Tuple

import pygame
import numpy as np
from PIL import Image


class Asset(abc.ABC):
    """Represents some visual resource.

    Assets must implement load, free, tobytes, and shape
    """

    def __init__(self, label, px_depth=4):
        """A subclass should initialize itself with enough
        information to later load its data into cpu memory.

        Parameters
        ----------
        label : Hashable
            some identifier that could be used as a key
        """
        self._px_depth = px_depth
        self.label = label
        self.texture = None

    @abc.abstractmethod
    def load(self):
        """Load the asset into cpu memory."""

    @abc.abstractmethod
    def free(self):
        """Clean-up state from load."""

    @abc.abstractmethod
    def tobytes(self):
        """Return the bytes for the already loaded assets.

        Returns
        -------
        data_for_gpu : bytes
        """

    @abc.abstractmethod
    def shape(self):
        """Return the px dimensions of this Asset after its been loaded.

        Returns
        -------
        im_shape : tuple[int, int]
        """

    def upload_texture(self, ctx, wrap_x=True, wrap_y=True, _free=True):
        """Load the data and upload it to the gpu.

        Parameters
        ----------
        ctx : moderngl.Context
        wrap_x : bool
            should the texture wrap along x-axis?
        wrap_y : bool
            should the texture wrap along y-axis?
        _free : bool
            Should the cpu memory be freed immediately?
            Used by TextureAtlas to index child assets before freeing them.

        Raises
        ------
        ValueError
            If texture has already been uploaded.
            Uploading a texture to multiple contexts not implemented.
        """
        if self.texture is not None:
            raise ValueError(
                "Expected Asset texture to be None. "
                "Existing texture must first be released."
            )
        self.load()
        gl_texture = ctx.texture(self.shape(), self._px_depth, self.tobytes())
        gl_texture.repeat_x = wrap_x
        gl_texture.repeat_y = wrap_y
        self.texture = TextureReference(self.shape(), gl_texture)
        if _free:
            self.free()

    def release_texture(self):
        """Releases the texture from GPU memory."""
        if not self.texture:
            return
        self.texture.gl.release()
        self.texture = None

    def __repr__(self):
        return f"<Asset(label={self.label})>"


class TextAsset(Asset):
    """Using pygame.font module for now to render a bit of text then upload
    that image to the GPU.

    In the future text rendering should get a more sophisticated
    implementation, as this will waste a ton of video memory at scale.
    """

    def __init__(self, label, font_size, color=(255, 255, 255)):
        """Initialize system default font. Label is the Asset label
        and the text to be rendered.

        Parameters
        ----------
        label : str
            Label for the asset is also the text to be displayed.
        font_size : int
        color : tuple
            Opacity not implemented at the moment.
        """
        super().__init__(label)
        self._font = pygame.font.Font(
            pygame.font.get_default_font(), font_size
        )
        self._color = color
        self._surface = None

    def load(self):
        # the array pygame returns for the surface seems to return bgr
        # so the simplest short term fix is this.
        color = (self._color[2], self._color[1], self._color[0])
        self._surface = self._font.render(self.label, True, color)

    def free(self):
        self._surface = None

    def shape(self):
        return self._surface.get_size()

    def tobytes(self):
        arr = pygame.surfarray.pixels2d(self._surface)
        return np.flip(arr.swapaxes(0, 1), 0).tobytes()


class ImageAsset(Asset):
    """An ImageAsset should be pointed at an image file.
    The filetype should be PIL compatible.
    """

    def __init__(self, label, path, px_depth=4):
        """Point the Asset at a file and give it a label.

        Parameters
        ----------
        label : Any
        path : pathlib.Path
        px_depth : int
        """
        super().__init__(label, px_depth)
        self.path = path
        self._im = None

    def load(self):
        self._im = Image.open(self.path).transpose(
            Image.FLIP_TOP_BOTTOM)
        self._px_depth = len(self._im.getbands())

    def free(self):
        self._im = None

    def shape(self):
        return self._im.size

    def tobytes(self):
        return self._im.tobytes()


class TextureAtlas(Asset):
    """A collection of visual assets combined into a
    single texture and stored in graphics memory.
    """

    def __init__(self, label, assets, allocator=None, writer=None, px_depth=4):
        """
        Parameters
        ----------
        label : Any
        assets : list[Asset]
        allocator : AtlasAllocator
        writer : AtlasWriter
        px_depth : int
            The number of texture components, default 4 for RGBA
        """
        super().__init__(label, px_depth)
        self._allocator = allocator or SimpleRowAllocator((2048, 2048), 32)
        self._writer = writer or PILWriter()
        self._asset_lookup = {asset.label: asset for asset in assets}
        self._allocations = None
        self._shape = None

    def load(self):
        for asset in self._asset_lookup.values():
            asset.load()
        self._allocations, self._shape = self._allocator.pack_assets(
            self._asset_lookup.values()
        )

    def free(self):
        for asset in self._asset_lookup.values():
            asset.free()

    def shape(self):
        return self._shape

    def tobytes(self):
        return self._writer.stitch_texture(self._allocations, self._shape)

    def upload_texture(self, ctx, **kwargs):
        # texture references need to be made for all the contained assets.
        super().upload_texture(ctx, _free=False)
        self._create_texture_references()

    def release_texture(self):
        # all the contained assets must have references cleaned up.
        for asset in self._asset_lookup.values():
            asset.texture = None
        super().release_texture()

    def _create_texture_references(self):
        total_width, total_height = self._shape
        for asset, pos in self._allocations.items():
            asset_width, asset_height = asset.shape()
            nx, ny = pos[0] / total_width, pos[1] / total_height
            nw, nh = asset_width / total_width, asset_height / total_height
            uv = (nx, ny, nw, nh)
            asset.texture = TextureReference(
                asset.shape(), self.texture.gl, uv
            )
        self.free()

    def __iter__(self):
        return iter(self._asset_lookup.values())

    def __getitem__(self, key) -> Asset:
        return self._asset_lookup[key]

    def __len__(self):
        return len(self._asset_lookup)

    def __repr__(self):
        return (
            f"<TextureAtlas(num_assets={len(self)},"
            f"size={None if not self.texture else self.texture.size})>"
        )


class TextureReference:
    """
    Reference to image data kept on the GPU.

    Images in an atlas can easily be referenced by pointing to
    the same opengl texture object with appropriate uv coordinates.
    """

    def __init__(self, size, gl_texture_object, uv=(0, 0, 1, 1)):
        """
        Parameters
        ----------
        size : tuple[int, int]
            the px dimensions of the graphic
        gl_texture_object : moderngl.Texture
        uv : tuple[float, float, float, float]
            left, bottom, width, height in 0-1 normalized space
        """
        self.size = size
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


class AtlasWriter(abc.ABC):
    """Writes a collection of images into a larger single
    image and keeps some metadata."""

    @abc.abstractmethod
    def stitch_texture(self, allocations, shape):
        """
        Stitch a group of image files into a single larger texture.

        Parameters
        ----------
        allocations : dict[Asset, tuple[int, int]]
        shape : tuple[int, int]
            the pixel dimensions allocations will fit into.

        Returns
        -------
        atlas_data : bytes
        """


class AtlasAllocator(abc.ABC):
    """Allocates space to paste smaller images
    into a single larger texture."""

    max_size: Tuple[int, int]

    @abc.abstractmethod
    def pack_assets(self, assets) -> tuple:
        """Pack assets into some geometry, preferably
        utilizing space efficiently.

        Parameters
        ----------
        assets : Iterable[Asset]
            List of Assets to be packed into this atlas.

        Returns
        -------
        allocations : dict[Asset, tuple[int, int]]
            A Dictionary mapping assets to their px coordinates in the atlas.
        shape : tuple[int, int]
            the total pixel dimensions to fit these allocations.

        Raises
        ------
        MemoryError:
            If max_size can't be respected.
        """


class PILWriter(AtlasWriter):
    def __init__(self, mode="RGBA"):
        self.mode = mode

    def stitch_texture(self, allocations, shape):
        atlas_image = Image.new(self.mode, shape)
        for asset, pos in allocations.items():
            current_image = Image.frombytes(
                "RGBA", asset.shape(), asset.tobytes()
            )
            atlas_image.paste(current_image, pos)
        return atlas_image.tobytes()


class SimpleRowAllocator(AtlasAllocator):
    """A very simple packing strategy.

    Packs assets in rows.
    Rows are only allowed to be allocated in parameterized steps.
    Tries first to pack an asset into an existing rows first.
    If there are no existing rows of the appropriate size, a new row is opened.
    If an existing row can't fit an asset, that row is closed.
    If the parameterized max_size cannot be respected, raises MemoryError.
    """

    def __init__(self, max_size, allocation_step):
        """Defines the rules for packing assets.

        Parameters
        ----------
        max_size : tuple[int, int]
            The pixel dimensions this allocator is confined to.
        allocation_step : int
            The rows will be allocated in steps by this value.
        """
        self._max_size = max_size
        self._step = allocation_step
        self._max_w, self._max_h = max_size
        self._reset()

    def _reset(self):
        self._current_height = 0
        self._current_width = 0
        self._rows = dict()

    def pack_assets(self, assets):
        result = (
            {asset: self.allocate(asset.shape()) for asset in assets},
            (self._current_width, self._current_height),
        )
        self._reset()
        return result

    def allocate(self, image_size):
        # calculate what row height this image fits into
        height = image_size[1]
        amount_over_interval = height % self._step
        if amount_over_interval:
            height += self._step - amount_over_interval

        # try to allocate to existing row first and return allocation
        if current_location := self._rows.get(height):
            if allocation := self._get_allocation(
                image_size, current_location, height
            ):
                return allocation

        # try to create a new row and return allocation
        current_location = self._begin_new_row(height)
        if allocation := self._get_allocation(
            image_size, current_location, height
        ):
            return allocation

        raise MemoryError(
            f"Unable to allocate to a newly created row. {image_size=}"
        )

    def _get_allocation(
        self, image_size: tuple, current_location: tuple, height: int
    ):
        row_x, row_y = current_location
        new_x = row_x + image_size[0]
        if new_x <= self._max_w:
            self._current_width = max(self._current_width, new_x)
            self._rows[height] = (new_x, row_y)
            return row_x, row_y
        else:
            del self._rows[height]

    def _begin_new_row(self, height):
        current_location = (0, self._current_height)
        self._rows[height] = current_location
        self._current_height += height
        if self._current_height > self._max_h:
            raise MemoryError(
                "Ran out of memory, can not begin new row for allocation."
            )
        return current_location
