from __future__ import annotations

import abc
import pathlib
import sys
from typing import Tuple, Dict

import PIL
import moderngl


class TextureAtlas:
    """
    A collection of visual assets combined into a single texture and stored in graphics memory.

    Example
    -------
    >>> gl_ctx = moderngl.create_context()
    >>> src_files = {'goblin': pathlib.Path('goblin.png'), 'orc': pathlib.Path('orc.png')}
    >>> atlas = TextureAtlas(gl_ctx, src_files)
    >>> goblin_asset = atlas['goblin']
    >>> texture_buffer_object = atlas.texture
    """

    def __init__(
        self,
        context,
        source_images,
        max_size=(2048, 2048),
        allocation_step=16,
        writer=None
    ):
        """
        Parameters
        ----------
        context : moderngl.Context
            Wrapper around an OpenGL context.
        source_images : dict[Hashable, pathlib.Path]
            Collection of image files to be present in this atlas.
            The keys are used to later retrieve the created Assets.
        max_size : Tuple[int, int]
            Constraint on final texture size.
        allocation_step : int
            Constraint on row height.
        writer : AtlasWriter
            An optional override. See AtlasWriter and AtlasAllocator Protocols.
        """
        # helper classes for creating the texture
        if writer is None:
            allocator = PILAllocator(max_size, allocation_step)
            writer = PILWriter(allocator)
        else:
            writer = writer

        bytes_, self._size, self._asset_lookup = writer.stitch_texture(source_images)
        self._video_memory = sys.getsizeof(bytes_)
        self.texture = context.texture(self._size, 4, bytes_)

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


class Asset:
    """Reference to data kept on the gpu."""

    def __init__(self, bounding_box, path_to_src, src_img_size):
        """
        Parameters
        ----------
        bounding_box : Tuple[float, float, float, float]
            Normalized region in a larger texture. (x, y, w, h)
        path_to_src : pathlib.Path
            The path where the source image can be found.
        src_img_size : Tuple[int, int]
            The pixel dimensions of the source image.
        """
        self.path = path_to_src
        self._px_size = src_img_size
        self._x, self._y, self._w, self._h = bounding_box

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

    @property
    def size(self):
        return self._px_size

    def __repr__(self):
        return f'<Asset({self.path})>'


class AtlasWriter(abc.ABC):
    """Writes a collection of images into a larger single image and keeps some metadata."""

    def stitch_texture(self, src_image_files) -> Tuple[bytes, tuple, dict]:
        """
        Stitch a group of image files into a single larger texture.

        Parameters
        ----------
        src_image_files : dict[Hashable, pathlib.Path]
            source files should be supplied in a dict where the keys will be used to later access Assets.

        Returns
        -------
        data : bytes
            The composite image data.
        dims : tuple[int, int]
            The composite image dimensions.
        assets : Dict[Hashable, Asset]
            The Assets representing each source image.
            Keys are the same as those passed in with source.

        Raises
        ------
        MemoryError:
            When allocation exceeds allocators max_size attribute.
        """


class AtlasAllocator(abc.ABC):
    """Allocates space to paste smaller images into a single larger texture."""

    max_size: Tuple[int, int]

    def allocate(self, image):
        """
        Parameters
        ----------
        image
            The current image requesting allocation.

        Returns
        -------
        allocation : tuple[int, int]
            The x, y coordinate where this image should be placed.

        Raises
        ------
        MemoryError
            If max_size can't be respected.
        """


class PILWriter(AtlasWriter):
    def __init__(self, allocator: PILAllocator, mode="RGBA"):
        self._allocator = allocator
        self._atlas_image = PIL.Image.new(mode, allocator.max_size)
        self._metadata = dict()
        self._bounding_boxes = dict()

    def stitch_texture(self, src_image_files):
        for label, path in src_image_files.items():
            current_image = PIL.Image.open(path).transpose(PIL.Image.FLIP_TOP_BOTTOM)
            x, y = self._allocator.allocate(current_image)
            self._metadata[label] = (path, (x, y, *current_image.size))
            self._atlas_image.paste(current_image, (x, y))

        highest_x, highest_y = self._allocator.highest_x, self._allocator.current_y
        self._atlas_image = self._atlas_image.crop((0, 0, highest_x, highest_y))
        assets = {
            label: self._create_asset(metadata)
            for label, metadata in self._metadata.items()
        }
        return self._atlas_image.tobytes(), self._atlas_image.size, assets

    def _create_asset(self, metadata):
        path, bounding_box = metadata
        x, y, w, h = bounding_box
        atlas_w, atlas_h = self._atlas_image.size
        normalized_texture_coordinates = (
            x / atlas_w,
            y / atlas_h,
            w / atlas_w,
            h / atlas_h,
        )
        return Asset(normalized_texture_coordinates, path, (w, h))


class PILAllocator(AtlasAllocator):
    def __init__(self, max_size: Tuple[int, int], allocation_step: int):
        self.highest_x = 0
        self.current_y = 0
        self.max_size = max_size
        self._step = allocation_step
        self._max_w, self._max_h = max_size
        self._rows = dict()

    def allocate(self, image: PIL.Image):
        height = image.height
        amount_over_interval = height % self._step
        if amount_over_interval:
            height += self._step - amount_over_interval

        # try to allocate to existing row first
        if row_pointer := self._rows.get(height):
            if allocation := self._get_allocation(image, row_pointer, height):
                return allocation

        # create a new row and allocate there
        row_pointer = self._begin_new_row(height)
        if allocation := self._get_allocation(image, row_pointer, height):
            return allocation

        raise MemoryError(f"Unable to allocate to a newly created row. {image.size=}")

    def _get_allocation(self, image, row_pointer, height):
        row_x, row_y = row_pointer
        new_x = row_x + image.width
        if new_x <= self._max_w:
            self._rows[height] = (new_x, row_y)
            if new_x > self.highest_x:
                self.highest_x = new_x
            return row_x, row_y
        else:
            del self._rows[height]

    def _begin_new_row(self, height):
        row_pointer = (0, self.current_y)
        self._rows[height] = row_pointer
        self.current_y += height
        if self.current_y > self._max_h:
            raise MemoryError(
                "Ran out of memory, can not begin new row for allocation."
            )
        return row_pointer
