"""Functions to transform the imagery data."""
from pathlib import Path

import numpy as np
from PIL import Image


def transform(
    write_path: Path,
    field: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Perform transformations on the data."""
    t_none(write_path, field, mask)
    t_quartile(write_path, field, mask)
    t_rotation(write_path, field, mask)


def t_none(write_path: Path, field: np.ndarray, mask: np.ndarray, tag: str = "t_none") -> None:
    """Apply no transformation, safe data as is."""
    img = Image.fromarray(np.uint8(field), "RGB")
    img.save(write_path / f"{tag}.png")
    img = Image.fromarray(np.uint8(mask), "L")
    img.save(write_path / f"{tag}_mask.png")


def t_quartile(
    write_path: Path,
    field: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Divide the information into four and save each of those."""
    width, height = mask.shape  # 2d array
    for i, (x, y) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        # Slice and recover shape
        field_slice = field[
            (width // 2) * x : (width // 2) * (x + 1), (height // 2) * y : (height // 2) * (y + 1)
        ]
        field_slice = field_slice.repeat(2, axis=0).repeat(2, axis=1)
        mask_slice = mask[
            (width // 2) * x : (width // 2) * (x + 1), (height // 2) * y : (height // 2) * (y + 1)
        ]
        mask_slice = mask_slice.repeat(2, axis=0).repeat(2, axis=1)

        # Normalise masking values
        values = sorted(
            set(np.unique(mask_slice))
            - {
                0,
            }
        )
        for idx, v in enumerate(values):
            mask_slice[mask_slice == v] = idx + 1

        # Store result
        t_none(write_path=write_path, field=field_slice, mask=mask_slice, tag=f"t_quartile_{i}")


def t_rotation(
    write_path: Path,
    field: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Rotate the data."""
    for i in range(4):
        field = np.rot90(field)
        mask = np.rot90(mask)
        if i < 3:
            t_none(write_path=write_path, field=field, mask=mask, tag=f"t_rotation_{i}")
