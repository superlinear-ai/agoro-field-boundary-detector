"""Utilisation functions."""
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from src.radix_co2_reduction.earth_engine.utils import get_dlat_dlng


def adjust_polygon(
    coordinate: Tuple[float, float],
    im_path: Path,
    im_polygon: List[List[Tuple[int, int]]],
    scale: int = 2,
) -> List[Tuple[float, float]]:
    """
    Convert the image-polygon to a polygon in (lat,lng) coordinates.

    :param coordinate: Center coordinate (lag,lng) of the image
    :param im_path: The path where the field-image is stored
    :param im_polygon: Polygon over the image, in pixel coordinates
    :param scale: Scale of the image (for NAIP images, this is 2)
    """
    _, pixel_center = get_image(im_path)

    # Transform the field's polygon
    offsets = []
    for x, y in im_polygon[0]:
        offsets.append(
            get_dlat_dlng(
                lat=coordinate[0],
                dx=scale * (x - pixel_center[0]),
                dy=-scale * (y - pixel_center[1]),
            )
        )
    return [(coordinate[0] + a, coordinate[1] + b) for a, b in offsets]


def get_image(im_path: Path) -> Tuple[Optional[Any], Tuple[int, int]]:
    """Load in the image stored under the given path, together with center pixel."""
    try:
        img = Image.open(im_path).convert("RGB")
        shape = np.asarray(img).shape
        pixel = shape[0] // 2, shape[1] // 2
        return img, pixel
    except UnidentifiedImageError:
        return None, (0, 0)
