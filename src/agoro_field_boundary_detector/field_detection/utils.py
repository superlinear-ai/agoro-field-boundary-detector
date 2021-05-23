"""Utilisation functions."""
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


# TODO: Only used in main?
def get_image(im_path: Path) -> Tuple[Optional[Any], Tuple[int, int]]:
    """Load in the image stored under the given path, together with center pixel."""
    try:
        img = Image.open(im_path).convert("RGB")
        shape = np.asarray(img).shape
        pixel = shape[0] // 2, shape[1] // 2
        return img, pixel
    except UnidentifiedImageError:
        return None, (0, 0)
