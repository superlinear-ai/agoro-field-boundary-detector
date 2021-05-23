"""Methods used to inspect the data and models."""
from pathlib import Path

from PIL import Image

# Create color palette (64 options)
palette = [0, 0, 0]  # Background
for r in range(100, 251, 50):
    for g in range(100, 251, 50):
        for b in range(100, 251, 50):
            palette += [r, g, b]


def color_mask(path: Path) -> Image:
    """Show the mask specified under the given path."""
    mask = Image.open(path)
    mask.putpalette(palette)
    return mask
