"""Utilisation functions to generate the data."""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


def from_vgg(path: Path) -> List[Dict[str, Any]]:
    """Transform the VGG output format (json) to custom output format; list of annotated samples."""
    with open(path, "r") as f:
        temp = json.load(f)

    # Filter out that do not have a region
    annot = []
    for v in temp.values():
        if not v["regions"]:
            continue
        item = {
            "field_id": int(Path(v["filename"]).with_suffix("").name),
            "regions": [],
        }
        for r in v["regions"]:
            x = r["shape_attributes"]["all_points_x"]
            y = r["shape_attributes"]["all_points_y"]
            item["regions"].append(list(zip(x, y)))  # type: ignore
        annot.append(item)
    return annot


def create_aggregated_mask(
    polygon_list: List[List[Tuple[int, int]]],
    width: int = 256,
    height: int = 256,
) -> np.ndarray:
    """Create an aggregated mask over all the polygons."""
    field = np.zeros((width, height))
    for i, polygon in enumerate(reversed(polygon_list)):  # Latest annot likely worse annot
        mask = create_mask(polygon, width=width, height=height)
        field += (i + 1) * mask
        field = np.clip(field, a_min=0, a_max=(i + 1))  # type: ignore
    return field


def create_mask(
    polygon: List[Tuple[int, int]],
    width: int = 256,
    height: int = 256,
) -> np.ndarray:
    """Create a masking image using the provided polygon."""
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    return np.array(img)  # Binary mask (0=background, 1=field)


def mask_to_polygons(
    mask: np.ndarray,
) -> List[List[Tuple[int, int]]]:
    """Transform a mask back to a polygon boundary."""
    polygons = []
    for v in sorted(
        set(np.unique(mask))
        - {
            0,
        }
    ):
        # Extract polygon
        contours, _ = cv2.findContours(
            np.asarray(mask == v, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        polygon = []
        for point in sorted(contours, key=lambda x: -len(x))[
            0
        ]:  # If multiple (shouldn't be); use largest
            polygon.append((int(point[0][0]), int(point[0][1])))

        # Trim redundant points
        i = 1
        while i < len(polygon) - 1:
            if _is_line(polygon[i - 1 : i + 2]):
                del polygon[i]
            else:
                i += 1
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        polygons.append(polygon)
    return polygons


def _is_line(data: Any) -> bool:
    """Check if straight line."""
    x, y = zip(*data)
    assert len(x) == 3
    for i in (-1, 0, 1):
        if x[0] - i == x[1] == x[2] + i:
            return True
        if y[0] - i == y[1] == y[2] + i:
            return True
    return False
