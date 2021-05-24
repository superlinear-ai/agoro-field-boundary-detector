"""Utilisation functions for data manipulation."""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_annotations(path: Path) -> Dict[str, List[List[Tuple[int, int]]]]:
    """Load in the annotations made by Label Studio."""
    with open(path, "r") as f:
        annotations = json.load(f)

    def _transform(values: List[Any]) -> List[List[Tuple[int, int]]]:
        """Transform the Label Studio syntax to image-coordinates."""
        boundaries = []
        for value in values:
            width, height = value["original_width"], value["original_height"]
            points = value["value"]["points"]
            points.append(points[0])  # Go full circle
            boundaries.append(
                [(round(width * a / 100), round(height * b / 100)) for a, b in points]
            )
        return boundaries

    # Load in the boundaries by field
    field_annotations = {}
    for annotation in annotations:
        name = annotation["file_upload"]
        if "_" in name:
            name = name.split("_")[0] + ".png"
        field_annotations[name] = _transform(annotation["annotations"][0]["result"])
    return field_annotations


def polygons_to_mask(
    polygons: List[List[Tuple[int, int]]],
    width: int = 1024,
    height: int = 1024,
) -> np.ndarray:
    """Create a masking image using the provided polygon."""
    mask = np.zeros((width, height))
    for idx, polygon in enumerate(reversed(polygons)):  # Latest annot likely worse annot
        img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask += (idx + 1) * np.array(img)  # Add binary mask (0=background, (idx+1)=field)
        mask = np.clip(mask, a_min=0, a_max=(idx + 1))  # type: ignore
    return mask


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
    for i in range(-2, 2 + 1):
        if x[0] - i == x[1] == x[2] + i:
            return True
        if y[0] - i == y[1] == y[2] + i:
            return True
    return False
