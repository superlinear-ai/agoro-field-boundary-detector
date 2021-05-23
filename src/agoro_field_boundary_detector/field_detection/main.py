"""Create a field-polygon for a given image."""
from pathlib import Path
from typing import Any, Optional

import torch

from src.agoro_field_boundary_detector.data import mask_to_polygons
from src.agoro_field_boundary_detector.field_detection.mask_rcnn.transforms import get_transform
from src.agoro_field_boundary_detector.field_detection.model import predict_mask
from src.agoro_field_boundary_detector.field_detection.utils import get_image


# TODO: Add to model.py
def load_model(
    model_path: Path = Path(__file__).parent / "../../models",
) -> Any:
    """Load in the field-detector Mask-RCNN model."""
    # Load in the model
    model = torch.load(model_path / "mask_rcnn", map_location=torch.device("cpu"))  # type: ignore

    # Move model to the right device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore
    print(f"Predicting field boundaries on device {device}")
    model.to(device)
    return model


def predict_im_polygon(
    model: Any,
    im_path: Path,
) -> Optional[Any]:
    """
    Predict the polygon of the mask covering the center pixel.

    :param model: Mask-RCNN model used to predict field boundaries
    :param im_path: Path indicating which field to check
    """
    # Predict masks for the images
    transform = get_transform()

    img, pixel = get_image(im_path)
    if not img:
        return None

    # Make masking predictions
    img, _ = transform(img, None)
    mask = predict_mask(
        model=model,
        img=img,
    )

    # Extract polygon corresponding masked pixel, if exists
    m_value = mask[pixel[0], pixel[1]]
    if m_value == 0:
        return None
    else:
        mask[mask != m_value] = 0
        mask[mask == m_value] = 1
        return mask_to_polygons(mask)
