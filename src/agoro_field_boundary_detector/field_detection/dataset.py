"""Dataset class."""
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import functional as F_vis


# TODO: Add to data/ folder
class Dataset(torch.utils.data.Dataset):  # type: ignore
    """Dataset used to train the Mask RCNN model."""

    def __init__(
        self,
        path: Path,
    ) -> None:
        """Initialise the dataset."""
        self.path = path
        self.fields_path = self.path / "fields"
        self.masks_path = self.path / "masks"
        self.tags = [Path(x).with_suffix("").name for x in glob(str(self.fields_path / "*"))]

    def __getitem__(self, idx: int) -> Any:
        """Get the item at the given index from the dataset."""
        # load image and mask under specified index
        with open(self.fields_path / f"{self.tags[idx]}.npy", "rb") as f:
            field = np.load(f)
        with open(self.masks_path / f"{self.tags[idx]}.npy", "rb") as f:
            mask = np.load(f)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # type: ignore
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)  # type: ignore
        masks = torch.as_tensor(masks, dtype=torch.uint8)  # type: ignore

        image_id = torch.tensor([idx])  # type: ignore
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # type: ignore
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # type: ignore

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }
        return F_vis.to_tensor(field), target

    def __len__(self) -> int:
        """Get the size of the dataset."""
        return len(self.tags)
