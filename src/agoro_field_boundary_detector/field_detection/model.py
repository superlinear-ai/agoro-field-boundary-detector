"""Methods related to the Mask-RCNN model."""
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F_vis

from src.agoro_field_boundary_detector.data import mask_to_polygons
from src.agoro_field_boundary_detector.field_detection.dataset import Dataset
from src.agoro_field_boundary_detector.field_detection.mask_rcnn.engine import (
    evaluate,
    train_one_epoch,
)


class FieldBoundaryDetector:
    """Field Boundary Detector model implemented in PyTorch."""

    def __init__(
        self,
        model_path: Path,
        n_classes: int = 2,
        n_hidden: int = 256,
    ):
        """
        Initialise the Field Boundary Detector model, either by loading in previous version or creating a new one.

        :param model_path: Path under which the model is saved or will be saved
        :param n_classes: Number of the classes (outputs) the model should have
        :param n_hidden: Number of hidden layers used in the MaskRCNN predictor
        """
        self.path = model_path
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.model: Optional[torchvision.models.detection.maskrcnn_resnet50_fpn] = None
        if not self.load():
            self.create_instance_segmentation_model()

    def __call__(
        self,
        im: np.ndarray,
        thr: float = 0.8,
    ) -> List[Tuple[int, int]]:
        """
        Predict the polygon of the mask covering the image's center pixel.

        # TODO: Update

        :param model: Mask-RCNN model used to predict field boundaries
        :param im_path: Path indicating which field to check
        """
        # Get the center of the image
        center = im.shape[0] // 2, im.shape[1] // 2

        # Make masking predictions
        mask = self.get_mask(
            im=im,
        )

        # Extract polygon corresponding masked pixel, if exists
        m_value = mask[center[0], center[1]]
        if m_value == 0:
            return []
        else:
            mask[mask != m_value] = 0
            mask[mask == m_value] = 1
            return mask_to_polygons(mask)[0]  # type: ignore

    def __str__(self) -> str:
        """Textual model representation."""
        return f"FieldBoundaryDetector(n_hid={self.n_hidden}, n_out={self.n_classes})"

    def __repr__(self) -> str:
        """Textual model representation."""
        return str(self)

    def create_instance_segmentation_model(self) -> None:
        """Create an instance segmentation model."""
        # Load an instance segmentation model pre-trained on COCO
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # Replace the pre-trained head (box and mask predictor)
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            self.model.roi_heads.box_predictor.cls_score.in_features,
            self.n_classes,
        )
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            self.model.roi_heads.mask_predictor.conv5_mask.in_channels,
            self.n_hidden,
            self.n_classes,
        )

    def train(
        self,
        dataset: Dataset,
        n_epoch: int = 10,
        batch_size: int = 1,
        n_workers: int = 0,
        val_frac: float = 0.1,
        early_stop: bool = True,
    ) -> None:
        """
        Train the model.

        :param dataset: Dataset to train ane evaluate the model on
        :param n_epoch: Number of epochs to train over
        :param batch_size: Training and validation batch size
        :param n_workers: Number of dataloader workers (0 for sequential training)
        :param val_frac: Fraction of dataset to use for validation
        :param early_stop: Whether or not to stop training when validation F1 score starts to decrease
        """
        # Split the dataset in train and test set
        frac_idx = round(val_frac * len(dataset))
        indices = torch.randperm(len(dataset)).tolist()  # type: ignore
        dataset_val = torch.utils.data.Subset(dataset, indices[:frac_idx])
        dataset_train = torch.utils.data.Subset(dataset, indices[frac_idx:])

        # Define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        # Move model to the right device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore
        self.model.to(device)  # type: ignore

        # Construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]  # type: ignore
        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005,
        )

        # Learning rate scheduler which decreases the learning rate by 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1,
        )

        # Train the model
        prev = 0
        for epoch in range(n_epoch):
            # Train for one epoch and update the learning rate afterwards
            train_one_epoch(
                self.model,
                optimizer,
                data_loader,
                device,
                epoch,
                print_freq=len(data_loader) // 10,
            )
            lr_scheduler.step()

            # Evaluate on the validation dataset
            f1 = evaluate(
                self.model,
                data_loader_val,
                device=device,
            )

            # Stop if validation F1 starts to decrease
            if early_stop and prev > f1:
                break
            prev = f1
        self.save()

    def test(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        n_workers: int = 0,
        n_show: int = 0,
        write_path: Optional[Path] = None,
    ) -> None:
        """
        Evaluate the model.

        :param dataset: Complete dataset (same split used as seen during training)
        :param batch_size: Testing batch size
        :param n_workers: Number of dataloader workers (0 for sequential training)
        :param n_show: Number of test-images for which to show the masked fields
        :param write_path: Path to write evaluation-images to (only if n_show > 0)
        """
        if n_show:
            assert write_path is not None

        # Move model to the right device and ensure evaluation mode
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore
        self.model.to(device)  # type: ignore
        self.model.eval()  # type: ignore

        # Extract evaluation-portion of model
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        # Evaluate the model, print out the resulting F1-score
        f1 = evaluate(
            self.model,
            data_loader,
            device=device,
        )
        print(f"F1: {f1}")

        # Eye-ball evaluation
        print("\nEye-ball evaluation:")
        time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        for idx in range(min(len(dataset), n_show)):
            img, _ = dataset[idx]
            mask = self.get_mask(im=img)
            polygons = mask_to_polygons(mask)
            img_tc = img.mul(255).permute(1, 2, 0).byte().numpy()
            plt.figure(figsize=(5, 5))
            plt.imshow(img_tc, interpolation="nearest")
            for polygon in polygons:
                x, y = zip(*polygon)
                plt.plot(x, y)
            plt.savefig(write_path / f"{time}_eval_{idx}.png")  # type: ignore
            plt.close()

    def get_mask(
        self,
        im: np.ndarray,
        thr: float = 0.8,
        size_thr: float = 0.003,
        overlap_thr: float = 0.8,
    ) -> np.ndarray:
        """
        Predict a layered mask for the given image.

        :param im: Image to make the prediction on
        :param thr: Threshold value to round mask to binary
        :param size_thr: Proportion of the image a mask must occupy in order to be valid
        :param overlap_thr: Proportion of the image that does not overlap in order to be valid
        :return: Image masks
        """
        # Transform image to PyTorch tensor and put on right device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore
        im_t = F_vis.to_tensor(im).to(device)
        with torch.no_grad():
            # Predict all masks
            prediction = self.model([im_t])[0]  # type: ignore

            # Merge the masks together
            i = 1
            result = np.zeros(im_t.shape[1:], dtype=np.int8)
            for mask in prediction["masks"]:
                mask = mask[0].cpu().numpy()
                mask[mask >= thr] = 1  # Ensure binary mask
                mask[mask < thr] = 0
                mask = mask.astype(np.int8)

                # Ignore masks if they occupy less than 0.5% of the image
                if mask.sum() / len(mask.flatten()) < size_thr:
                    continue

                # Ignore masks that mainly (50% or more) overlap with previously detected masks
                previous = np.clip(result, a_min=0, a_max=1)
                if (mask * previous).sum() > (1 - overlap_thr) * mask.sum():
                    continue

                # Add mask to the result
                result += i * mask
                result = np.clip(result, a_min=0, a_max=i)  # type: ignore
                i += 1
        return result

    def load(self) -> bool:
        """Load a previously saved model."""
        if self.path.is_file():
            self.model = torch.load(self.path, map_location=torch.device("cpu"))  # type: ignore
            return True
        return False

    def save(self) -> None:
        """Save the model."""
        torch.save(
            self.model,
            self.path,
        )
