"""Methods related to the Mask-RCNN model."""
import json
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import src.radix_co2_reduction.field_detection.mask_rcnn.utils as utils
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError
from src.radix_co2_reduction.field_detection.data import mask_to_polygons
from src.radix_co2_reduction.field_detection.dataset import Dataset
from src.radix_co2_reduction.field_detection.mask_rcnn.engine import evaluate, train_one_epoch
from src.radix_co2_reduction.field_detection.mask_rcnn.transforms import get_transform
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

# Randomisation seed
SEED = 42


def get_instance_segmentation_model(num_classes: int) -> Any:
    """Create an instance segmentation model."""
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True,
    )

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )
    return model


def train_mdl(
    model: Any,
    dataset: Any,
    epochs: int = 10,
    early_stop: bool = True,
) -> Any:
    """
    Train the model.

    :param model: Model to train
    :param dataset: Dataset to train ane evaluate the model on
    :param epochs: Number of epochs to train over
    :param early_stop: Whether or not to stop training when validation F1 score starts to decrease
    """
    # split the dataset in train and test set
    frac = round(0.9 * (len(dataset) - 1))
    torch.manual_seed(SEED)
    indices = torch.randperm(len(dataset)).tolist()  # type: ignore
    dataset_train = torch.utils.data.Subset(dataset, indices[:frac])
    print(f"Training: {len(dataset_train)} samples")
    dataset_test = torch.utils.data.Subset(dataset, indices[frac:])
    print(f" Testing: {len(dataset_test)} samples")

    # Define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    # Move model to the right device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore
    print(f"Training on device {device}")
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
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

    # let's train
    print()
    prev = 0
    for epoch in range(epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        f1 = evaluate(model, data_loader_test, device=device)
        print(f"F1: {f1}")

        # Stop if validation F1 starts to decrease
        if early_stop and prev > f1:
            break
        prev = f1

    # Return the trained model
    model.to("cpu")  # Ensure it is on CPU
    return model


def predict_mask(
    model: Any,
    img: Any,
    device: Optional[torch.device] = None,  # type: ignore
    thr: float = 0.8,
    size_thr: float = 0.003,
    overlap_thr: float = 0.8,
) -> np.ndarray:
    """
    Predict a layered mask for the given image using the model.

    :param model: Model used to make the prediction
    :param img: Image to make the prediction on
    :param device: Device on which to run the model on
    :param thr: Threshold value to round mask to binary
    :param size_thr: Proportion of the image a mask must occupy in order to be valid
    :param overlap_thr: Proportion of the image that does not overlap in order to be valid
    """
    if not device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore
    with torch.no_grad():
        # Predict all masks
        prediction = model([img.to(device)])

        # Merge the masks together
        i = 1
        result = np.zeros(img.shape[1:], dtype=np.int8)
        for mask in prediction[0]["masks"]:
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


def eval_mdl(
    model: Any,
    dataset: Dataset,
    write_path: Path,
    n_eye: int = 10,
) -> None:
    """
    Evaluate the model.

    :param model: The model to evaluate
    :param dataset: Complete dataset (same split used as seen during training)
    :param write_path: Path to write evaluation-images to
    :param n_eye: Number of samples to eye-ball
    """
    # Move model to the right device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore
    print(f"Evaluating on device {device}")
    model.to(device)

    # Extract evaluation-portion of model
    frac = round(0.9 * (len(dataset) - 1))
    torch.manual_seed(SEED)
    indices = torch.randperm(len(dataset)).tolist()  # type: ignore
    dataset_test = torch.utils.data.Subset(dataset, indices[frac:])
    print(f" Testing: {len(dataset_test)} samples")
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    # Ensure model is in evaluation mode
    model.eval()

    # evaluate on the test dataset
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore
    print(f"Training on device {device}")
    print("\nBuilt-in evaluation:")
    f1 = evaluate(model, data_loader_test, device=device)
    print(f"F1: {f1}")

    # Eye-ball evaluation
    print("\nEye-ball evaluation:")
    time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    for idx in range(min(len(dataset_test), n_eye)):
        img, _ = dataset_test[idx]
        mask = predict_mask(
            model=model,
            img=img,
            device=device,
        )
        polygons = mask_to_polygons(mask)
        img_tc = img.mul(255).permute(1, 2, 0).byte().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(img_tc, interpolation="nearest")
        for polygon in polygons:
            x, y = zip(*polygon)
            plt.plot(x, y)
        plt.savefig(write_path / f"{time}_eval_{idx}.png")
        plt.close()


def infer_mdl(
    model: Any,
    path: Path,
) -> None:
    """
    Predict the polygon of the mask covering the specified pixel.

    :param model: Model used during inference
    :param path: Path where the raw field-images are stored
    """
    # Move model to the right device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore
    print(f"Evaluating on device {device}")
    model.to(device)

    # Load in all the images
    images = glob(str(path / "*/true_color.png"))

    # Predict masks for the images
    results: Dict[str, Optional[Any]] = {}
    transform = get_transform()
    for im_path in tqdm(sorted(images)):
        field_id = Path(im_path).parent.name

        # Try to import image
        try:
            img = Image.open(im_path).convert("RGB")
        except UnidentifiedImageError:
            results[field_id] = None
            continue

        # Extract center of image
        shape = np.asarray(img).shape
        pixel = shape[0] // 2, shape[1] // 2

        # Make masking predictions
        img, _ = transform(img, None)
        mask = predict_mask(
            model=model,
            img=img,
            device=device,
        )

        # Extract polygon corresponding masked pixel, if exists
        m_value = mask[pixel[0], pixel[1]]
        if m_value == 0:
            results[field_id] = None
        else:
            mask[mask != m_value] = 0
            mask[mask == m_value] = 1
            polygons = mask_to_polygons(mask)
            results[field_id] = polygons

    # Write the results
    with open(path / "polygons.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mdl",
        # default='',
        default=Path.home() / "data/agoro/models/mask_rcnn_2021_04_26_17_19",
        help="Path to previous model, create new one if empty",
    )
    parser.add_argument(
        "--n_classes",
        default=2,
        help="Number of classes (defaults to 2: background and field)",
    )
    parser.add_argument(
        "--n_epochs",
        default=20,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--train",
        default=0,
        help="Whether or not to train the model",
    )
    parser.add_argument(
        "--eval",
        default=0,
        help="Whether or not to evaluate the model",
    )
    parser.add_argument(
        "--infer",
        default=1,
        help="Whether or not to evaluate the model",
    )
    args = parser.parse_args()

    # Create a model
    if args.mdl:
        mdl = torch.load(args.mdl, map_location=torch.device("cpu"))  # type: ignore
        print("Loaded previously trained model")
    else:
        mdl = get_instance_segmentation_model(
            args.n_classes,
        )
        print("Created new model")

    # Load in the dataset
    data = Dataset(path=Path.home() / "data/agoro/field_annotation/generated")
    print(f"Loaded in {len(data)} data samples")

    # Train the model
    if args.train:
        mdl = train_mdl(
            model=mdl,
            dataset=data,
            epochs=args.n_epochs,
        )
        args.mdl = (
            Path.home() / f'data/agoro/models/mask_rcnn_{datetime.now().strftime("%Y_%m_%d_%H_%M")}'
        )
        torch.save(mdl, args.mdl)

    # Evaluate the model
    if args.eval:
        eval_mdl(
            model=mdl, dataset=data, write_path=Path.home() / "data/agoro/field_annotation/eval"
        )

    # Predict polygons for the fields
    if args.infer:
        infer_mdl(
            model=mdl,
            path=Path.home() / "data/agoro/fields_raw",
        )
