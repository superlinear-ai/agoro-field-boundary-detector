"""Create, train and evaluate the Field Boundary Detector model."""
import os
from pathlib import Path
from typing import Optional

from src.agoro_field_boundary_detector.field_detection.dataset import Dataset
from src.agoro_field_boundary_detector.field_detection.model import FieldBoundaryDetector

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def train(
    model: FieldBoundaryDetector,
    n_epoch: int = 10,
    batch_size: int = 1,
    n_workers: int = 0,
    val_frac: float = 0.1,
    early_stop: bool = True,
    patience: int = 3,
) -> None:
    """Train the model."""
    dataset = Dataset(path=Path(__file__).parent / "../../data/augmented")
    model.train(
        dataset=dataset,
        n_epoch=n_epoch,
        batch_size=batch_size,
        n_workers=n_workers,
        val_frac=val_frac,
        early_stop=early_stop,
        patience=patience,
    )


def evaluate(
    model: FieldBoundaryDetector,
    batch_size: int = 1,
    n_workers: int = 0,
    n_show: int = 5,
    write_path: Optional[Path] = None,
) -> None:
    """Evaluate the model."""
    if write_path is None:
        write_path = Path.cwd()
    dataset = Dataset(path=Path(__file__).parent / "../../data/test")
    model.test(
        dataset=dataset,
        batch_size=batch_size,
        n_workers=n_workers,
        n_show=n_show,
        write_path=write_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", default=Path(__file__).parent / "../../models/mask_rcnn", type=str
    )
    parser.add_argument("--train", default=0, type=int)
    parser.add_argument("--test", default=1, type=int)
    args = parser.parse_args()

    # Load in the model
    field_detector = FieldBoundaryDetector(model_path=args.model_path)

    # Train, if requested
    if args.train:
        train(
            model=field_detector,
        )
    # Test, if requested
    if args.test:
        evaluate(
            model=field_detector,
        )
