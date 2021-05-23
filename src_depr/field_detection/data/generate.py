"""Main class to generate training data."""
import re
from glob import glob
from pathlib import Path
from shutil import copy, rmtree

import numpy as np
from PIL import Image
from src.radix_co2_reduction.field_detection.data.transforms import transform
from src.radix_co2_reduction.field_detection.data.utils import create_aggregated_mask, from_vgg


def generate(read_path: Path, write_path: Path, clean: bool = True) -> None:
    """
    Generate data for the field detector by augmenting annotated images.

    :param read_path: Path to read annotations from
    :param write_path: Output folder
    :param clean: Clean previously generated data before creating new ones
    """
    # Clean folder
    if clean and write_path.is_dir():
        rmtree(write_path)

    # Load in the annotations
    annotations = from_vgg(read_path / "field_annotations.json")

    # Mask the annotations
    (write_path / "masks").mkdir(exist_ok=True, parents=True)
    for annot in annotations:
        # Load in the field
        field = np.asarray(Image.open(read_path / f"{annot['field_id']}.png"))
        width, height, _ = field.shape

        # Create mask given the regions
        mask = create_aggregated_mask(
            polygon_list=annot["regions"],
            width=width,
            height=height,
        )

        # Generate data transformations and save accordingly
        (write_path / f"{annot['field_id']}").mkdir(exist_ok=True, parents=True)
        transform(
            write_path=write_path / f"{annot['field_id']}",
            field=field,
            mask=mask,
        )

    # Put all generated files into single folder
    field_path = write_path / "fields"
    if field_path.is_dir():
        rmtree(field_path)
    field_path.mkdir(exist_ok=True, parents=True)
    mask_path = write_path / "masks"
    if mask_path.is_dir():
        rmtree(mask_path)
    mask_path.mkdir(exist_ok=True, parents=True)
    files = glob(str(write_path / "*/*.png"))
    for path in files:
        path2 = Path(path).with_suffix("")
        field_id = path2.parent.name
        if re.search("[^0-9]", field_id):
            continue
        field_name = path2.name
        if "_mask" in field_name:
            write_dir = mask_path
            field_name = field_name[:-5]
        else:
            write_dir = field_path
        copy(path, write_dir / f"{field_id}_{field_name}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--read_path",
        default=Path.home() / "data/agoro/field_annotation/annotated",
        help="Path to annotation dataset",
    )
    parser.add_argument(
        "--write_path",
        default=Path.home() / "data/agoro/field_annotation/generated",
        help="Path to write results to",
    )
    args = parser.parse_args()

    # Generate the training data
    generate(
        read_path=args.read_path,
        write_path=args.write_path,
    )
