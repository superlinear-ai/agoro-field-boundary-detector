"""Setup module for this Python package."""
import pathlib

from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

INSTALL_REQUIRES = [
    "tqdm",
    "torch~=1.8.1",
    "torchvision~=0.9.1",
    "pycocotools~=2.0.2",
    "earthengine-api~=0.1.267",
    "opencv-python~=4.5.2.52",
]

setup(
    name="agoro_field_boundary_detector",
    version="0.1.1",
    description="Detect field boundaries using satellite imagery.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/radix-ai/agoro-field-boundary-detector",
    author="Radix",
    author_email="developers@radix.ai",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=("data", "models", "notebooks", "tasks")),
    license="LICENSE",
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
)
