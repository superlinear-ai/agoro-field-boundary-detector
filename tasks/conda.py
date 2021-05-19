"""Conda tasks."""

import logging

from invoke import task

logger = logging.getLogger(__name__)


@task
def create(c):
    """Recreate the conda environment."""
    logger.info("Recreating conda environment agoro-field-boundary-detector-env...")
    c.run("conda-merge environment.run.yml environment.dev.yml > environment.yml", pty=True)
    c.run("mamba env create --force", pty=True)
    c.run("rm environment.yml", pty=True)
    logger.info("Installing editable agoro_field_boundary_detector into conda environment...")
    c.run("pip install --editable .", pty=True)


@task
def update(c):
    """Update the conda environment."""
    logger.info("Updating conda environment agoro-field-boundary-detector-env...")
    c.run("conda-merge environment.run.yml environment.dev.yml > environment.yml", pty=True)
    c.run("mamba env update --prune", pty=True)
    c.run("rm environment.yml", pty=True)
    logger.info("Installing editable agoro_field_boundary_detector into conda environment...")
    c.run("pip install --editable .", pty=True)
