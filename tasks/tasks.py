"""Main tasks."""

import logging
import os

from invoke import task

logger = logging.getLogger(__name__)
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@task
def lint(c):
    """Lint this package."""
    logger.info("Running pre-commit checks...")
    c.run("pre-commit run --all-files --color always", pty=True)
    c.run("safety check --full-report", warn=True, pty=True)


@task
def test(c):
    """Test this package."""
    logger.info("Running Pytest...")
    c.run(
        "env PYTHONPATH=src pytest --cov=src --cov-report term-missing --cov-report html:coverage/ tests",
        pty=True,
    )  # PYTHONPATH=src is needed for CI


@task
def lab(c):
    """Run Jupyter Lab."""
    notebooks_path = os.path.join(REPO_PATH, "notebooks")
    os.makedirs(notebooks_path, exist_ok=True)
    with c.cd(notebooks_path):
        c.run("jupyter lab --allow-root", pty=True)


@task
def docs(c, browser=False, output_dir="site"):
    """Generate this package's docs."""
    if browser:
        c.run("portray in_browser", pty=True)
    else:
        c.run(f"portray as_html --output_dir {output_dir} --overwrite", pty=True)
        logger.info("Package documentation available at ./site/index.html")


@task
def bump(c, part, dry_run=False):
    """Bump the major, minor, patch, or post-release part of this package's version."""
    c.run(f"bump2version {'--dry-run --verbose ' + part if dry_run else part}", pty=True)
