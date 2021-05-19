"""Package tasks."""

from invoke import Collection

from . import conda
from .logging import configure_root_logger
from .tasks import bump, docs, lab, lint, test

configure_root_logger()

ns = Collection()
ns.add_task(bump)
ns.add_task(docs)
ns.add_task(lab)
ns.add_task(lint)
ns.add_task(test)
ns.add_collection(conda)
