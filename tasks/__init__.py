"""Package tasks."""

from invoke import Collection

from . import conda
from .logging import configure_root_logger
from .tasks import docs, lab, lint

configure_root_logger()

ns = Collection()
ns.add_task(docs)
ns.add_task(lab)
ns.add_task(lint)
ns.add_collection(conda)
