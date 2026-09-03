from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


_ENVIRONMENT_LOCK = threading.RLock()


@contextmanager
def scoped_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Serialize process-global environment changes across build and inference."""

    with _ENVIRONMENT_LOCK:
        previous = {name: os.environ.get(name) for name in values}
        os.environ.update(values)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


__all__ = ["scoped_environment"]
