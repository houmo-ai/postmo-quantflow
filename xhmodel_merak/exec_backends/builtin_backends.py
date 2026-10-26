"""Explicit registration of lightweight built-in backend declarations.

Add a declaration module to _BUILTINS to enable another backend. Declaration
modules contain import paths only; they must not import SDK/runtime classes.
"""

from . import hmonnx_backend, tcim_backend


_BUILTINS = (hmonnx_backend, tcim_backend)
BACKENDS = {backend.NAME: backend.CLASSES for backend in _BUILTINS}
BACKEND_ALIASES = {alias: backend.NAME for backend in _BUILTINS for alias in backend.ALIASES}

BACKEND_POLICIES = {backend.NAME: backend for backend in _BUILTINS}
