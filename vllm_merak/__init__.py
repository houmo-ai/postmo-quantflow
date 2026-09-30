"""Resolve the in-repository project directory to its Python package.

The project and import package are both named ``vllm_merak``.  When Python is
started from the ModelZoo root, this outer directory is found before the
editable-install finder.  Point the package search path at the inner package
and execute its lightweight entry point so imports behave the same from every
working directory.
"""

from pathlib import Path


_INNER_PACKAGE = Path(__file__).resolve().parent / "vllm_merak"
_INNER_INIT = _INNER_PACKAGE / "__init__.py"
__path__ = [str(_INNER_PACKAGE)]
__file__ = str(_INNER_INIT)
exec(compile(_INNER_INIT.read_bytes(), str(_INNER_INIT), "exec"), globals(), globals())
