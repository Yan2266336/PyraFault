"""Verifier protocol and implementations.

`pyarrow` is imported before anything that may pull in `kornia`: `kornia_rs` binds
the system libstdc++.so.6, which lacks GLIBCXX_3.4.31, and a later `import pyarrow`
then fails against the already-bound copy. Importing pyarrow first binds the conda
library, which satisfies both.
"""
import pyarrow as _pyarrow  # noqa: F401  (import order is load-bearing; see above)

from pyrafault.verify.base import DECISIONS, Localisation, Verifier, decide

__all__ = ["DECISIONS", "Localisation", "Verifier", "decide"]
