"""Guard the suite against the CPython-3.14 / NumPy-2.2.x temporary-elision defect.

On CPython 3.14.0 with NumPy < 2.3, the interpreter's ``LOAD_FAST_BORROW`` opcode
loads function locals without incrementing their refcount, so NumPy's
temporary-elision optimisation sees a live local array as a dead temporary and
writes the result of ``a + b`` / ``a * b`` **into** ``a``. Measured consequences in
this repository before it was understood:

* ``deformation._horn_schunck_level`` corrupted its own gradient and returned
  all-NaN flow (15 test failures);
* ``nonrigid_gate.measure_localisation`` built its permutation null from
  fourth-power values and flipped a gate verdict from RUN_NONRIGID to
  ACCEPT_RIGID;
* ``simulated + noise`` inside a *test function* returned ``simulated`` itself,
  so the test asserted on a zero residual.

The last one is the reason for this conftest: the defect corrupts arbitrary
array arithmetic in **test code**, where no amount of library hardening helps. A
corrupted test can also silently *pass* while checking nothing, which is worse
than failing. So: probe the defect once, and if it fires, skip the whole suite
with an explanation instead of reporting nonsense.

Known-good interpreters: CPython <= 3.13 (any NumPy), or NumPy >= 2.3 (where the
elision/borrow interaction is fixed).
"""
from __future__ import annotations

import numpy as np
import pytest


def _elision_corrupts_live_locals() -> bool:
    """The minimal reproducer found in this project (strided view * fresh local)."""
    big = np.full((1488, 1816), 3.0, dtype=np.float32)
    p = big[::4, ::4]
    a = np.ones((372, 454), dtype=np.float32)
    _t = p * a  # noqa: F841  -- may be elided INTO ``a`` on defective interpreters
    return bool(a.max() != 1.0)


_DEFECTIVE = _elision_corrupts_live_locals()


def pytest_collection_modifyitems(config, items):
    if not _DEFECTIVE:
        return
    reason = (
        "NumPy temporary-elision corrupts live function locals on this "
        f"interpreter (CPython + NumPy {np.__version__}): 'a + b' inside a "
        "function writes into 'a'. Test arithmetic itself is unreliable -- "
        "results would be meaningless in BOTH directions. Run the suite under "
        "CPython <= 3.13 or NumPy >= 2.3 "
        "(e.g. PYTHONNOUSERSITE=1 <env-with-numpy-1.26-or-2.3>/bin/python -m pytest)."
    )
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        item.add_marker(marker)
