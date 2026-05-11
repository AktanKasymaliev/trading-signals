"""Indicators package.

Originally `pandas-ta==0.3.14b` was specified but it has been pulled from PyPI for
Python 3.11 (latest 0.4.x requires Python 3.12+). We use the maintained fork
`pandas-ta-classic` which uses `numpy.nan` correctly — no monkey-patch needed.

The legacy numpy.NaN guard is kept defensively for environments that might still
import the original pandas-ta.
"""

import numpy as np

if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]

# Expose pandas_ta_classic as `pandas_ta` to keep call sites pep8-clean.
import pandas_ta_classic as pandas_ta  # noqa: E402, F401
