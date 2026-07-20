import os
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


try:
    import ccxt  # noqa: F401
except ImportError:
    ccxt = types.ModuleType("ccxt")
    ccxt.NetworkError = type("NetworkError", (Exception,), {})
    ccxt.RequestTimeout = type("RequestTimeout", (Exception,), {})
    sys.modules["ccxt"] = ccxt


try:
    import matplotlib  # noqa: F401
except ImportError:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot


@pytest.fixture(scope="session", autouse=True)
def keep_qapplication_alive():
    """Use one QApplication for all Qt tests to avoid backend teardown crashes."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
