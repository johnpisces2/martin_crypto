import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import ccxt  # noqa: F401
except ImportError:
    ccxt = types.ModuleType("ccxt")
    ccxt.NetworkError = type("NetworkError", (Exception,), {})
    ccxt.RequestTimeout = type("RequestTimeout", (Exception,), {})
    sys.modules["ccxt"] = ccxt


try:
    import matplotlib.pyplot  # noqa: F401
except ImportError:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot
