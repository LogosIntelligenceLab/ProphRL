"""Vendored OpenPI package used by the Pi05 backend."""

import sys


# OpenPI uses absolute imports internally. Register the vendored package under
# its upstream name before importing any submodule.
sys.modules.setdefault("openpi", sys.modules[__name__])
