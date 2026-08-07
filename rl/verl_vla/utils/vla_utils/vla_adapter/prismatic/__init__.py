"""Vendored Prismatic package used by the VLA-Adapter backend."""

import sys


# Upstream Prismatic uses absolute imports internally.
sys.modules.setdefault("prismatic", sys.modules[__name__])

from .models import available_model_names, available_models, get_model_description, load
