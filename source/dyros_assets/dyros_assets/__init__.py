"""Package containing Dyros Lab assets."""

import os
import toml

# Conveniences to other module directories via relative paths
DYROS_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

DYROS_ASSETS_DATA_DIR = os.path.join(DYROS_ASSETS_EXT_DIR, "data/usd")
"""Path to the extension data directory."""

DYROS_ASSETS_METADATA = toml.load(os.path.join(DYROS_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = DYROS_ASSETS_METADATA["package"]["version"]

from .robots import *
