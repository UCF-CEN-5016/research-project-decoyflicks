import sys
from importlib import util

try:
    module = util.find_spec("commons")
except ImportError as e:
    print(f"ModuleNotFoundError: {e}")
    sys.exit(1)