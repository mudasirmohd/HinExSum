import os
import sys

# Ensure the repo root (containing the `hinexsum` package) is importable when
# pytest is invoked from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
