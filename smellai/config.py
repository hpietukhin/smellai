from pathlib import Path

# Project root is 2 levels up from this file (smellai/config.py -> smellai/ -> root/)
# Wait, if this file is smellai/config.py, then:
# smellai/config.py -> parent is smellai/ -> parent is root/
# So parents[1] is root.
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
RMINER_DATA_DIR = PROJECT_ROOT / "rminer_data"
