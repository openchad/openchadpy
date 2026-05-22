import sys
import os
# Set up the path to include the 'src' directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "src"))
# Import the main function from the package
from openchadpy.main import main
if __name__ == "__main__":
    # Start the integrated application (FastAPI + Tauri)
    sys.exit(main())
