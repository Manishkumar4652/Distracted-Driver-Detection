"""
Streamlit demo inside demo_on_image directory.
Executes the main project application.
"""
import os
import sys

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app

# Run app
if __name__ == '__main__':
    pass
