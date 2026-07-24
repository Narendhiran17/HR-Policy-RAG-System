import os
import sys

# Ensure the root directory is in the python path to resolve 'app.*' imports on Vercel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
