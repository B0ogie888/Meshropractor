"""Double-click launcher: select the project environment and run the shared entry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from desktop_launcher import launch

if __name__ == '__main__':
    launch()
