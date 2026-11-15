"""
Run RQ1 with explicit input and output paths passed through the command line.
"""
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).with_name('rq1_standard_di_vulnerability.py')), *sys.argv[1:]], check=True)
