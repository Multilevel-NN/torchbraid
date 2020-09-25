import os
import subprocess

def git_rev():
  path = os.path.dirname(os.path.abspath(__file__))
  return subprocess.check_output(['git', 'rev-parse', 'HEAD'],cwd=path).decode('ascii').strip()
