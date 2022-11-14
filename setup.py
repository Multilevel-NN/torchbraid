from setuptools import setup
from setuptools import Extension

from Cython.Build import cythonize,build_ext

import numpy
import os
import mpi4py
import subprocess

def git(*args):
    return subprocess.check_call(['git'] + list(args))

braid_dir = './xbraid/braid'
if "EXTRA_FLAGS" in os.environ.keys():
  extra_compile_args = os.environ["EXTRA_FLAGS"]
else:
  extra_compile_args = []

if "CC" not in os.environ.keys():
  os.environ["CC"] = mpi4py.get_config()['mpicc']

ext_modules = [Extension(
                name='torchbraid.torchbraid_app',
                sources=["torchbraid/*.pyx"],
                libraries=["braid"],
                library_dirs=[braid_dir],
                include_dirs=[braid_dir,numpy.get_include()],
                extra_compile_args=extra_compile_args)]

if not os.path.isdir('./xbraid'):
  print('cloning xbraid...')
  git('clone','https://github.com/XBraid/xbraid.git')

  print('buildign xbraid...')
  subprocess.check_call(['make','debug=no','braid'],cwd='./xbraid')

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext}
)
