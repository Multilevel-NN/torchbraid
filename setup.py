import os
import subprocess

import mpi4py
import numpy
from Cython.Build import cythonize, build_ext
from setuptools import setup, Extension, find_packages

# MPICC is needed to compile xbraid; environ is not persistent and will revert to original choice
os.environ["CC"] = mpi4py.get_config()['mpicc']

braid_dir = './src/xbraid/braid'


class myMake(build_ext):

  def run(self):
    self.pre_build()
    super().run()

  def pre_build(self):
    print(os.path.exists(os.path.join(os.getcwd(), 'src', 'xbraid')))
    if not os.path.exists(os.path.join(os.getcwd(), 'src', 'xbraid')):
      print('cloning xbraid...')
      subprocess.check_call(['git', 'clone', 'https://github.com/XBraid/xbraid.git'], cwd='./src')

    print('building xbraid...')
    subprocess.check_call(['make', 'debug=no', 'braid', 'MPICC=cc'], cwd='./src/xbraid')


braid_sources = ['access.c', 'adjoint.c', 'base.c', 'braid.c', 'braid_status.c',
                 'braid_test.c', 'communication.c', 'distribution.c', 'drive.c',
                 'grid.c', 'hierarchy.c', 'interp.c', 'mpistubs.c', 'norm.c', 'refine.c',
                 'relax.c', 'residual.c', 'restrict.c', 'space.c', 'step.c', 'tape.c',
                 'util.c', 'uvector.c']
braid_sources = ['src/xbraid/braid/' + item for item in braid_sources]

extension = [Extension(
  name="torchbraid.torchbraid_app",
  sources=["src/torchbraid/torchbraid_app.pyx"] + braid_sources,
  libraries=["braid"],
  library_dirs=[braid_dir],
  include_dirs=[braid_dir, numpy.get_include()],
),
  Extension(
    name="torchbraid.test_fixtures.test_cbs",
    sources=["src/torchbraid/test_fixtures/test_cbs.pyx"],
    libraries=["braid"],
    library_dirs=[braid_dir],
    include_dirs=[braid_dir, numpy.get_include()],
  )
]

install_requires = [
  'setuptools',
  'mpi4py',
  'cython==0.29.32',
  'numpy',
  'torch==2.0.1',
  'torchvision==0.15.2',
  'matplotlib'
]

setup(
  ext_modules=cythonize(extension, language_level="3"),
  install_requires=install_requires,
  packages=find_packages(where="src"),
  package_dir={"": "src"},
  package_data={"torchbraid": ["*.pyx", "*.pxd"],
                "torchbraid.test_fixtures": ["*.pyx"],
                "xbraid.braid": ["*.h"]},
  cmdclass={'build_ext': myMake}
)
