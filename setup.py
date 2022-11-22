import os
import subprocess

import mpi4py
import numpy
from Cython.Build import cythonize, build_ext
from setuptools import setup, Extension

if "CC" not in os.environ.keys():
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
            subprocess.check_call(['git', 'clone','https://github.com/XBraid/xbraid.git'], cwd='./src')

        print('building xbraid...')
        subprocess.check_call(['make', 'debug=no', 'braid'], cwd='./src/xbraid')


braid_sources = ['access.c', 'adjoint.c', 'base.c', 'braid.c', 'braid_status.c',
                 'braid_test.c', 'communication.c', 'distribution.c', 'drive.c',
                 'grid.c', 'hierarchy.c', 'interp.c', 'mpistubs.c', 'norm.c', 'refine.c',
                 'relax.c', 'residual.c', 'restrict.c', 'space.c', 'step.c', 'tape.c',
                 'util.c', 'uvector.c']
braid_sources = ['src/xbraid/braid/' + item for item in braid_sources]

extension = [Extension(
    name="torchbraid.torchbraid_app",
    sources=["src/torchbraid/torchbraid_app.pyx"]+braid_sources,
    libraries=["braid"],
    library_dirs=[braid_dir],
    include_dirs=[braid_dir, numpy.get_include()],
)]

setup(
    ext_modules=cythonize(extension),
    cmdclass={'build_ext': myMake}
)
