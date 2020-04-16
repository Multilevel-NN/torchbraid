from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

module_name = 'torchbraid_test'

braid_dir = os.environ["XBRAID_ROOT"]
if "EXTRA_FLAGS" in os.environ.keys():
  extra_compile_args = os.environ["EXTRA_FLAGS"]
else:
  extra_compile_args = []

torchbraid_ext = Extension(
		name=module_name,
		sources=["%s.pyx" % module_name],
		libraries=["braid"],
		library_dirs=[braid_dir],
		include_dirs=[braid_dir,numpy.get_include()],
		extra_compile_args=extra_compile_args
)

setup(name=module_name,
      ext_modules=cythonize([torchbraid_ext],
                            annotate=True,
                            compiler_directives={'boundscheck': False}))
