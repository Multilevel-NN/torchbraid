from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

module_name = 'torchbraid'

braid_dir = '/Users/eccyr/Packages/xbraid/braid'

torchbraid_ext = Extension(
    name=module_name,
    sources=["%s.pyx" % module_name],
    libraries=["braid"],
    library_dirs=[braid_dir],
    include_dirs=[braid_dir,numpy.get_include()]
)

setup(name=module_name,
      ext_modules=cythonize([torchbraid_ext],
                            annotate=True,
                            compiler_directives={'boundscheck': False}))
