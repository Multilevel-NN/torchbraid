from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

module_name = 'torchbraid'
on_mac = False
on_mac = True

if not on_mac:
  braid_dir = '/home/eccyr/Packages/xbraid/braid'
  
  os.environ["CC"] = 'mpicc'
  os.environ["LDSHARED"] = 'mpicc -shared'
  extra_compile_args=['fPIC']

else:
  braid_dir = '/Users/eccyr/Packages/xbraid/braid'
  
  os.environ["CC"] = 'mpicc'
  extra_compile_args=[]

# end else
  
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
