#@HEADER
# ************************************************************************
# 
#                        Torchbraid v. 0.1
# 
# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC 
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. 
# Government retains certain rights in this software.
# 
# Torchbraid is licensed under 3-clause BSD terms of use:
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name National Technology & Engineering Solutions of Sandia, 
# LLC nor the names of the contributors may be used to endorse or promote 
# products derived from this software without specific prior written permission.
# 
# Questions? Contact Eric C. Cyr (eccyr@sandia.gov)
# 
# ************************************************************************
#@HEADER

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

module_name = 'test_cbs'

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
