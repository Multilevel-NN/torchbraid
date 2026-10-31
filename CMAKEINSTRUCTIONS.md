# Building with CMake

TorchBraid is built with CMake. The [PIP Install](README.md#build-torchbraid-pip-recommended) runs this same
build through [scikit-build-core](https://scikit-build-core.readthedocs.io), or it can be run with CMake directly as
described here. Either way, all generated files (Cython C sources, the XBraid library, and the compiled extensions)
go into the build directory, and nothing is written into `src/`.

## Passing options through pip

Any of the [options](#options) below can be given to `pip install` with `-C cmake.define.<OPTION>=<value>`, e.g.

   `pip install . -C cmake.define.TORCHBRAID_XBRAID_ROOT=/path/to/xbraid/braid`  
   `pip install . -C cmake.define.TORCHBRAID_XBRAID_BUILD=make`

or through the environment, which also works when pip is called by other tools such as tox:

   `SKBUILD_CMAKE_DEFINE="TORCHBRAID_XBRAID_ROOT=/path/to/xbraid/braid" pip install .`

See [Using a prebuilt XBraid](#using-a-prebuilt-xbraid) for what `TORCHBRAID_XBRAID_ROOT` must point at.

pip keeps its CMake build tree in `build/<wheel tag>`, so reinstalling only rebuilds what changed. Options are
stored in that tree's CMake cache, so they stay in effect for later installs until they are passed with a new
value (an empty value clears a path such as `TORCHBRAID_XBRAID_ROOT`) or `build/<wheel tag>` is deleted. With
`pip install -e .`, changes to `.py` files apply directly; changes to `.pyx` files require a re-installation.

With pip's default build isolation, mpi4py is compiled in a temporary environment against the first `mpicc` found.
To build against an mpi4py you have already installed, use `pip install --no-build-isolation .` (this requires
`scikit-build-core`, `cython`, `numpy`, and `mpi4py` to be installed first).

## Requirements:

  + CMake >= 3.18
  + An MPI C compiler/library (found with CMake's `FindMPI`)
  + python libs:
    cython>=0.29.32,
    numpy,
    mpi4py (built against the same MPI),
    torch>=2.0.1,
    torchvision>=0.15.2,
    matplotlib

CMake uses the python found by `FindPython`; an activated virtual environment is preferred. To pick a specific
interpreter pass `-DPython_EXECUTABLE=/path/to/python`.

## Build TorchBraid
  1. Configure:
     `cmake -S . -B build/cmake`
  1. Build:
     `cmake --build build/cmake -j`
  1. Use it directly from the build tree:
     `export PYTHONPATH=${PYTHONPATH}:/path/to/TorchBraid/build/cmake/python`
  1. Or install it:
     `cmake --install build/cmake --prefix /path/to/prefix`
     which places the package in `<prefix>/lib/pythonX.Y/site-packages` (change with `-DTORCHBRAID_INSTALL_PYTHONDIR=...`).

## XBraid

By default XBraid is built from source. The source comes from:

  1. `src/xbraid` if it exists, otherwise
  1. a git clone into the build directory, controlled by `-DTORCHBRAID_XBRAID_GIT_REPOSITORY=...` and
     `-DTORCHBRAID_XBRAID_GIT_TAG=...` (default `master`).

A different checkout can be used with `-DTORCHBRAID_XBRAID_SOURCE_DIR=/path/to/xbraid`.

How it is built is selected with `-DTORCHBRAID_XBRAID_BUILD=cmake|make`:

  + `cmake` (default): CMake compiles XBraid into a static library in the build directory, with the same MPI
    compiler and `CMAKE_BUILD_TYPE` flags as the rest of TorchBraid. The source list is read from the
    `BRAID_FILES` assignment in `xbraid/braid/Makefile`, so it follows XBraid as files are added or removed.
  + `make`: CMake runs XBraid's own `make debug=no braid`, so XBraid gets the compiler flags from its
    `makefile.inc`. The MPI compiler found by CMake is passed as `MPICC`. XBraid's Makefile builds in place,
    so the object files and `libbraid.a` are written into the XBraid checkout.

### Using a prebuilt XBraid

To use an XBraid you have already built (`make debug=no braid`), point at its `braid` directory instead:

   `cmake -S . -B build/cmake -DTORCHBRAID_XBRAID_ROOT=/path/to/xbraid/braid`  
   `pip install . -C cmake.define.TORCHBRAID_XBRAID_ROOT=/path/to/xbraid/braid`

When `TORCHBRAID_XBRAID_ROOT` is set, XBraid is never fetched or compiled and `TORCHBRAID_XBRAID_BUILD` is
ignored. There is no fallback: configuring fails if the directory does not contain both `braid.h` and
`libbraid.a`. These must be in the same directory, as in the `braid` directory of an XBraid checkout.

The prebuilt library must be
  + compiled with `-fPIC`, since it is linked into Python extension modules (XBraid's `makefile.inc` does this), and
  + built with the same MPI as mpi4py and TorchBraid.

The value is stored in the CMake cache of the build directory, and pip reuses its build directory
(`build/<wheel tag>`). A later `cmake` or `pip install .` that does not pass the option will therefore still use
the prebuilt XBraid. To build XBraid from source again, pass it empty (`-DTORCHBRAID_XBRAID_ROOT=` or
`-C cmake.define.TORCHBRAID_XBRAID_ROOT=`), or delete the build directory.

## Options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | Compiler optimization for XBraid and the extensions |
| `TORCHBRAID_XBRAID_ROOT` | empty | Prebuilt `xbraid/braid` directory with `braid.h` and `libbraid.a` |
| `TORCHBRAID_XBRAID_BUILD` | `cmake` | Build XBraid with CMake (`cmake`) or with XBraid's Makefile (`make`) |
| `TORCHBRAID_XBRAID_SOURCE_DIR` | `src/xbraid` | XBraid checkout to compile from |
| `TORCHBRAID_XBRAID_GIT_REPOSITORY` | XBraid GitHub | Repository to fetch XBraid from |
| `TORCHBRAID_XBRAID_GIT_TAG` | `master` | Branch, tag, or commit to fetch |
| `TORCHBRAID_INSTALL_PYTHONDIR` | `lib/pythonX.Y/site-packages` | Install location of the package (not used by pip) |
| `TORCHBRAID_ENABLE_TESTS` | `ON` (`OFF` with pip) | Register the unit tests with CTest |
| `TORCHBRAID_ENABLE_GPU_TESTS` | `OFF` | Register the GPU-direct MPI test |

## Run Unit Tests

The tests run against the package in the build tree:

   `ctest --test-dir build/cmake --output-on-failure`

Use `-L serial` or `-L parallel` to run a subset (`parallel` runs on 3 MPI ranks). Extra `mpiexec` arguments,
such as `--oversubscribe` on machines with fewer than 3 cores, can be given at configure time with
`-DMPIEXEC_PREFLAGS=--oversubscribe`.

## To clean the directory:

   `rm -fr build/cmake`
