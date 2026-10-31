cmake_minimum_required(VERSION 3.18)

project(torchbraid VERSION 0.1 LANGUAGES C)

# The TorchBraid build. It is used directly with cmake, and by pip through
# scikit-build-core (see pyproject.toml), which sets SKBUILD. Everything is
# written to the build tree; nothing is generated inside src/.
# See CMAKEINSTRUCTIONS.md.

if(SKBUILD)
  set(tests_default OFF)
else()
  set(tests_default ON)
endif()
option(TORCHBRAID_ENABLE_TESTS     "Register the unit tests with CTest"                  ${tests_default})
option(TORCHBRAID_ENABLE_GPU_TESTS "Register the GPU-direct MPI test with CTest"         OFF)

set(TORCHBRAID_XBRAID_ROOT "" CACHE PATH
    "Prebuilt xbraid/braid directory containing braid.h and libbraid.a. If empty, XBraid is built from source.")
set(TORCHBRAID_XBRAID_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/xbraid" CACHE PATH
    "XBraid checkout to build from when TORCHBRAID_XBRAID_ROOT is empty. Fetched with git if it does not exist.")
set(TORCHBRAID_XBRAID_GIT_REPOSITORY "https://github.com/XBraid/xbraid.git" CACHE STRING
    "Repository used to fetch XBraid")
set(TORCHBRAID_XBRAID_GIT_TAG "master" CACHE STRING
    "Branch, tag or commit used to fetch XBraid")
set(TORCHBRAID_XBRAID_BUILD "cmake" CACHE STRING
    "How XBraid is built when TORCHBRAID_XBRAID_ROOT is empty: 'cmake' compiles it here, 'make' runs XBraid's own Makefile")
set_property(CACHE TORCHBRAID_XBRAID_BUILD PROPERTY STRINGS cmake make)
if(NOT TORCHBRAID_XBRAID_BUILD MATCHES "^(cmake|make)$")
  message(FATAL_ERROR "TORCHBRAID_XBRAID_BUILD must be 'cmake' or 'make', got '${TORCHBRAID_XBRAID_BUILD}'")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # matches 'make debug=no' for XBraid
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)

######################################################################
# Dependencies
######################################################################

find_package(MPI REQUIRED COMPONENTS C)
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

function(torchbraid_python_query out_var code what)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "${code}"
    OUTPUT_VARIABLE value
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${what} is not importable by ${Python_EXECUTABLE}")
  endif()
  set(${out_var} "${value}" PARENT_SCOPE)
endfunction()

torchbraid_python_query(CYTHON_VERSION "import Cython; print(Cython.__version__)" "Cython")
if(CYTHON_VERSION VERSION_LESS 0.29.32)
  message(FATAL_ERROR "Cython >= 0.29.32 is required, found ${CYTHON_VERSION}")
endif()
message(STATUS "Found Cython: ${CYTHON_VERSION}")

# torchbraid_app.pyx does 'cimport mpi4py.MPI', so mpi4py must be visible to Cython
torchbraid_python_query(MPI4PY_INCLUDE_DIR "import mpi4py; print(mpi4py.get_include())" "mpi4py")
message(STATUS "Found mpi4py: ${MPI4PY_INCLUDE_DIR}")

######################################################################
# XBraid
######################################################################

if(TORCHBRAID_XBRAID_ROOT)
  set(braid_lib "${TORCHBRAID_XBRAID_ROOT}/libbraid.a")
  if(NOT EXISTS "${TORCHBRAID_XBRAID_ROOT}/braid.h" OR NOT EXISTS "${braid_lib}")
    message(FATAL_ERROR "TORCHBRAID_XBRAID_ROOT=${TORCHBRAID_XBRAID_ROOT} must contain braid.h and libbraid.a "
                        "(run 'make debug=no braid' in the XBraid checkout)")
  endif()
  message(STATUS "Using prebuilt XBraid: ${braid_lib}")

  add_library(xbraid STATIC IMPORTED)
  set_target_properties(xbraid PROPERTIES
    IMPORTED_LOCATION "${braid_lib}"
    INTERFACE_INCLUDE_DIRECTORIES "${TORCHBRAID_XBRAID_ROOT}"
    INTERFACE_LINK_LIBRARIES "MPI::MPI_C;m")
else()
  if(EXISTS "${TORCHBRAID_XBRAID_SOURCE_DIR}/braid/braid.h")
    set(xbraid_source_dir "${TORCHBRAID_XBRAID_SOURCE_DIR}")
  else()
    include(FetchContent)
    message(STATUS "Fetching XBraid (${TORCHBRAID_XBRAID_GIT_TAG}) from ${TORCHBRAID_XBRAID_GIT_REPOSITORY}")
    # XBraid has no CMakeLists.txt, so this only downloads it into the build tree
    FetchContent_Declare(xbraid
      GIT_REPOSITORY "${TORCHBRAID_XBRAID_GIT_REPOSITORY}"
      GIT_TAG        "${TORCHBRAID_XBRAID_GIT_TAG}")
    FetchContent_MakeAvailable(xbraid)
    set(xbraid_source_dir "${xbraid_SOURCE_DIR}")
  endif()
  set(braid_dir "${xbraid_source_dir}/braid")
  message(STATUS "Building XBraid from source with ${TORCHBRAID_XBRAID_BUILD}: ${xbraid_source_dir}")

  if(TORCHBRAID_XBRAID_BUILD STREQUAL "cmake")
    # Take the source list from the 'BRAID_FILES =' assignment in XBraid's Makefile
    # so it tracks XBraid; reconfigure if that Makefile changes
    set(braid_makefile "${braid_dir}/Makefile")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${braid_makefile}")
    file(READ "${braid_makefile}" makefile_text)
    string(REGEX MATCH "BRAID_FILES[ \t]*=([^\n\\\\]*\\\\\n)*[^\n]*" braid_files "${makefile_text}")
    string(REGEX REPLACE "^BRAID_FILES[ \t]*=" "" braid_files "${braid_files}")
    string(REPLACE "\\" " " braid_files "${braid_files}")
    string(REGEX MATCHALL "[^ \t\n]+" braid_sources "${braid_files}")
    if(NOT braid_sources)
      message(FATAL_ERROR "Could not find the BRAID_FILES list in ${braid_makefile}; "
                          "use -DTORCHBRAID_XBRAID_BUILD=make instead")
    endif()
    list(TRANSFORM braid_sources PREPEND "${braid_dir}/")
    foreach(source IN LISTS braid_sources)
      if(NOT EXISTS "${source}")
        message(FATAL_ERROR "${source} is listed in ${braid_makefile} but does not exist")
      endif()
    endforeach()

    add_library(xbraid STATIC ${braid_sources})
    set_target_properties(xbraid PROPERTIES
      OUTPUT_NAME braid
      POSITION_INDEPENDENT_CODE ON)
    target_include_directories(xbraid PUBLIC "${braid_dir}")
    target_link_libraries(xbraid PUBLIC MPI::MPI_C m)
  else()
    # XBraid's Makefile builds in place, so its objects and libbraid.a are written
    # into the XBraid checkout (as with the pip build)
    find_program(TORCHBRAID_MAKE_EXECUTABLE NAMES gmake make REQUIRED)
    set(make_args debug=no braid)
    if(MPI_C_COMPILER)
      # overrides the compiler XBraid's makefile.inc picks from the hostname
      list(APPEND make_args "MPICC=${MPI_C_COMPILER}")
    endif()

    set(braid_lib "${braid_dir}/libbraid.a")
    file(GLOB braid_make_deps CONFIGURE_DEPENDS "${braid_dir}/*.c" "${braid_dir}/*.h")
    add_custom_command(
      OUTPUT  "${braid_lib}"
      COMMAND "${TORCHBRAID_MAKE_EXECUTABLE}" ${make_args}
      DEPENDS ${braid_make_deps} "${braid_dir}/Makefile" "${xbraid_source_dir}/makefile.inc"
      WORKING_DIRECTORY "${xbraid_source_dir}"
      COMMENT "Building XBraid with make"
      VERBATIM)
    add_custom_target(xbraid_make DEPENDS "${braid_lib}")

    add_library(xbraid STATIC IMPORTED)
    set_target_properties(xbraid PROPERTIES
      IMPORTED_LOCATION "${braid_lib}"
      INTERFACE_INCLUDE_DIRECTORIES "${braid_dir}"
      INTERFACE_LINK_LIBRARIES "MPI::MPI_C;m")
    add_dependencies(xbraid xbraid_make)
  endif()
endif()

######################################################################
# Python package, staged in <build>/python so it can be imported directly
######################################################################

set(TORCHBRAID_SOURCE_ROOT "${PROJECT_SOURCE_DIR}/src")
set(TORCHBRAID_PYTHON_DIR  "${PROJECT_BINARY_DIR}/python")

file(GLOB_RECURSE package_files CONFIGURE_DEPENDS RELATIVE "${TORCHBRAID_SOURCE_ROOT}"
  "${TORCHBRAID_SOURCE_ROOT}/torchbraid/*.py"
  "${TORCHBRAID_SOURCE_ROOT}/torchbraid/*.pyx"
  "${TORCHBRAID_SOURCE_ROOT}/torchbraid/*.pxd"
  "${TORCHBRAID_SOURCE_ROOT}/torchbraid/*.h")

set(staged_files)
foreach(file IN LISTS package_files)
  add_custom_command(
    OUTPUT  "${TORCHBRAID_PYTHON_DIR}/${file}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${TORCHBRAID_SOURCE_ROOT}/${file}" "${TORCHBRAID_PYTHON_DIR}/${file}"
    DEPENDS "${TORCHBRAID_SOURCE_ROOT}/${file}"
    COMMENT "Staging ${file}"
    VERBATIM)
  list(APPEND staged_files "${TORCHBRAID_PYTHON_DIR}/${file}")
endforeach()
add_custom_target(torchbraid_python ALL DEPENDS ${staged_files})

# Cython 'include' statements pull .pyx files into each other, so any change
# to a .pyx/.pxd in the package regenerates every extension
file(GLOB_RECURSE cython_deps CONFIGURE_DEPENDS
  "${TORCHBRAID_SOURCE_ROOT}/torchbraid/*.pyx"
  "${TORCHBRAID_SOURCE_ROOT}/torchbraid/*.pxd")

# torchbraid_add_extension(<dotted.module.name> [LINK_XBRAID])
function(torchbraid_add_extension module)
  cmake_parse_arguments(arg "LINK_XBRAID" "" "" ${ARGN})

  string(REPLACE "." "/" module_path "${module}")
  string(REPLACE "." "_" target "${module}")
  get_filename_component(module_dir  "${module_path}" DIRECTORY)
  get_filename_component(module_leaf "${module_path}" NAME)

  set(pyx_file "${TORCHBRAID_SOURCE_ROOT}/${module_path}.pyx")
  set(c_file   "${PROJECT_BINARY_DIR}/cython/${module_path}.c")

  get_filename_component(c_dir "${c_file}" DIRECTORY)
  add_custom_command(
    OUTPUT  "${c_file}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${c_dir}"
    COMMAND "${Python_EXECUTABLE}" -m cython -3
            -I "${TORCHBRAID_SOURCE_ROOT}"
            -o "${c_file}" "${pyx_file}"
    DEPENDS ${cython_deps}
    WORKING_DIRECTORY "${TORCHBRAID_SOURCE_ROOT}"
    COMMENT "Cythonizing ${module}"
    VERBATIM)

  Python_add_library(${target} MODULE WITH_SOABI "${c_file}")
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME ${module_leaf}
    # the generator expression stops multi-config generators adding a per-config subdirectory
    LIBRARY_OUTPUT_DIRECTORY "$<1:${TORCHBRAID_PYTHON_DIR}/${module_dir}>")
  target_include_directories(${target} PRIVATE
    "${TORCHBRAID_SOURCE_ROOT}/${module_dir}"
    "${MPI4PY_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE Python::NumPy MPI::MPI_C)
  if(arg_LINK_XBRAID)
    target_link_libraries(${target} PRIVATE xbraid)
  endif()
  add_dependencies(${target} torchbraid_python)

  if(SKBUILD)
    # pip build: the python sources come from wheel.packages in pyproject.toml
    install(TARGETS ${target} LIBRARY DESTINATION "${module_dir}")
  endif()
endfunction()

torchbraid_add_extension(torchbraid.torchbraid_app LINK_XBRAID)
torchbraid_add_extension(torchbraid.test_fixtures.test_cbs LINK_XBRAID)
torchbraid_add_extension(torchbraid.test_fixtures.gpumpi_check)

######################################################################
# Install
######################################################################

# pip builds install the extensions into the wheel in torchbraid_add_extension
if(NOT SKBUILD)
  set(TORCHBRAID_INSTALL_PYTHONDIR
      "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages"
      CACHE PATH "Directory (relative to CMAKE_INSTALL_PREFIX, or absolute) where the torchbraid package is installed")

  install(DIRECTORY "${TORCHBRAID_PYTHON_DIR}/torchbraid"
          DESTINATION "${TORCHBRAID_INSTALL_PYTHONDIR}"
          PATTERN "__pycache__" EXCLUDE)
endif()

######################################################################
# Tests (mirrors tox.ini, with the MPI tests also run on 1 rank)
######################################################################

if(TORCHBRAID_ENABLE_TESTS)
  enable_testing()

  set(test_dir "${PROJECT_SOURCE_DIR}/tests")
  set(test_env "PYTHONPATH=${TORCHBRAID_PYTHON_DIR}")

  foreach(test IN ITEMS test_ContextTimer test_mean_initial_guess test_data_parallel)
    add_test(NAME ${test}
             COMMAND "${Python_EXECUTABLE}" "${test_dir}/${test}.py"
             WORKING_DIRECTORY "${test_dir}")
    set_tests_properties(${test} PROPERTIES ENVIRONMENT "${test_env}" LABELS serial)
  endforeach()

  # torchbraid_add_mpi_test(<name> <nprocs> <label>)
  function(torchbraid_add_mpi_test test nprocs label)
    add_test(NAME ${test}_np${nprocs}
             COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${nprocs} ${MPIEXEC_PREFLAGS}
                     "${Python_EXECUTABLE}" "${test_dir}/${test}.py" ${MPIEXEC_POSTFLAGS}
             WORKING_DIRECTORY "${test_dir}")
    set_tests_properties(${test}_np${nprocs} PROPERTIES
      ENVIRONMENT "${test_env}"
      PROCESSORS ${nprocs}
      LABELS "mpi;${label}")
  endfunction()

  foreach(test IN ITEMS test_callbacks test_FlatPackUnpack)
    torchbraid_add_mpi_test(${test} 1 serial)
  endforeach()

  foreach(test IN ITEMS test_layer_parallel test_layer_parallel_multinode test_composite
                        test_grad_update test_gru_layer_parallel)
    torchbraid_add_mpi_test(${test} 1 serial)
    torchbraid_add_mpi_test(${test} 3 parallel)
  endforeach()

  if(TORCHBRAID_ENABLE_GPU_TESTS)
    torchbraid_add_mpi_test(test_gpu_direct_commu 2 gpu)
  endif()
endif()
