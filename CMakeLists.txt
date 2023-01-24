cmake_minimum_required(VERSION 3.16)
project(PatchMatch)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(src)

add_library(cnpy SHARED "src/cnpy.cpp")

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-march=native" _march_native_works)
check_cxx_compiler_flag("-xHost" _xhost_works)

set(_CXX_FLAGS)
if(_march_native_works)
  message(STATUS "Using processor's vector instructions (-march=native compiler flag set)")
  set(_CXX_FLAGS "-march=native")
elseif(_xhost_works)
  message(STATUS "Using processor's vector instructions (-xHost compiler flag set)")
  set(_CXX_FLAGS "-xHost")
else()
  message(STATUS "No suitable compiler flag found for vectorization")
endif()

set(_CXX_FLAGS "-fno-math-errno")
set(_CXX_FLAGS "-Ofast")
set(_CXX_FLAGS "-DNDEBUG")
set(_CXX_FLAGS "-flto")
set(_CXX_FLAGS "-funroll-loops")
add_executable(PatchMatch main.cpp)
target_compile_options(PatchMatch PUBLIC ${_CXX_FLAGS})
target_link_libraries(PatchMatch ${OpenCV_LIBS} cnpy OpenMP::OpenMP_CXX)