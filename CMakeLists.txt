cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(mpm)

set(CMAKE_CXX_STANDARD 17)
set(CXX_EXTENSIONS ON)

include(CMakePackageConfigHelpers)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

file(GLOB TARGET_SOURCES src/*.cpp src/*.c)

add_executable(mpm ${TARGET_SOURCES})
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenGL 4.1 REQUIRED)
find_package(glfw3 3 REQUIRED)
find_package(Boost 1.6 REQUIRED COMPONENTS filesystem)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/shaders ${BOOST_INCLUDE_DIRS} ${CONAN_INCLUDE_DIRS})
include_directories(${COMMON_INCLUDES})

set(COMMON_LIBRARIES ${CONAN_LIBS} ${OPENGL_gl_LIBRARY} glfw OpenMP::OpenMP_CXX Boost::filesystem ${PNG_LIBRARIES} ${CMAKE_DL_LIBS})

target_link_libraries(mpm ${COMMON_LIBRARIES})

#-------------------
# Testing
#-------------------

# Compile main sources as lib for testing.
# add_library(mpm_lib ${TARGET_SOURCES})
# target_link_libraries(mpm_lib ${COMMON_LIBRARIES})
#
# enable_testing()
#
# file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/*.cpp)
#
# foreach(_test_file ${TEST_SRC_FILES})
#     get_filename_component(_test_name ${_test_file} NAME_WE)
#     add_executable(${_test_name} ${_test_file})
#     target_link_libraries(${_test_name} mpm_lib ${COMMON_LIBRARIES} ${CONAN_LIBS_GTEST})
#     add_test(${_test_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_test_name})
#     set_tests_properties(${_test_name} PROPERTIES TIMEOUT 5)
# endforeach()


