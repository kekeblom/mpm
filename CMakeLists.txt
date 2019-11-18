cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(mpm)

set(CMAKE_CXX_STANDARD 17)
set(CXX_EXTENSIONS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

set(USE_OPENMP TRUE)

include(CMakePackageConfigHelpers)

file(GLOB TARGET_SOURCES src/*.cpp src/*.c)

add_executable(mpm ${TARGET_SOURCES})

find_package(OpenGL 4.1 REQUIRED)
find_package(glfw3 3 REQUIRED)
find_package(Boost 1.6 REQUIRED COMPONENTS filesystem)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(GTest REQUIRED)
if(USE_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message("using openMP")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/shaders ${BOOST_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})
include_directories(${COMMON_INCLUDES} ${LOCAL_INCLUDE_DIR})
link_directories(${LOCAL_LIB_DIR})

set(COMMON_LIBRARIES ${OPENGL_gl_LIBRARY} glfw  Boost::filesystem ${PNG_LIBRARIES} ${CMAKE_DL_LIBS})

if(USE_OPENMP)
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES} OpenMP::OpenMP_CXX)
endif()


# add Partio (if available)
find_library(PARTIO partio PATH ${LOCAL_LIB_DIR})
if(PARTIO)
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES} ${PARTIO})
    add_definitions(-DPARTIO_AVAILABLE)
else()
    message("Warnig: Partio library not found, will not be able to write particles to file.")
endif()

target_link_libraries(mpm ${COMMON_LIBRARIES})

#-------------------
# Testing
#-------------------

# Compile main sources as lib for testing.
add_library(mpm_lib ${TARGET_SOURCES})
target_link_libraries(mpm_lib ${COMMON_LIBRARIES})

enable_testing()

file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/*.cpp)

foreach(_test_file ${TEST_SRC_FILES})
    get_filename_component(_test_name ${_test_file} NAME_WE)
    add_executable(${_test_name} ${_test_file})
    target_link_libraries(${_test_name} mpm_lib ${COMMON_LIBRARIES} ${GTEST_LIBRARIES} pthread)
    message(${_test_name})
    add_test(${_test_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_test_name})
    set_tests_properties(${_test_name} PROPERTIES TIMEOUT 5)
endforeach()


