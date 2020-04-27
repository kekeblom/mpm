cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(mpm LANGUAGES CXX C CUDA)

#-------------------
# build options
#-------------------
option(USE_OPENMP "use parallelization with OpenMp" TRUE)

#-------------------
# General
#-------------------
set(CMAKE_CXX_STANDARD 17)
set(CXX_EXTENSIONS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)
set(CUDA_SEPARABLE_COMPILATION ON)

include(CMakePackageConfigHelpers)

file(GLOB SOURCES src/*.cpp src/*.c src/*.cu)
list(REMOVE_ITEM SOURCES ${PROJECT_SOURCE_DIR}/src/main.cu)

add_library(mpm_lib ${SOURCES})
set_target_properties(mpm_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(mpm src/main.cu)

find_package(OpenGL 4.1 REQUIRED)
find_package(glfw3 3 REQUIRED)
find_package(Boost 1.65 REQUIRED COMPONENTS filesystem)
find_package(Eigen3 3.3.7 REQUIRED)
find_package(GTest)
find_package(CUDA REQUIRED)

# find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# message("-- Using openMP")
set(CMAKE_CUDA_FLAGS_DEBUG "--expt-relaxed-constexpr -G -g -Xcompiler -fopenmp")
set(CMAKE_CUDA_FLAGS_RELEASE "--expt-relaxed-constexpr -O3 -Xcompiler -fopenmp")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/shaders ${BOOST_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} submodules/cpptoml/include/)
include_directories(SYSTEM ${COMMON_INCLUDES} ${LOCAL_INCLUDE_DIR})
link_directories(${LOCAL_LIB_DIR})

set(COMMON_LIBRARIES ${OPENGL_gl_LIBRARY} glfw Boost::filesystem ${PNG_LIBRARIES} ${CMAKE_DL_LIBS})

if(USE_OPENMP)
    find_package(OpenMP)
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES} OpenMP::OpenMP_CXX)
endif()

# add Partio (if available, otherwise no exporting of particles)
find_library(PARTIO partio PATH ${LOCAL_LIB_DIR})
if(PARTIO)
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES} ${PARTIO})
    add_definitions(-DPARTIO_AVAILABLE)
else()
    message("Warnig: Partio library not found, will not be able to write particles to file.")
endif()

set_target_properties(mpm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(mpm mpm_lib ${COMMON_LIBRARIES})

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

#-------------------
# Testing (optional)
#-------------------
if(GTest_FOUND)
    # Compile main sources as lib for testing.
    target_link_libraries(mpm_lib ${COMMON_LIBRARIES})

    enable_testing()

    file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/*.cpp ${PROJECT_SOURCE_DIR}/tests/*.cu)

    foreach(_test_file ${TEST_SRC_FILES})
        get_filename_component(_test_name ${_test_file} NAME_WE)
        add_executable(${_test_name} ${_test_file})
        target_link_libraries(${_test_name} mpm_lib ${COMMON_LIBRARIES} ${GTEST_LIBRARIES} pthread)
        add_test(${_test_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_test_name})
				set_target_properties(${_test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_tests_properties(${_test_name} PROPERTIES TIMEOUT 5)
        add_dependencies(check ${_test_name})
    endforeach()
else()
    message("Warnig: GTest not found. No testing possible.")
endif()

