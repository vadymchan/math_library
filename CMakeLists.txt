cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
project(math_library)

#-------- lib ----------------------------------------

option(MATH_LIBRARY_USE_NORMALIZE_IN_PLACE "Use in-place normalization" ON)

file(GLOB_RECURSE LIB_SOURCES "src/lib/*.cpp")
file(GLOB_RECURSE PUBLIC_HEADERS "include/math_library/*.h")
file(GLOB_RECURSE PRIVATE_HEADERS "src/lib/**/*.h")

if(LIB_SOURCES) 
    #Static Lib
    message(STATUS "Build math library as static library")
    add_library(math_library STATIC ${PUBLIC_HEADERS} ${PRIVATE_HEADERS} ${LIB_SOURCES})
    target_include_directories(math_library PUBLIC include)
    target_include_directories(math_library PRIVATE src/lib)
    if(MATH_LIBRARY_USE_NORMALIZE_IN_PLACE)
        target_compile_definitions(math_library PUBLIC MATH_LIBRARY_USE_NORMALIZE_IN_PLACE)
    endif()
else()
    #Header Only
    message(STATUS "Build math library as header only library")
    add_library(math_library INTERFACE ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})
    target_include_directories(math_library INTERFACE include)
    if(MATH_LIBRARY_USE_NORMALIZE_IN_PLACE)
        target_compile_definitions(math_library INTERFACE MATH_LIBRARY_USE_NORMALIZE_IN_PLACE)
    endif()
endif()

#-------- generate_defines ----------------------------------------

add_executable(generate_SIMD_defines scripts/generate_simd_defines.cpp)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/simd/precompiled/simd_defines.h
    COMMAND generate_SIMD_defines
    DEPENDS generate_SIMD_defines
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
    run_generate_defines ALL
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/simd/precompiled/simd_defines.h
)

add_dependencies(math_library run_generate_defines)

#-------- exe ----------------------------------------

option(MATH_LIBRARY_BUILD_TESTS "Build the tests and benchmarks for math_library" OFF)

if(MATH_LIBRARY_BUILD_TESTS)

file(GLOB_RECURSE EXE_SOURCES "src/exe/*.cpp")
file(GLOB_RECURSE EXE_HEADERS "src/exe/*.h")
include_directories(${PROJECT_SOURCE_DIR}/include ${CMAKE_BINARY_DIR})

add_executable(math_library_exe ${EXE_HEADERS} ${EXE_SOURCES})

target_link_libraries(math_library_exe math_library)

endif()

#-------- fetch content ----------------------------------------

include(CMakeDependentOption)

cmake_dependent_option(MATH_LIBRARY_INCLUDE_G_TEST "Include the G_TEST library" ON "MATH_LIBRARY_BUILD_TESTS" ON)
cmake_dependent_option(MATH_LIBRARY_INCLUDE_SPDLOG "Include the SPDLOG library" OFF "MATH_LIBRARY_BUILD_TESTS" ON)
cmake_dependent_option(MATH_LIBRARY_INCLUDE_G_BENCHMARK "Include the G_BENCHMARK library" ON "MATH_LIBRARY_BUILD_TESTS" ON)

if(MATH_LIBRARY_BUILD_TESTS)
    
        
    include(FetchContent)
    set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/third_party)

    if(MATH_LIBRARY_INCLUDE_SPDLOG)
        message(STATUS "Fetching SPDLOG...")
        FetchContent_Declare(
            spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG v1.9.2
        )
        FetchContent_MakeAvailable(spdlog)
        target_link_libraries(math_library_exe spdlog)
    endif()

    if(MATH_LIBRARY_INCLUDE_G_TEST)
        message(STATUS "Fetching Google Test...")
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.13.0
        )
        FetchContent_MakeAvailable(googletest)
        target_link_libraries(math_library_exe gtest gtest_main)
    endif()

    #set to compile the gtest using Multi-threaded Debug /MTd runtime library
    if(MSVC) 
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    endif()

    if(MATH_LIBRARY_INCLUDE_G_BENCHMARK)
        message(STATUS "Fetching Google Benchmark...")
        FetchContent_Declare(
            googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.6.0
        )
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
        FetchContent_MakeAvailable(googlebenchmark)
        get_target_property(BENCHMARK_INCLUDE_DIRS benchmark::benchmark INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(math_library_exe SYSTEM PRIVATE ${BENCHMARK_INCLUDE_DIRS})
        target_link_libraries(math_library_exe benchmark::benchmark)
    endif()

    # Startup project for VS 
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT math_library_exe)
    endif()

endif()