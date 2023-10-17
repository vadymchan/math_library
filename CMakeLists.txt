cmake_minimum_required(VERSION 3.10)
project(math_library)
set(CMAKE_CXX_STANDARD 20)

#-------- lib ----------------------------------------

file(GLOB_RECURSE LIB_SOURCES "src/lib/*.cpp")
file(GLOB_RECURSE PUBLIC_HEADERS "include/*.h")
file(GLOB_RECURSE PRIVATE_HEADERS "src/lib/**/*.h")

if(LIB_SOURCES) 
    message(STATUS "Build math library as static library")
    add_library(math_library STATIC ${PUBLIC_HEADERS} ${LIB_SOURCES})
    target_include_directories(math_library PUBLIC include)
    target_include_directories(math_library PRIVATE src/lib)
else()
    #Header Only
    message(STATUS "Build math library as header only library")
    add_library(math_library INTERFACE ${PUBLIC_HEADERS} ${PRIVATE_HEADERS})
    target_include_directories(math_library INTERFACE include src/lib)
endif()

#-------- exe ----------------------------------------

option(INCLUDE_G_TEST "Include the G_TEST library" ON)
option(INCLUDE_SPDLOG "Include the SPDLOG library" OFF)
option(INCLUDE_G_BENCHMARK "Include the G_BENCHMARK library" ON)

file(GLOB_RECURSE EXE_SOURCES "src/exe/*.cpp")
include_directories(${PROJECT_SOURCE_DIR}/include ${CMAKE_BINARY_DIR})

add_executable(math_library_exe ${EXE_HEADERS} ${EXE_SOURCES})

target_link_libraries(math_library_exe math_library)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/third_party)

#-------- generate_defines ----------------------------------------

add_executable(generate_SIMD_defines scripts/GenerateSIMDdefines.cpp)

add_custom_command(
    # OUTPUT include/SIMDdefines.h
    OUTPUT src/lib/simd/precompiled/SIMDdefines.h
    COMMAND generate_SIMD_defines
    DEPENDS generate_SIMD_defines
)

add_custom_target(
    run_generate_defines ALL
    DEPENDS src/lib/simd/precompiled/SIMDdefines.h
)

#-------- fetch content ----------------------------------------

include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/third_party)

if(INCLUDE_SPDLOG)
    message(STATUS "Fetching SPDLOG...")
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.9.2
    )
    FetchContent_MakeAvailable(spdlog)
    target_link_libraries(math_library_exe spdlog)
endif()

if(INCLUDE_G_TEST)
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

if(INCLUDE_G_BENCHMARK)
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

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT math_library_exe)
