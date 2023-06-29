cmake_minimum_required(VERSION 3.10)
project(math_library)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

#-------- lib ----------------------------------------

# Get all source and header files for the library
file(GLOB_RECURSE LIB_SOURCES "src/lib/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

# Create the library
add_library(math_library STATIC ${HEADERS} ${LIB_SOURCES})

#-------- exe ----------------------------------------

# Get all source files for the executable
file(GLOB_RECURSE EXE_SOURCES "src/exe/*.cpp")

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Create the executable
add_executable(math_library_exe ${EXE_SOURCES})

# Link the library to the executable
target_link_libraries(math_library_exe math_library)
