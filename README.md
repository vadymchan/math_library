# math_library

header-only math library for computer graphics. Uses SIMD set instructions

## Requirements

- A C++ compiler supporting C++20 (or later)
- CPU with the following SIMD support (at least one of them): AVX 2, AVX, SSE 4.2, SSE 4.1, SSSE3, SSE 3

---

### Features:

- generates macros during build of the project for choosing the best SIMD instruction set for your CPU
- uses SIMD instructions for faster calculations
- header-only library (but you can build it as a static library if you want - just need to add .cpp file)
- uses C++20 features (concepts)
- uses CMake for building
- uses Google Test for unit testing
- uses Doxygen for documentation
- uses Google Benchmark for benchmarking
- uses spdlog for logging (for now not used, but will be used in the future)
- implements both row-major and column-major matrices
- implements both left-handed and right-handed coordinate systems
- uses clang format for enforcing a consistent code style and formatting
- uses clang tidy for static code analysis and catching potential bugs and style issues
- implements various variables and functions for graphics (such as view, perspective/orthogonal projection matrix) with different configurations (row/column major, left/right hand/ different depth for projection matrices etc.)

### Classes

- Matrix
- Vector
- Point
- Dimension

### Third parties:

- [Google Test](https://github.com/google/googletest)
- [Google Benchmark](https://github.com/google/benchmark)
- [spdlog](https://github.com/gabime/spdlog) - currently not used

---

### How to build:

1. Clone the repository
   `git clone git@github.com:vadymchan/math_library.git`
2. Build project:
   ` cmake .`
   P.S. You can use cmake-gui for more comfortable building (and there you can choose options like build with third parties or not)

---

### Examples:

No examples for now, but you can look at tests and benchmarks for examples of usage

---

### TODO:

- [ ] add more tests (currently not tested different types (only float was tested), SIMD instruction (only AVX was tested)
- [ ] add more benchmarks (the same as with tests)
- [ ] add more documentation (currently no documentation at all =) )
- [ ] add logging (currently no logging at all =) )

---

### tree hirerarchy

```
math_library
│
├── src
│   ├── exe (for testing)
│   │   ├── benchmarks.h
│   │   ├── main.cpp
│   │   └── tests.h
│   │
│   └── lib (internal implementation - hidden from the user)
│       ├── options
│       │   └── options.h
│       │
│       ├── simd
│       │   ├── instruction_set
│       │   │   ├── instruction_set.h
│       │   │   ├── instruction_set_double.h
│       │   │   ├── instruction_set_float.h
│       │   │   ├── instruction_set_int.h
│       │   │   └── instruction_set_uint32_t.h
│       │   │
│       │   └── precompiled
│       │       └── simd_defines.h
│       │
│       └── utils
│           ├── concepts.h
│           └── utils.h
│ 
├── scripts (for generation instruction for current hardware)
│   └── generate_simd_defines.cpp
│
├── include (for user)
│    ├── math_library
│    │   ├── all.h
│    │   ├── dimension.h
│    │   ├── graphics.h
│    │   ├── matrix.h
│    │   ├── point.h
│    │   └── vector.h
│
└── CMakeLists.txt

```

## naming conventions for this project:

| Code Element                     | Naming Convention                                 | Example                                  |
| -------------------------------- | ------------------------------------------------- | ---------------------------------------- |
| Classes                          | CamelCase                                         | `GameEngine`                             |
| Structures                       | CamelCase                                         | `Vector2D`                               |
| Unions                           | CamelCase                                         | `DataUnion`                              |
| Functions / Methods              | camelCase with `g_` prefix (for global functions) | `updatePosition()`, `g_initializeGame()` |
| Public Member Variables          | `m_` prefix + camelCase                           | `m_position`                             |
| Private Member Variables         | `m_` prefix + camelCase + `_` postfix             | `m_position_`                            |
| Protected Member Variables       | `m_` prefix + camelCase + `_` postfix             | `m_counter_`                             |
| Public Methods                   | camelCase                                         | `updatePosition()`                       |
| Protected Methods                | camelCase + `_` postfix                           | `run_()`                                 |
| Private Methods                  | camelCase + `_` postfix                           | `initialize_()`                          |
| Enums (both scoped and unscoped) | CamelCase                                         | `Color`                                  |
| Enum Constants                   | CamelCase                                         | `Difficulty::Easy`, `RED`                |
| Namespaces                       | lowercase with underscores                        | `game_logic`                             |
| Interface Classes                | `I` prefix + CamelCase                            | `ICollidable`                            |
| Template Parameters              | Single uppercase letters (contradicts the table)  | `T`, `U`                                 |
| Macros                           | UPPER_CASE_WITH_UNDERSCORES                       | `MAX_HEALTH`                             |
| Typedefs and Type Aliases        | CamelCase                                         | `BigInt`                                 |
| Static Constant Member Variables | `s_k` prefix + CamelCase                          | `s_kMaxValue`                            |
| Class Constant Member Variables  | `s_k` prefix + CamelCase                          | `s_kDefaultColor`                        |
| Constants                        | `k` prefix + CamelCase                            | `kMaxPlayers`                            |
| Static Variables                 | `s_` prefix + camelCase                           | `s_instanceCount`                        |
| Global Variables                 | `g_` prefix + camelCase                           | `g_gameState`                            |
| Global Constants                 | `g_k` prefix + CamelCase                          | `g_kInitialSpeed`                        |
| Class Members                    | `s_` prefix + camelCase                           | `s_memberVariable`                       |
| Class Methods                    | CamelCase (no prefix)                             | `ClassMethod()`                          |
| Template Value                   | CamelCase                                         | `DefaultValue`                           |
| Type Template                    | CamelCase                                         | `TypeParam`                              |
| Files                            | lowercase with underscores                        | `game_engine.h`, `game_engine.cc`        |
| Folders                          | lowercase with underscores                        | `physics_engine/`, `graphics_engine/`    |
| Function Parameters              | camelCase                                         | `playerScore`, `gameLevel`               |
| Constant Parameters              | `k` prefix + CamelCase                            | `kMaxLevels`, `kInitialHealth`           |

P.S. for some elements i'm still not sure:

- for class methods do i really need to add `s_` prefix 🤔
- do i need to add `s_k`, `g_k` prefixes 🤔

### Issues

#### include in other project using CMake FetchContent and math_library build as header only library

this problem occurs since math_library is header only library (INTERFACE), so it doesn't have to build before the main project. But math_library dependent on run_generate_defines which generates
simd_defines.h file for internal use. So if math_library is not built before the main project, then simd_defines.h is not generated and math_library can't be used.

Solution:
add to CMakeLists.txt of the main project where you want to use math_library:

```cmake
add_dependency (<project_name> math_library)
```

this will forcefully build math_library (specifically run_generate_defines) before the main project.
