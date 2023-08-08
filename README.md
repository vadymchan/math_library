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

### Third parties:
- [Google Test](https://github.com/google/googletest)
- [Google Benchmark](https://github.com/google/benchmark)
- [spdlog](https://github.com/gabime/spdlog)

---
### How to build:

1. Clone the repository 
``` git clone git@github.com:vadymchan/math_library.git ```
2. Build project:
``` cmake .```
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
│       │   └── Options.h
│       │
│       └── simd
│           ├── instruction_set
│           │   ├── InstructionSet.h
│           │   ├── InstructionSetDouble.h
│           │   ├── InstructionSetFloat.h
│           │   └── InstructionSetInt.h
│           │
│           └── precompiled
│               └── SIMDdefines.h
│ 
├── scripts (for generation instruction for current hardware)
│   └── GenerateSIMDdefines.cpp
│
├── include (for user)
│    ├── All.h
│    ├── Graphics.h
│    ├── Matrix.h
│    └── Vector.h
│
└── CMakeLists.txt

```

## naming conventions for this project:
 

| Code Element | Naming Convention | Example |
| --- | --- | --- |
| Classes | PascalCase | `GameEngine` |
| Structures | PascalCase | `Vector2D` |
| Functions / Methods | camelCase | `updatePosition()` |
| Public Variables | camelCase | `playerHealth` |
| Member Variables | `m_` prefix + camelCase | `m_position` |
| Private Member Variables | `m_` prefix + camelCase + `_` postfix | `m_position_` | 
| Private Methods | camelCase + `_` postfix | `updatePosition_()` | 
| Constants (const and constexpr) | `k` prefix + PascalCase | `kMaxPlayers` | - i've seen it in Google's C++ style guide
| Enums | PascalCase for type and values | `enum class Difficulty { Easy, Medium, Hard };` |
| Namespaces | lowercase with underscores | `game_logic` |
| Interface Classes | `I` prefix + PascalCase | `ICollidable` |
| Boolean Variables | `is` or `has` prefix + camelCase | `isVisible`, `hasPowerUp` |
| Template Parameters | Single uppercase letters | `template <class T>` |
| File Names | lowercase with underscores, match class name | `game_engine.h` |
| Macros | UPPER_CASE_WITH_UNDERSCORES | `#define MAX_HEALTH 100` |
| Typedefs and Type Aliases | PascalCase | `typedef long int BigNum;` or `using BigNum = long int;` |
| Global Variables | g_ prefix + camelCase | `g_gameState` |
| Static Variables | s_ prefix + camelCase | `s_instanceCount` |
| Concepts | PascalCase | `Sortable`, `Drawable` |

P.S if the static variable is an member of the class (struct), then the priority will be given to the `s_` prefix.

P.S.S I'm still finding the best naming convention for this project, so it may change in the future. (feel free to discuss about this topic and propose your variants)
- For now, i wondering about const static variables (for me, using something like `ks_maxHealth` is really weird and i'm not sure if this is right).
- Also, mayyybe in the future i will switch naming convention for const variable to `UPPER_CASE_WITH_UNDERSCORES`
