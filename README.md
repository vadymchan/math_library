# math_library
static math library for computer graphics. Uses SIMD set instructions

---
## TODO:

### Current:
- [ ] instruction set.h - separate them
- [ ] [[#instruction set - templated]]
- [ ] add implementation for matrix multiplication (naive and unpolished solution)

### Matrix:
- [ ] basic operations:
  - [x] addition (scalar, matrix by matrix)
   - [x] base and naive
   - [ ] with bounding cases (column / row)
   - [ ] optimised (performance)
  - [ ] substraction (scalar, matrix by matrix)
  - [ ] multiplication (scalar, matrix by matrix)
  - [ ] division (scalar, )
  - [ ] transposition
  - [ ] finding determinant
  - [ ] inverse
  - [ ] rank 
- [ ] alias for the most common matrix size (2x2, 3x3, 4x4)
- [ ] support for both row-major and column-major storage
- [ ] SIMD optimizations (AVX and SSE) for operations where applicable
- [ ] support for different numerical types (float, double, etc.)
- [ ] view matrix
- [ ] projection matrix

### Vector:
- [ ] Basic operations:
  - [ ] addition (scalar, vector by vector)
  - [ ] subtraction (scalar, vector by vector)
  - [ ] multiplication (scalar, dot product, cross product)
  - [ ] division (scalar)
  - [ ] magnitude (length)
  - [ ] normalization
- [ ] alias for the most common vector sizes (2D, 3D, 4D)


### General:
- [ ] OpenMP / TBB (Intel Threading Building Blocks) support for parallelization
- [ ] Comprehensive test suite to verify correctness of operations
- [ ] Performance benchmarks to measure speed of operations
- [ ] Documentation for all classes and functions (it's in my dreams =) )
- [ ] Examples and tutorials for users
- [ ] Continuous integration setup for automated testing
---
## tree hirerarchy

```
math_library/
│
├── src/
│   ├── lib/
│   │   ├── vec3.cpp
│   │   ├── mat4.cpp
│   │   └── ... 
│   │
│   └── exe/
│       ├── main.cpp
│       └── ...
│
├── include/
│   ├── vec3.h
│   ├── mat4.h
│   └── ... 
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

P.S if the static variable is an member of the class (struct), then the priority will be given to the `s_` prefix.

P.S.S I'm still finding the best naming convention for this project, so it may change in the future. (feel free to discuss about this topic and propose your variants)
- For now, i wondering about const static variables (for me, using something like `ks_maxHealth` is really weird and i'm not sure if this is right).
- Also, mayyybe in the future i will switch naming convention for const variable to `UPPER_CASE_WITH_UNDERSCORES`
