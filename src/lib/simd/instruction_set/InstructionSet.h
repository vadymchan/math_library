/**
 * @file InstructionSet.h
 */

#pragma once

#include "../../options/Options.h"
#include "../precompiled/SIMDdefines.h"

#include <immintrin.h>

#include <type_traits>

namespace math {

template <typename T>
concept SimdSupportedType = std::is_same_v<T, int> || std::is_same_v<T, float>
                         || std::is_same_v<T, double>;

template <typename T>
class InstructionSet {
  public:
  static_assert(SimdSupportedType<T>,
                "InstructionSet supports only int, float, and double types");

  using AddFunc = void (*)(T*, const T*, size_t);

  static auto GetAddFunc() -> AddFunc {
#ifdef SUPPORTS_AVX2
    return AddAvx2;
#elif defined(SUPPORTS_AVX)
    return AddAvx;
#elif defined(SUPPORTS_SSE4_2)
    return AddSse42;
#elif defined(SUPPORTS_SSE4_1)
    return AddSse41;
#elif defined(SUPPORTS_SSSE3)
    return AddSsse3;
#elif defined(SUPPORTS_SSE3)
    return AddSse3;
#else
    return AddFallback;
#endif
  }

  using AddScalarFunc = void (*)(T*, T, size_t);

  static auto GetAddScalarFunc() -> AddScalarFunc {
#ifdef SUPPORTS_AVX2
    return AddScalarAvx2;
#elif defined(SUPPORTS_AVX)
    return AddScalarAvx;
#elif defined(SUPPORTS_SSE4_2)
    return AddScalarSse42;
#elif defined(SUPPORTS_SSE4_1)
    return AddScalarSse41;
#elif defined(SUPPORTS_SSSE3)
    return AddScalarSsse3;
#elif defined(SUPPORTS_SSE3)
    return AddScalarSse3;
#else
    return AddScalarFallback;
#endif
  }

  using SubFunc = void (*)(T*, const T*, size_t);

  static auto GetSubFunc() -> SubFunc {
#ifdef SUPPORTS_AVX2
    return SubAvx2;
#elif defined(SUPPORTS_AVX)
    return SubAvx;
#elif defined(SUPPORTS_SSE4_2)
    return SubSse42;
#elif defined(SUPPORTS_SSE4_1)
    return SubSse41;
#elif defined(SUPPORTS_SSSE3)
    return SubSsse3;
#elif defined(SUPPORTS_SSE3)
    return SubSse3;
#else
    return SubFallback;
#endif
  }

  using SubScalarFunc = void (*)(T*, T, size_t);

  static auto GetSubScalarFunc() -> SubScalarFunc {
#ifdef SUPPORTS_AVX2
    return SubScalarAvx2;
#elif defined(SUPPORTS_AVX)
    return SubScalarAvx;
#elif defined(SUPPORTS_SSE4_2)
    return SubScalarSse42;
#elif defined(SUPPORTS_SSE4_1)
    return SubScalarSse41;
#elif defined(SUPPORTS_SSSE3)
    return SubScalarSsse3;
#elif defined(SUPPORTS_SSE3)
    return SubScalarSse3;
#else
    return SubScalarFallback;
#endif
  }

  template <Options Option>
  using MulFunc = void (*)(
      T*, const T*, const T*, const size_t, const size_t, const size_t);

  template <Options Option>
  static auto GetMulFunc() -> MulFunc<Option> {
#ifdef SUPPORTS_AVX2
    return MulAvx2<Option>;
#elif defined(SUPPORTS_AVX)
    return MulAvx<Option>;
#elif defined(SUPPORTS_SSE4_2)
    return MulSse42<Option>;
#elif defined(SUPPORTS_SSE4_1)
    return MulSse41<Option>;
#elif defined(SUPPORTS_SSSE3)
    return MulSsse3<Option>;
#elif defined(SUPPORTS_SSE3)
    return MulSse3<Option>;
#else
    return MulFallback<Option>;
#endif
  }

  using MulScalarFunc = void (*)(T*, T, size_t);

  static auto GetMulScalarFunc() -> MulScalarFunc {
#ifdef SUPPORTS_AVX2
    return MulScalarAvx2;
#elif defined(SUPPORTS_AVX)
    return MulScalarAvx;
#elif defined(SUPPORTS_SSE4_2)
    return MulScalarSse42;
#elif defined(SUPPORTS_SSE4_1)
    return MulScalarSse41;
#elif defined(SUPPORTS_SSSE3)
    return MulScalarSsse3;
#elif defined(SUPPORTS_SSE3)
    return MulScalarSse3;
#else
    return MulScalarFallback;
#endif
  }

  using DivScalarFunc = void (*)(T*, T, size_t);

  static auto GetDivScalarFunc() -> DivScalarFunc {
#ifdef SUPPORTS_AVX2
    return DivScalarAvx2;
#elif defined(SUPPORTS_AVX)
    return DivScalarAvx;
#elif defined(SUPPORTS_SSE4_2)
    return DivScalarSse42;
#elif defined(SUPPORTS_SSE4_1)
    return DivScalarSse41;
#elif defined(SUPPORTS_SSSE3)
    return DivScalarSsse3;
#elif defined(SUPPORTS_SSE3)
    return DivScalarSse3;
#else
    return DivScalarFallback;
#endif
  }

  private:
  // BEGIN: add two arrays
  //----------------------------------------------------------------------------

  static void AddAvx2(T* a, const T* b, size_t size) {
    AddFallback(a, b, size);
  }

  static void AddAvx(T* a, const T* b, size_t size) { AddFallback(a, b, size); }

  static void AddSse42(T* a, const T* b, size_t size) {
    AddFallback(a, b, size);
  }

  static void AddSse41(T* a, const T* b, size_t size) {
    AddFallback(a, b, size);
  }

  static void AddSsse3(T* a, const T* b, size_t size) {
    AddFallback(a, b, size);
  }

  static void AddSse3(T* a, const T* b, size_t size) {
    AddFallback(a, b, size);
  }

  static void AddFallback(T* a, const T* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] += b[i];
    }
  }

  // END: add two arrays
  //----------------------------------------------------------------------------

  // BEGIN: add scalar
  //----------------------------------------------------------------------------

  static void AddScalarAvx2(T* a, T scalar, size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarAvx(T* a, T scalar, size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarSse42(T* a, T scalar, size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarSse41(T* a, T scalar, size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarSsse3(T* a, T scalar, size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarSse3(T* a, T scalar, size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarFallback(T* a, T scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] += scalar;
    }
  }

  // END: add scalar
  //----------------------------------------------------------------------------

  // BEGIN: subtraction array
  //----------------------------------------------------------------------------

  static void SubAvx2(T* a, const T* b, size_t size) {
    SubFallback(a, b, size);
  }

  static void SubAvx(T* a, const T* b, size_t size) { SubFallback(a, b, size); }

  static void SubSse42(T* a, const T* b, size_t size) {
    SubFallback(a, b, size);
  }

  static void SubSse41(T* a, const T* b, size_t size) {
    SubFallback(a, b, size);
  }

  static void SubSsse3(T* a, const T* b, size_t size) {
    SubFallback(a, b, size);
  }

  static void SubSse3(T* a, const T* b, size_t size) {
    SubFallback(a, b, size);
  }

  static void SubFallback(T* a, const T* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  // END: subtraction array
  //----------------------------------------------------------------------------

  // BEGIN: subtraction scalar
  //----------------------------------------------------------------------------

  static void SubScalarAvx2(T* a, T scalar, size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarAvx(T* a, T scalar, size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarSse42(T* a, T scalar, size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarSse41(T* a, T scalar, size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarSsse3(T* a, T scalar, size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarSse3(T* a, T scalar, size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarFallback(T* a, T scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  // END: subtraction scalar
  //----------------------------------------------------------------------------

  // BEGIN: multiplication array
  //----------------------------------------------------------------------------

  template <Options Option>
  static void MulAvx2(T* result, const T* a, const T* b, size_t size) {
    MulFallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void MulAvx(T* result, const T* a, const T* b, size_t size) {
    MulFallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void MulSse42(T* result, const T* a, const T* b, size_t size) {
    MulFallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void MulSse41(T* result, const T* a, const T* b, size_t size) {
    MulFallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void MulSsse3(T* result, const T* a, const T* b, size_t size) {
    MulFallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void MulSse3(T* result, const T* a, const T* b, size_t size) {
    MulFallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void MulFallback(
      T* result, const T* a, const T* b, size_t size, size_t dim) {
    if constexpr (Option == Options::ColumnMajor) {
      for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
          T sum = 0;
          for (size_t k = 0; k < dim; ++k) {
            sum += a[i + k * dim] * b[k + j * dim];
          }
          result[i + j * dim] = sum;
        }
      }
    } else if constexpr (Option == Options::RowMajor) {
      for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
          T sum = 0;
          for (size_t k = 0; k < dim; ++k) {
            sum += a[i * dim + k] * b[k * dim + j];
          }
          result[i * dim + j] = sum;
        }
      }
    }
  }

  // END: multiplication array
  //----------------------------------------------------------------------------

  // BEGIN: multiplication scalar
  //----------------------------------------------------------------------------

  static void MulScalarAvx2(T* a, T scalar, size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarAvx(T* a, T scalar, size_t size) {}

  static void MulScalarSse42(T* a, T scalar, size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarSse41(T* a, T scalar, size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarSsse3(T* a, T scalar, size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarSse3(T* a, T scalar, size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarFallback(T* a, T scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  // END: multiplication scalar
  //----------------------------------------------------------------------------

  // BEGIN: division scalar
  //----------------------------------------------------------------------------

  static void DivScalarAvx2(T* a, T scalar, size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarAvx(T* a, T scalar, size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarSse42(T* a, T scalar, size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarSse41(T* a, T scalar, size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarSsse3(T* a, T scalar, size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarSse3(T* a, T scalar, size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarFallback(T* a, T scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  // END: division scalar
  //----------------------------------------------------------------------------
};

}  // namespace math

#include "InstructionSetDouble.h"
#include "InstructionSetFloat.h"
#include "InstructionSetInt.h"
