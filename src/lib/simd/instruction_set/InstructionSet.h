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
concept SimdSupportedType
    = std::is_same<T, int>::value || std::is_same<T, float>::value
   || std::is_same<T, double>::value;

template <typename T>
class InstructionSet {
  public:
  static_assert(SimdSupportedType<T>,
                "InstructionSet supports only int, float, and double types");

  using AddFunc = void (*)(T*, const T*, size_t);

  static AddFunc getAddFunc() {
#ifdef SUPPORTS_AVX2
    return add_avx2;
#elif defined(SUPPORTS_AVX)
    return add_avx;
#elif defined(SUPPORTS_SSE4_2)
    return add_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
    return add_sse4_1;
#elif defined(SUPPORTS_SSSE3)
    return add_ssse3;
#elif defined(SUPPORTS_SSE3)
    return add_sse3;
#else
    return add_fallback;
#endif
  }

  using AddScalarFunc = void (*)(T*, T, size_t);

  static AddScalarFunc getAddScalarFunc() {
#ifdef SUPPORTS_AVX2
    return add_scalar_avx2;
#elif defined(SUPPORTS_AVX)
    return add_scalar_avx;
#elif defined(SUPPORTS_SSE4_2)
    return add_scalar_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
    return add_scalar_sse4_1;
#elif defined(SUPPORTS_SSSE3)
    return add_scalar_ssse3;
#elif defined(SUPPORTS_SSE3)
    return add_scalar_sse3;
#else
    return add_scalar_fallback;
#endif
  }

  using SubFunc = void (*)(T*, const T*, size_t);

  static SubFunc getSubFunc() {
#ifdef SUPPORTS_AVX2
    return sub_avx2;
#elif defined(SUPPORTS_AVX)
    return sub_avx;
#elif defined(SUPPORTS_SSE4_2)
    return sub_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
    return sub_sse4_1;
#elif defined(SUPPORTS_SSSE3)
    return sub_ssse3;
#elif defined(SUPPORTS_SSE3)
    return sub_sse3;
#else
    return sub_fallback;
#endif
  }

  using SubScalarFunc = void (*)(T*, T, size_t);

  static SubScalarFunc getSubScalarFunc() {
#ifdef SUPPORTS_AVX2
    return sub_scalar_avx2;
#elif defined(SUPPORTS_AVX)
    return sub_scalar_avx;
#elif defined(SUPPORTS_SSE4_2)
    return sub_scalar_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
    return sub_scalar_sse4_1;
#elif defined(SUPPORTS_SSSE3)
    return sub_scalar_ssse3;
#elif defined(SUPPORTS_SSE3)
    return sub_scalar_sse3;
#else
    return sub_scalar_fallback;
#endif
  }

  template <Options Option>
  using MulFunc = void (*)(T*, const T*, const T*, size_t);

  template <Options Option>
  static MulFunc<Option> getMulFunc() {
#ifdef SUPPORTS_AVX2
    return mul_avx2<Option>;
#elif defined(SUPPORTS_AVX)
    return mul_avx<Option>;
#elif defined(SUPPORTS_SSE4_2)
    return mul_sse4_2<Option>;
#elif defined(SUPPORTS_SSE4_1)
    return mul_sse4_1<Option>;
#elif defined(SUPPORTS_SSSE3)
    return mul_ssse3<Option>;
#elif defined(SUPPORTS_SSE3)
    return mul_sse3<Option>;
#else
    return mul_fallback<Option>;
#endif
  }

  using MulScalarFunc = void (*)(T*, T, size_t);

  static MulScalarFunc getMulScalarFunc() {
#ifdef SUPPORTS_AVX2
    return mul_scalar_avx2;
#elif defined(SUPPORTS_AVX)
    return mul_scalar_avx;
#elif defined(SUPPORTS_SSE4_2)
    return mul_scalar_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
    return mul_scalar_sse4_1;
#elif defined(SUPPORTS_SSSE3)
    return mul_scalar_ssse3;
#elif defined(SUPPORTS_SSE3)
    return mul_scalar_sse3;
#else
    return mul_scalar_fallback;
#endif
  }

  using DivScalarFunc = void (*)(T*, T, size_t);

  static DivScalarFunc getDivScalarFunc() {
#ifdef SUPPORTS_AVX2
    return div_scalar_avx2;
#elif defined(SUPPORTS_AVX)
    return div_scalar_avx;
#elif defined(SUPPORTS_SSE4_2)
    return div_scalar_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
    return div_scalar_sse4_1;
#elif defined(SUPPORTS_SSSE3)
    return div_scalar_ssse3;
#elif defined(SUPPORTS_SSE3)
    return div_scalar_sse3;
#else
    return div_scalar_fallback;
#endif
  }

  private:
  // BEGIN: add two arrays
  //----------------------------------------------------------------------------

  static void add_avx2(T* a, const T* b, size_t size) {
    add_fallback(a, b, size);
  }

  static void add_avx(T* a, const T* b, size_t size) {
    add_fallback(a, b, size);
  }

  static void add_sse4_2(T* a, const T* b, size_t size) {
    add_fallback(a, b, size);
  }

  static void add_sse4_1(T* a, const T* b, size_t size) {
    add_fallback(a, b, size);
  }

  static void add_ssse3(T* a, const T* b, size_t size) {
    add_fallback(a, b, size);
  }

  static void add_sse3(T* a, const T* b, size_t size) {
    add_fallback(a, b, size);
  }

  static void add_fallback(T* a, const T* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] += b[i];
    }
  }

  // END: add two arrays
  //----------------------------------------------------------------------------

  // BEGIN: add scalar
  //----------------------------------------------------------------------------

  static void add_scalar_avx2(T* a, T scalar, size_t size) {
    add_scalar_fallback(a, scalar, size);
  }

  static void add_scalar_avx(T* a, T scalar, size_t size) {
    add_scalar_fallback(a, scalar, size);
  }

  static void add_scalar_sse4_2(T* a, T scalar, size_t size) {
    add_scalar_fallback(a, scalar, size);
  }

  static void add_scalar_sse4_1(T* a, T scalar, size_t size) {
    add_scalar_fallback(a, scalar, size);
  }

  static void add_scalar_ssse3(T* a, T scalar, size_t size) {
    add_scalar_fallback(a, scalar, size);
  }

  static void add_scalar_sse3(T* a, T scalar, size_t size) {
    add_scalar_fallback(a, scalar, size);
  }

  static void add_scalar_fallback(T* a, T scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] += scalar;
    }
  }

  // END: add scalar
  //----------------------------------------------------------------------------

  // BEGIN: subtraction array
  //----------------------------------------------------------------------------

  static void sub_avx2(T* a, const T* b, size_t size) {
    sub_fallback(a, b, size);
  }

  static void sub_avx(T* a, const T* b, size_t size) {
    sub_fallback(a, b, size);
  }

  static void sub_sse4_2(T* a, const T* b, size_t size) {
    sub_fallback(a, b, size);
  }

  static void sub_sse4_1(T* a, const T* b, size_t size) {
    sub_fallback(a, b, size);
  }

  static void sub_ssse3(T* a, const T* b, size_t size) {
    sub_fallback(a, b, size);
  }

  static void sub_sse3(T* a, const T* b, size_t size) {
    sub_fallback(a, b, size);
  }

  static void sub_fallback(T* a, const T* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  // END: subtraction array
  //----------------------------------------------------------------------------

  // BEGIN: subtraction scalar
  //----------------------------------------------------------------------------

  static void sub_scalar_avx2(T* a, T scalar, size_t size) {
    sub_scalar_fallback(a, scalar, size);
  }

  static void sub_scalar_avx(T* a, T scalar, size_t size) {
    sub_scalar_fallback(a, scalar, size);
  }

  static void sub_scalar_sse4_2(T* a, T scalar, size_t size) {
    sub_scalar_fallback(a, scalar, size);
  }

  static void sub_scalar_sse4_1(T* a, T scalar, size_t size) {
    sub_scalar_fallback(a, scalar, size);
  }

  static void sub_scalar_ssse3(T* a, T scalar, size_t size) {
    sub_scalar_fallback(a, scalar, size);
  }

  static void sub_scalar_sse3(T* a, T scalar, size_t size) {
    sub_scalar_fallback(a, scalar, size);
  }

  static void sub_scalar_fallback(T* a, T scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  // END: subtraction scalar
  //----------------------------------------------------------------------------

  // BEGIN: multiplication array
  //----------------------------------------------------------------------------

  template <Options Option>
  static void mul_avx2(T* result, const T* a, const T* b, size_t size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_avx(T* result, const T* a, const T* b, size_t size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_sse4_2(T* result, const T* a, const T* b, size_t size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_sse4_1(T* result, const T* a, const T* b, size_t size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_ssse3(T* result, const T* a, const T* b, size_t size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_sse3(T* result, const T* a, const T* b, size_t size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_fallback(
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

  static void mul_scalar_avx2(T* a, T scalar, size_t size) {
    mul_scalar_fallback(a, scalar, size);
  }

  static void mul_scalar_avx(T* a, T scalar, size_t size) {}

  static void mul_scalar_sse4_2(T* a, T scalar, size_t size) {
    mul_scalar_fallback(a, scalar, size);
  }

  static void mul_scalar_sse4_1(T* a, T scalar, size_t size) {
    mul_scalar_fallback(a, scalar, size);
  }

  static void mul_scalar_ssse3(T* a, T scalar, size_t size) {
    mul_scalar_fallback(a, scalar, size);
  }

  static void mul_scalar_sse3(T* a, T scalar, size_t size) {
    mul_scalar_fallback(a, scalar, size);
  }

  static void mul_scalar_fallback(T* a, T scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  // END: multiplication scalar
  //----------------------------------------------------------------------------

  // BEGIN: division scalar
  //----------------------------------------------------------------------------

  static void div_scalar_avx2(T* a, T scalar, size_t size) {
    div_scalar_fallback(a, scalar, size);
  }

  static void div_scalar_avx(T* a, T scalar, size_t size) {
    div_scalar_fallback(a, scalar, size);
  }

  static void div_scalar_sse4_2(T* a, T scalar, size_t size) {
    div_scalar_fallback(a, scalar, size);
  }

  static void div_scalar_sse4_1(T* a, T scalar, size_t size) {
    div_scalar_fallback(a, scalar, size);
  }

  static void div_scalar_ssse3(T* a, T scalar, size_t size) {
    div_scalar_fallback(a, scalar, size);
  }

  static void div_scalar_sse3(T* a, T scalar, size_t size) {
    div_scalar_fallback(a, scalar, size);
  }

  static void div_scalar_fallback(T* a, T scalar, size_t size) {
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
