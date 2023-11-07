/**
 * @file InstructionSetInt.h
 */

#pragma once

#include "../../options/Options.h"

#include <immintrin.h>

namespace math {
template <typename T>
class InstructionSet;

template <>
class InstructionSet<int> {
  public:
  using AddFunc = void (*)(int*, const int*, size_t);

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

  using AddScalarFunc = void (*)(int*, int, size_t);

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

  using SubFunc = void (*)(int*, const int*, size_t);

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

  using SubScalarFunc = void (*)(int*, int, size_t);

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
  using MulFunc = void (*)(int*, const int*, const int*, size_t);

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

  using MulScalarFunc = void (*)(int*, int, size_t);

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

  using DivScalarFunc = void (*)(int*, int, size_t);

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
  static constexpr size_t AVX_SIMD_WIDTH = 8;
  static constexpr size_t SSE_SIMD_WIDTH = 4;

  // BEGIN: add two arrays
  //----------------------------------------------------------------------------

  static void add_avx2(int* a, const int* b, size_t size) {
    add_avx(a, b, size);
  }

  static void add_avx(int* a, const int* b, size_t size) {
    const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
      __m256i ymm1 = _mm256_loadu_si256((const __m256i*)(a + i));
      __m256i ymm2 = _mm256_loadu_si256((const __m256i*)(b + i));
      ymm1         = _mm256_add_epi32(ymm1, ymm2);
      _mm256_storeu_si256((__m256i*)(a + i), ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += b[i];
    }
  }

  static void add_sse4_2(int* a, const int* b, size_t size) {
    add_sse3(a, b, size);
  }

  static void add_sse4_1(int* a, const int* b, size_t size) {
    add_sse3(a, b, size);
  }

  static void add_ssse3(int* a, const int* b, size_t size) {
    add_sse3(a, b, size);
  }

  static void add_sse3(int* a, const int* b, size_t size) {
    const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128i xmm1 = _mm_loadu_si128((const __m128i*)(a + i));
      __m128i xmm2 = _mm_loadu_si128((const __m128i*)(b + i));
      xmm1         = _mm_add_epi32(xmm1, xmm2);
      _mm_storeu_si128((__m128i*)(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += b[i];
    }
  }

  static void add_fallback(int* a, const int* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] += b[i];
    }
  }

  // END: add two arrays
  //----------------------------------------------------------------------------

  // BEGIN: add scalar
  //----------------------------------------------------------------------------

  static void add_scalar_avx2(int* a, int scalar, size_t size) {
    add_scalar_avx(a, scalar, size);
  }

  static void add_scalar_avx(int* a, int scalar, size_t size) {
    __m256i      ymm0      = _mm256_set1_epi32(scalar);
    const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
      __m256i ymm1 = _mm256_loadu_si256((const __m256i*)(a + i));
      ymm1         = _mm256_add_epi32(ymm1, ymm0);
      _mm256_storeu_si256((__m256i*)(a + i), ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void add_scalar_sse4_2(int* a, int scalar, size_t size) {
    add_scalar_sse3(a, scalar, size);
  }

  static void add_scalar_sse4_1(int* a, int scalar, size_t size) {
    add_scalar_sse3(a, scalar, size);
  }

  static void add_scalar_ssse3(int* a, int scalar, size_t size) {
    add_scalar_sse3(a, scalar, size);
  }

  static void add_scalar_sse3(int* a, int scalar, size_t size) {
    __m128i      xmm0      = _mm_set1_epi32(scalar);
    const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128i xmm1 = _mm_loadu_si128((const __m128i*)(a + i));
      xmm1         = _mm_add_epi32(xmm1, xmm0);
      _mm_storeu_si128((__m128i*)(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void add_scalar_fallback(int* a, int scalar, size_t size) {
    // no SIMD
    for (size_t i = 0; i < size; ++i) {
      a[i] += scalar;
    }
  }

  // END: add scalar
  //----------------------------------------------------------------------------

  // BEGIN: subtract two arrays
  //----------------------------------------------------------------------------

  static void sub_avx2(int* a, const int* b, size_t size) {
    sub_avx(a, b, size);
  }

  static void sub_avx(int* a, const int* b, size_t size) {
    const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
      __m256i ymm1 = _mm256_loadu_si256((const __m256i*)(a + i));
      __m256i ymm2 = _mm256_loadu_si256((const __m256i*)(b + i));
      ymm1         = _mm256_sub_epi32(ymm1, ymm2);
      _mm256_storeu_si256((__m256i*)(a + i), ymm1);
    }

    // Handling remaining elements
    for (; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  static void sub_sse4_2(int* a, const int* b, size_t size) {
    sub_sse3(a, b, size);
  }

  static void sub_sse4_1(int* a, const int* b, size_t size) {
    sub_sse3(a, b, size);
  }

  static void sub_ssse3(int* a, const int* b, size_t size) {
    sub_sse3(a, b, size);
  }

  static void sub_sse3(int* a, const int* b, size_t size) {
    const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128i xmm1 = _mm_loadu_si128((const __m128i*)(a + i));
      __m128i xmm2 = _mm_loadu_si128((const __m128i*)(b + i));
      xmm1         = _mm_sub_epi32(xmm1, xmm2);
      _mm_storeu_si128((__m128i*)(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  static void sub_fallback(int* a, const int* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  // END: subtract two arrays
  //----------------------------------------------------------------------------

  static void sub_scalar_avx2(int* a, int scalar, size_t size) {
    sub_scalar_avx(a, scalar, size);
  }

  static void sub_scalar_avx(int* a, int scalar, size_t size) {
    __m256i      ymm0      = _mm256_set1_epi32(scalar);
    const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
      __m256i ymm1 = _mm256_loadu_si256((const __m256i*)(a + i));
      ymm1         = _mm256_sub_epi32(ymm1, ymm0);
      _mm256_storeu_si256((__m256i*)(a + i), ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void sub_scalar_sse4_2(int* a, int scalar, size_t size) {
    sub_scalar_sse3(a, scalar, size);
  }

  static void sub_scalar_sse4_1(int* a, int scalar, size_t size) {
    sub_scalar_sse3(a, scalar, size);
  }

  static void sub_scalar_ssse3(int* a, int scalar, size_t size) {
    sub_scalar_sse3(a, scalar, size);
  }

  static void sub_scalar_sse3(int* a, int scalar, size_t size) {
    __m128i      xmm0      = _mm_set1_epi32(scalar);
    const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128i xmm1 = _mm_loadu_si128((const __m128i*)(a + i));
      xmm1         = _mm_sub_epi32(xmm1, xmm0);
      _mm_storeu_si128((__m128i*)(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void sub_scalar_fallback(int* a, int scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  // END: subtract scalar
  //----------------------------------------------------------------------------

  // BEGIN: multiplication array
  //----------------------------------------------------------------------------

  template <Options Option>
  static void mul_avx2(int* result, const int* a, const int* b, size_t size) {
    float breakpoint = 0;
    // AVX2 does not support direct multiplication of 32-bit integers. You may
    // need to use a workaround or fallback to SSE4.1
  }

  template <Options Option>
  static void mul_avx(int* result, const int* a, const int* b, size_t size) {
    // AVX does not support direct multiplication of 32-bit integers. You may
    // need to use a workaround or fallback to SSE4.1
  }

  template <Options Option>
  static void mul_sse4_2(int* result, const int* a, const int* b, size_t size) {
    size_t aligned_size = (size / SSE_SIMD_WIDTH) * SSE_SIMD_WIDTH;
    size_t i            = 0;

    for (i = 0; i < aligned_size; i += SSE_SIMD_WIDTH) {
      __m128i xmm1 = _mm_loadu_si128((const __m128i*)(a + i));
      __m128i xmm2 = _mm_loadu_si128((const __m128i*)(b + i));
      xmm1         = _mm_mullo_epi32(xmm1, xmm2);
      _mm_storeu_si128((__m128i*)(result + i), xmm1);
    }

    // Handling remaining elements
    for (; i < size; ++i) {
      result[i] = a[i] * b[i];
    }
  }

  template <Options Option>
  static void mul_sse4_1(int* result, const int* a, const int* b, size_t size) {
    mul_sse4_2(result, a, b, size);
  }

  template <Options Option>
  static void mul_ssse3(int* result, const int* a, const int* b, size_t size) {
    // SSSE3 does not support direct multiplication of 32-bit integers. You may
    // need to use a workaround or fallback to SSE4.1
  }

  template <Options Option>
  static void mul_sse3(int* result, const int* a, const int* b, size_t size) {
    // SSE3 does not support direct multiplication of 32-bit integers. You may
    // need to use a workaround or fallback to SSE4.1
  }

  template <Options Option>
  static void mul_fallback(
      int* result, const int* a, const int* b, size_t size, size_t dim) {
    if constexpr (Option == Options::ColumnMajor) {
      for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
          int sum = 0;
          for (size_t k = 0; k < dim; ++k) {
            sum += a[i + k * dim] * b[k + j * dim];
          }
          result[i + j * dim] = sum;
        }
      }
    } else if constexpr (Option == Options::RowMajor) {
      for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
          int sum = 0;
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

  static void mul_scalar_avx2(int* a, int scalar, size_t size) {
    __m256i      ymm0       = _mm256_set1_epi32(scalar);
    const size_t avx2_limit = size - (size % AVX_SIMD_WIDTH);
    size_t       i          = 0;

    for (; i < avx2_limit; i += AVX_SIMD_WIDTH) {
      __m256i ymm1 = _mm256_loadu_si256((__m256i*)(a + i));
      ymm1         = _mm256_mullo_epi32(ymm1, ymm0);
      _mm256_storeu_si256((__m256i*)(a + i), ymm1);
    }

    // Handle any remaining values that didn't fit into the last AVX2 vector
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void mul_scalar_avx(int* a, int scalar, size_t size) {
    mul_scalar_sse4_2(a, scalar, size);  // downgrade to SSE 4.2 since AVX does
                                         // not support direct multiplication of
                                         // 32-bit integers
  }

  static void mul_scalar_sse4_2(int* a, int scalar, size_t size) {
    mul_scalar_sse4_1(a, scalar, size);
  }

  static void mul_scalar_sse4_1(int* a, int scalar, size_t size) {
    __m128i      xmm0      = _mm_set1_epi32(scalar);
    const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128i xmm1 = _mm_loadu_si128((const __m128i*)(a + i));
      xmm1         = _mm_mullo_epi32(xmm1, xmm0);
      _mm_storeu_si128((__m128i*)(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void mul_scalar_ssse3(int* a, int scalar, size_t size) {
    // SSSE3 does not include instructions for multiplying a vector of 32-bit
    // integers by a scalar.
    mul_scalar_fallback(a, scalar, size);
  }

  static void mul_scalar_sse3(int* a, int scalar, size_t size) {
    // SSE3 does not include instructions for multiplying a vector of 32-bit
    // integers by a scalar.
    mul_scalar_fallback(a, scalar, size);
  }

  static void mul_scalar_fallback(int* a, int scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  // END: multiplication scalar
  //----------------------------------------------------------------------------

  // BEGIN: division scalar
  //----------------------------------------------------------------------------

  // Note: SIMD instructions do not support integer division. So we use a loop
  // for this operation.

  static void div_scalar_avx2(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    div_scalar_avx(a, scalar, size);
#else
    div_scalar_fallback(a, scalar, size);
#endif
  }

  static void div_scalar_avx(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
    __m256i      ymm0      = _mm256_set1_epi32(scalar);
    size_t       i         = 0;

    for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
      __m256i ymm1 = _mm256_loadu_epi32(a + i);
      ymm1         = _mm256_div_epi32(ymm1, ymm0);
      _mm256_storeu_epi32(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] /= scalar;
    }
#else
    div_scalar_fallback(a, scalar, size);
#endif
  }

  static void div_scalar_sse4_2(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    div_scalar_sse3(a, scalar, size);
#else
    div_scalar_fallback(a, scalar, size);
#endif
  }

  static void div_scalar_sse4_1(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    div_scalar_sse3(a, scalar, size);
#else
    div_scalar_fallback(a, scalar, size);
#endif
  }

  static void div_scalar_ssse3(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    div_scalar_sse3(a, scalar, size);
#else
    div_scalar_fallback(a, scalar, size);
#endif
  }

  static void div_scalar_sse3(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
    __m128i      xmm0      = _mm_set1_epi32(scalar);
    size_t       i         = 0;

    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128i xmm1 = _mm_loadu_epi32(a + i);
      xmm1         = _mm_div_epi32(xmm1, xmm0);
      _mm_storeu_epi32(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] /= scalar;
    }
#else
    div_scalar_fallback(a, scalar, size);
#endif
  }

  static void div_scalar_fallback(int* a, int scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  // END: division scalar
  //----------------------------------------------------------------------------
};

}  // namespace math
