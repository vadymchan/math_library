/**
 * @file InstructionSetDouble.h
 */

#pragma once

#include "../../options/Options.h"

#include <immintrin.h>

namespace math {

template <typename T>
class InstructionSet;

template <>
class InstructionSet<double> {
  public:
  using AddFunc = void (*)(double*, const double*, size_t);

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

  using AddScalarFunc = void (*)(double*, double, size_t);

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

  using SubFunc = void (*)(double*, const double*, size_t);

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

  using SubScalarFunc = void (*)(double*, double, size_t);

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
  using MulFunc = void (*)(double*, const double*, const double*, size_t);

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

  using MulScalarFunc = void (*)(double*, double, size_t);

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

  using DivScalarFunc = void (*)(double*, double, size_t);

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
  static constexpr size_t AVX_SIMD_WIDTH = 4;
  static constexpr size_t SSE_SIMD_WIDTH = 2;

  // BEGIN: add two arrays
  //----------------------------------------------------------------------------

  static void add_avx2(double* a, const double* b, size_t size) {
    add_avx(a, b, size);
  }

  static void add_avx(double* a, const double* b, size_t size) {
    const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
      __m256d ymm1 = _mm256_loadu_pd(a + i);
      __m256d ymm2 = _mm256_loadu_pd(b + i);
      ymm1         = _mm256_add_pd(ymm1, ymm2);
      _mm256_storeu_pd(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += b[i];
    }
  }

  static void add_sse3(double* a, const double* b, size_t size) {
    const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128d xmm1 = _mm_loadu_pd(a + i);
      __m128d xmm2 = _mm_loadu_pd(b + i);
      xmm1         = _mm_add_pd(xmm1, xmm2);
      _mm_storeu_pd(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += b[i];
    }
  }

  static void add_fallback(double* a, const double* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] += b[i];
    }
  }

  // END: add two arrays
  //----------------------------------------------------------------------------

  // BEGIN: add scalar
  //----------------------------------------------------------------------------

  static void add_scalar_avx2(double* a, double scalar, size_t size) {
    add_scalar_avx(a, scalar, size);
  }

  static void add_scalar_avx(double* a, double scalar, size_t size) {
    __m256d      ymm0      = _mm256_set1_pd(scalar);
    const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
      __m256d ymm1 = _mm256_loadu_pd(a + i);
      ymm1         = _mm256_add_pd(ymm1, ymm0);
      _mm256_storeu_pd(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void add_scalar_sse3(double* a, double scalar, size_t size) {
    __m128d      xmm0 = _mm_set1_pd(scalar);
    const size_t sse_limit
        = size - (size % SSE_SIMD_WIDTH);  // Compute the limit for SSE3 loop
    size_t i = 0;

    // Process full SSE3 widths
    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128d xmm1 = _mm_loadu_pd(a + i);
      xmm1         = _mm_add_pd(xmm1, xmm0);
      _mm_storeu_pd(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void add_scalar_fallback(double* a, double scalar, size_t size) {
    // no SIMD
    for (size_t i = 0; i < size; ++i) {
      a[i] += scalar;
    }
  }

  // END: add scalar
  //----------------------------------------------------------------------------

  // BEGIN: subtract two arrays
  //----------------------------------------------------------------------------

  static void sub_avx2(double* a, const double* b, size_t size) {
    sub_avx(a, b, size);
  }

  static void sub_avx(double* a, const double* b, size_t size) {
    const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
      __m256d ymm1 = _mm256_loadu_pd(a + i);
      __m256d ymm2 = _mm256_loadu_pd(b + i);
      ymm1         = _mm256_sub_pd(ymm1, ymm2);
      _mm256_storeu_pd(a + i, ymm1);
    }

    // Handling remaining elements
    for (; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  static void sub_sse3(double* a, const double* b, size_t size) {
    const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128d xmm1 = _mm_loadu_pd(a + i);
      __m128d xmm2 = _mm_loadu_pd(b + i);
      xmm1         = _mm_sub_pd(xmm1, xmm2);
      _mm_storeu_pd(a + i, xmm1);
    }

    // Handling remaining elements
    for (; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  static void sub_fallback(double* a, const double* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  // END: subtract two arrays
  //----------------------------------------------------------------------------

  // BEGIN: subtract scalar
  //----------------------------------------------------------------------------

  static void sub_scalar_avx2(double* a, double scalar, size_t size) {
    sub_scalar_avx(a, scalar, size);
  }

  static void sub_scalar_avx(double* a, double scalar, size_t size) {
    __m256d      ymm0      = _mm256_set1_pd(scalar);
    const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
      __m256d ymm1 = _mm256_loadu_pd(a + i);
      ymm1         = _mm256_sub_pd(ymm1, ymm0);
      _mm256_storeu_pd(a + i, ymm1);
    }

    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void sub_scalar_sse3(double* a, double scalar, size_t size) {
    __m128d      xmm0      = _mm_set1_pd(scalar);
    const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
    size_t       i         = 0;

    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128d xmm1 = _mm_loadu_pd(a + i);
      xmm1         = _mm_sub_pd(xmm1, xmm0);
      _mm_storeu_pd(a + i, xmm1);
    }

    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void sub_scalar_fallback(double* a, double scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  // END: subtract scalar
  //----------------------------------------------------------------------------

  // BEGIN: multiplication array
  //----------------------------------------------------------------------------

  template <Options Option>
  static void mul_avx2(double*       result,
                       const double* a,
                       const double* b,
                       size_t        size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_avx(double*       result,
                      const double* a,
                      const double* b,
                      size_t        size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_sse4_2(double*       result,
                         const double* a,
                         const double* b,
                         size_t        size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_sse4_1(double*       result,
                         const double* a,
                         const double* b,
                         size_t        size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_ssse3(double*       result,
                        const double* a,
                        const double* b,
                        size_t        size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_sse3(double*       result,
                       const double* a,
                       const double* b,
                       size_t        size) {
    mul_fallback<Option>(result, a, b, size);
  }

  template <Options Option>
  static void mul_fallback(double*       result,
                           const double* a,
                           const double* b,
                           size_t        size,
                           size_t        dim) {
    if constexpr (Option == Options::ColumnMajor) {
      for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
          double sum = 0;
          for (size_t k = 0; k < dim; ++k) {
            sum += a[i + k * dim] * b[k + j * dim];
          }
          result[i + j * dim] = sum;
        }
      }
    } else if constexpr (Option == Options::RowMajor) {
      for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
          double sum = 0;
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

  static void mul_scalar_avx2(double* a, double scalar, size_t size) {
    mul_scalar_avx(a, scalar, size);
  }

  static void mul_scalar_avx(double* a, double scalar, size_t size) {
    const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
    __m256d      ymm0      = _mm256_set1_pd(scalar);
    size_t       i         = 0;

    for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
      __m256d ymm1 = _mm256_loadu_pd(a + i);
      ymm1         = _mm256_mul_pd(ymm1, ymm0);
      _mm256_storeu_pd(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void mul_scalar_sse3(double* a, double scalar, size_t size) {
    const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
    __m128d      xmm0      = _mm_set1_pd(scalar);
    size_t       i         = 0;

    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128d xmm1 = _mm_loadu_pd(a + i);
      xmm1         = _mm_mul_pd(xmm1, xmm0);
      _mm_storeu_pd(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void mul_scalar_fallback(double* a, double scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  // END: multiplication scalar
  //----------------------------------------------------------------------------

  // BEGIN: division scalar
  //----------------------------------------------------------------------------

  static void div_scalar_avx2(double* a, double scalar, size_t size) {
    div_scalar_avx(a, scalar, size);
  }

  static void div_scalar_avx(double* a, double scalar, size_t size) {
    const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
    __m256d      ymm0      = _mm256_set1_pd(scalar);
    size_t       i         = 0;

    for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
      __m256d ymm1 = _mm256_loadu_pd(a + i);
      ymm1         = _mm256_div_pd(ymm1, ymm0);
      _mm256_storeu_pd(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  static void div_scalar_sse3(double* a, double scalar, size_t size) {
    const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
    __m128d      xmm0      = _mm_set1_pd(scalar);
    size_t       i         = 0;

    for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
      __m128d xmm1 = _mm_loadu_pd(a + i);
      xmm1         = _mm_div_pd(xmm1, xmm0);
      _mm_storeu_pd(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  static void div_scalar_fallback(double* a, double scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  // END: division scalar
  //----------------------------------------------------------------------------
};
}  // namespace math
