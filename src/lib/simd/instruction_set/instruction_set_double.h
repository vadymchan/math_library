/**
 * @file instruction_set_double.h
 */

#ifndef MATH_LIBRARY_INSTRUCTION_SET_DOUBLE_H
#define MATH_LIBRARY_INSTRUCTION_SET_DOUBLE_H

#include "../../options/options.h"
#include "../precompiled/simd_defines.h"

#include <immintrin.h>

#include <cstddef>

namespace math {

template <typename T>
class InstructionSet;

template <>
class InstructionSet<double> {
  public:
  using AddFunc = void (*)(double*, const double*, std::size_t);

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

  using AddScalarFunc = void (*)(double*, double, std::size_t);

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

  using SubFunc = void (*)(double*, const double*, std::size_t);

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

  using SubScalarFunc = void (*)(double*, double, std::size_t);

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

  using NegFunc = void (*)(double*, std::size_t);

  static auto GetNegFunc() -> NegFunc {
#ifdef SUPPORTS_AVX2
    return NegAvx2;
#elif defined(SUPPORTS_AVX)
    return NegAvx;
#elif defined(SUPPORTS_SSE4_2)
    return NegSse42;
#elif defined(SUPPORTS_SSE4_1)
    return NegSse41;
#elif defined(SUPPORTS_SSSE3)
    return NegSsse3;
#elif defined(SUPPORTS_SSE3)
    return NegSse3;
#else
    return NegFallback;
#endif
  }

  template <Options Option>
  using MulFunc = void (*)(double*,
                           const double*,
                           const double*,
                           const std::size_t,
                           const std::size_t,
                           const std::size_t);

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

  using MulScalarFunc = void (*)(double*, double, std::size_t);

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

  using DivScalarFunc = void (*)(double*, double, std::size_t);

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

  using CmpFunc = int (*)(const double*, const double*, std::size_t);

  static auto GetCmpFunc() -> CmpFunc {
#ifdef SUPPORTS_AVX2
    return CmpAvx2;
#elif defined(SUPPORTS_AVX)
    return CmpAvx;
#elif defined(SUPPORTS_SSE4_2)
    return CmpSse42;
#elif defined(SUPPORTS_SSE4_1)
    return CmpSse41;
#elif defined(SUPPORTS_SSSE3)
    return CmpSsse3;
#elif defined(SUPPORTS_SSE3)
    return CmpSse3;
#else
    return CmpFallback;
#endif
  }

  private:
  static constexpr std::size_t s_kAvxSimdWidth
      = sizeof(__m256d) / sizeof(double);  // 4
  static constexpr std::size_t s_kSseSimdWidth
      = sizeof(__m128d) / sizeof(double);  // 2

  // BEGIN: add two arrays
  //----------------------------------------------------------------------------

  static void AddAvx2(double* a, const double* b, std::size_t size) {
    AddAvx(a, b, size);
  }

  static void AddAvx(double* a, const double* b, std::size_t size) {
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
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

  static void AddSse3(double* a, const double* b, std::size_t size) {
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
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

  static void AddFallback(double* a, const double* b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] += b[i];
    }
  }

  // END: add two arrays
  //----------------------------------------------------------------------------

  // BEGIN: add scalar
  //----------------------------------------------------------------------------

  static void AddScalarAvx2(double* a, double scalar, std::size_t size) {
    AddScalarAvx(a, scalar, size);
  }

  static void AddScalarAvx(double* a, double scalar, std::size_t size) {
    __m256d      ymm0      = _mm256_set1_pd(scalar);
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256d ymm1 = _mm256_loadu_pd(a + i);
      ymm1         = _mm256_add_pd(ymm1, ymm0);
      _mm256_storeu_pd(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void AddScalarSse3(double* a, double scalar, std::size_t size) {
    __m128d      xmm0 = _mm_set1_pd(scalar);
    const std::size_t kSseLimit
        = size - (size % s_kSseSimdWidth);  // Compute the limit for SSE3 loop
    std::size_t i = 0;

    // Process full SSE3 widths
    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128d xmm1 = _mm_loadu_pd(a + i);
      xmm1         = _mm_add_pd(xmm1, xmm0);
      _mm_storeu_pd(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void AddScalarFallback(double* a, double scalar, std::size_t size) {
    // no SIMD
    for (std::size_t i = 0; i < size; ++i) {
      a[i] += scalar;
    }
  }

  // END: add scalar
  //----------------------------------------------------------------------------

  // BEGIN: subtract two arrays
  //----------------------------------------------------------------------------

  static void SubAvx2(double* a, const double* b, std::size_t size) {
    SubAvx(a, b, size);
  }

  static void SubAvx(double* a, const double* b, std::size_t size) {
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
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

  static void SubSse3(double* a, const double* b, std::size_t size) {
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
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

  static void SubFallback(double* a, const double* b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  // END: subtract two arrays
  //----------------------------------------------------------------------------

  // BEGIN: subtract scalar
  //----------------------------------------------------------------------------

  static void SubScalarAvx2(double* a, double scalar, std::size_t size) {
    SubScalarAvx(a, scalar, size);
  }

  static void SubScalarAvx(double* a, double scalar, std::size_t size) {
    __m256d      ymm0      = _mm256_set1_pd(scalar);
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256d ymm1 = _mm256_loadu_pd(a + i);
      ymm1         = _mm256_sub_pd(ymm1, ymm0);
      _mm256_storeu_pd(a + i, ymm1);
    }

    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void SubScalarSse3(double* a, double scalar, std::size_t size) {
    __m128d      xmm0      = _mm_set1_pd(scalar);
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128d xmm1 = _mm_loadu_pd(a + i);
      xmm1         = _mm_sub_pd(xmm1, xmm0);
      _mm_storeu_pd(a + i, xmm1);
    }

    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void SubScalarFallback(double* a, double scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  // END: subtract scalar
  //----------------------------------------------------------------------------

  // BEGIN: negation array
  //----------------------------------------------------------------------------

  static void NegAvx2(double* a, std::size_t size) { NegAvx(a, size); }

  static void NegAvx(double* a, std::size_t size) {
    __m256d      negZero   = _mm256_set1_pd(-0.0);
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256d ymm1 = _mm256_loadu_pd(a + i);
      ymm1         = _mm256_xor_pd(ymm1, negZero);
      _mm256_storeu_pd(a + i, ymm1);
    }

    // Handle any remaining elements without SIMD
    for (; i < size; ++i) {
      a[i] = -a[i];
    }
  }

  static void NegSse42(double* a, std::size_t size) { NegSse3(a, size); }

  static void NegSse41(double* a, std::size_t size) { NegSse3(a, size); }

  static void NegSsse3(double* a, std::size_t size) { NegSse3(a, size); }

  static void NegSse3(double* a, std::size_t size) {
    __m128d      negZero   = _mm_set1_pd(-0.0);
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128d xmm1 = _mm_loadu_pd(a + i);
      xmm1         = _mm_xor_pd(xmm1, negZero);
      _mm_storeu_pd(a + i, xmm1);
    }

    // Handle any remaining elements without SIMD
    for (; i < size; ++i) {
      a[i] = -a[i];
    }
  }

  // Fallback adapted for double type
  static void NegFallback(double* a, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] = -a[i];
    }
  }

  // END: negation array
  //----------------------------------------------------------------------------

  // BEGIN: multiplication array
  //----------------------------------------------------------------------------

  // BEGIN: multiplication array utility functions

  template <Options Option>
  static inline auto IndexA(const std::size_t kCurrentRowA,
                            const std::size_t kInnerIndex,
                            const std::size_t kRowsA,
                            const std::size_t kColsARowsB) -> std::size_t {
    if constexpr (Option == Options::ColumnMajor) {
      return kCurrentRowA + kInnerIndex * kRowsA;
    } else if constexpr (Option == Options::RowMajor) {
      return kCurrentRowA * kColsARowsB + kInnerIndex;
    }
  }

  template <Options Option>
  static inline auto IndexB(const std::size_t kInnerIndex,
                            const std::size_t kCurrentColB,
                            const std::size_t kColsB,
                            const std::size_t kColsARowsB) -> std::size_t {
    if constexpr (Option == Options::ColumnMajor) {
      return kInnerIndex + kCurrentColB * kColsARowsB;
    } else if constexpr (Option == Options::RowMajor) {
      return kInnerIndex * kColsB + kCurrentColB;
    }
  }

  template <Options Option>
  static inline auto IndexResult(const std::size_t kCurrentRowA,
                                 const std::size_t kCurrentColB,
                                 const std::size_t kRowsA,
                                 const std::size_t kColsB) -> std::size_t {
    if constexpr (Option == Options::ColumnMajor) {
      return kCurrentRowA + kCurrentColB * kRowsA;
    } else if constexpr (Option == Options::RowMajor) {
      return kCurrentRowA * kColsB + kCurrentColB;
    }
  }

  // BEGIN: AVX multiplication array utility functions

  template <Options Option>
  static inline auto LoadAAvx(const double* a,
                           const std::size_t  kCurrentRowA,
                           const std::size_t  kInnerIndex,
                           const std::size_t  kRowsA,
                           const std::size_t  kColsARowsB) -> __m256d {
    if constexpr (Option == Options::RowMajor) {
      return _mm256_loadu_pd(
          &a[IndexA<Option>(kCurrentRowA, kInnerIndex, kRowsA, kColsARowsB)]);
    } else {
      return _mm256_set_pd(
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 3, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 2, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 1, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex, kRowsA, kColsARowsB)]);
    }
  }

  template <Options Option>
  static inline auto LoadBAvx(const double* b,
                           const std::size_t  kInnerIndex,
                           const std::size_t  kCurrentColB,
                           const std::size_t  kColsB,
                           const std::size_t  kColsARowsB) -> __m256d {
    if constexpr (Option == Options::RowMajor) {
      return _mm256_set_pd(
          b[IndexB<Option>(kInnerIndex + 3, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 2, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 1, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]);
    } else {
      return _mm256_loadu_pd(
          &b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]);
    }
  }

  // END: AVX multiplication array utility functions

  // BEGIN: SSE multiplication array utility functions

  template <Options Option>
  static inline auto LoadASse(const double* a,
                              const std::size_t  kCurrentRowA,
                              const std::size_t  kInnerIndex,
                              const std::size_t  kRowsA,
                              const std::size_t  kColsARowsB) -> __m128d {
    if constexpr (Option == Options::RowMajor) {
      return _mm_loadu_pd(
          &a[IndexA<Option>(kCurrentRowA, kInnerIndex, kRowsA, kColsARowsB)]);
    } else {
      return _mm_set_pd(
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 1, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex, kRowsA, kColsARowsB)]);
    }
  }

  template <Options Option>
  static inline auto LoadBSse(const double* b,
                              const std::size_t  kInnerIndex,
                              const std::size_t  kCurrentColB,
                              const std::size_t  kColsB,
                              const std::size_t  kColsARowsB) -> __m128d {
    if constexpr (Option == Options::ColumnMajor) {
      return _mm_loadu_pd(
          &b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]);
    } else {
      return _mm_set_pd(
          b[IndexB<Option>(kInnerIndex + 1, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]);
    }
  }

  // END: SSE multiplication array utility functions

  template <Options Option>
  static void MulAvx2(double*       result,
                      const double* a,
                      const double* b,
                      const std::size_t  kRowsA,
                      const std::size_t  kColsB,
                      const std::size_t  kColsARowsB) {
    MulAvx<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulAvx(double*       result,
                     const double* a,
                     const double* b,
                     const std::size_t  kRowsA,
                     const std::size_t  kColsB,
                     const std::size_t  kColsARowsB) {
    for (std::size_t currentRowA = 0; currentRowA < kRowsA; ++currentRowA) {
      for (std::size_t currentColB = 0; currentColB < kColsB; ++currentColB) {
        __m256d sum        = _mm256_setzero_pd();
        std::size_t  innerIndex = 0;
        for (; innerIndex + s_kAvxSimdWidth - 1 < kColsARowsB;
             innerIndex += s_kAvxSimdWidth) {
          __m256d aVec = LoadAAvx<Option>(
              a, currentRowA, innerIndex, kRowsA, kColsARowsB);
          __m256d bVec = LoadBAvx<Option>(
              b, innerIndex, currentColB, kColsB, kColsARowsB);

          sum = _mm256_fmadd_pd(aVec, bVec, sum);
        }
        double tmp[s_kAvxSimdWidth];
        _mm256_storeu_pd(tmp, sum);
        double finalSum = 0.0;
        for (double i : tmp) {
          finalSum += i;
        }
        for (; innerIndex < kColsARowsB; ++innerIndex) {
          finalSum
              += a[IndexA<Option>(currentRowA, innerIndex, kRowsA, kColsARowsB)]
               * b[IndexB<Option>(
                   innerIndex, currentColB, kColsB, kColsARowsB)];
        }
        result[IndexResult<Option>(currentRowA, currentColB, kRowsA, kColsB)]
            = finalSum;
      }
    }
  }

  template <Options Option>
  static void MulSse42(double*       result,
                       const double* a,
                       const double* b,
                       const std::size_t  kRowsA,
                       const std::size_t  kColsB,
                       const std::size_t  kColsARowsB) {
    MulSse3<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSse41(double*       result,
                       const double* a,
                       const double* b,
                       const std::size_t  kRowsA,
                       const std::size_t  kColsB,
                       const std::size_t  kColsARowsB) {
    MulSse3<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSsse3(double*       result,
                       const double* a,
                       const double* b,
                       const std::size_t  kRowsA,
                       const std::size_t  kColsB,
                       const std::size_t  kColsARowsB) {
    MulSse3<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSse3(double*       result,
                      const double* a,
                      const double* b,
                      const std::size_t  kRowsA,
                      const std::size_t  kColsB,
                      const std::size_t  kColsARowsB) {
    for (std::size_t currentRowA = 0; currentRowA < kRowsA; ++currentRowA) {
      for (std::size_t currentColB = 0; currentColB < kColsB; ++currentColB) {
        __m128d sum        = _mm_setzero_pd();
        std::size_t  innerIndex = 0;
        for (; innerIndex + s_kSseSimdWidth - 1 < kColsARowsB;
             innerIndex += s_kSseSimdWidth) {
          __m128d aVec = LoadASse<Option>(
              a, currentRowA, innerIndex, kRowsA, kColsARowsB);
          __m128d bVec = LoadBSse<Option>(
              b, innerIndex, currentColB, kColsB, kColsARowsB);
          sum = _mm_add_pd(sum, _mm_mul_pd(aVec, bVec));
        }
        double tmp[s_kSseSimdWidth];
        _mm_storeu_pd(tmp, sum);
        double finalSum = 0.0;
        for (double i : tmp) {
          finalSum += i;
        }
        for (; innerIndex < kColsARowsB; ++innerIndex) {
          finalSum
              += a[IndexA<Option>(currentRowA, innerIndex, kRowsA, kColsARowsB)]
               * b[IndexB<Option>(
                   innerIndex, currentColB, kColsB, kColsARowsB)];
        }
        result[IndexResult<Option>(currentRowA, currentColB, kRowsA, kColsB)]
            = finalSum;
      }
    }
  }

  template <Options Option>
  static void MulFallback(double*       result,
                          const double* a,
                          const double* b,
                          const std::size_t  kRowsA,
                          const std::size_t  kColsB,
                          const std::size_t  kColsARowsB) {
    for (std::size_t i = 0; i < kRowsA; ++i) {
      for (std::size_t j = 0; j < kColsB; ++j) {
        double sum = 0;
        for (std::size_t k = 0; k < kColsARowsB; ++k) {
          sum += a[IndexA<Option>(i, k, kRowsA, kColsARowsB)]
               * b[IndexB<Option>(k, j, kColsB, kColsARowsB)];
        }
        result[IndexResult<Option>(i, j, kRowsA, kColsB)] = sum;
      }
    }
  }

  // END: multiplication array
  //----------------------------------------------------------------------------

  // BEGIN: multiplication scalar
  //----------------------------------------------------------------------------

  static void MulScalarAvx2(double* a, double scalar, std::size_t size) {
    MulScalarAvx(a, scalar, size);
  }

  static void MulScalarAvx(double* a, double scalar, std::size_t size) {
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    __m256d      ymm0      = _mm256_set1_pd(scalar);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256d ymm1 = _mm256_loadu_pd(a + i);
      ymm1         = _mm256_mul_pd(ymm1, ymm0);
      _mm256_storeu_pd(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void MulScalarSse42(double* a, double scalar, std::size_t size) {
    MulScalarSse3(a, scalar, size);
  }

  static void MulScalarSse41(double* a, double scalar, std::size_t size) {
    MulScalarSse3(a, scalar, size);
  }

  static void MulScalarSsse3(double* a, double scalar, std::size_t size) {
    MulScalarSse3(a, scalar, size);
  }

  static void MulScalarSse3(double* a, double scalar, std::size_t size) {
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    __m128d      xmm0      = _mm_set1_pd(scalar);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128d xmm1 = _mm_loadu_pd(a + i);
      xmm1         = _mm_mul_pd(xmm1, xmm0);
      _mm_storeu_pd(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void MulScalarFallback(double* a, double scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  // END: multiplication scalar
  //----------------------------------------------------------------------------

  // BEGIN: division scalar
  //----------------------------------------------------------------------------

  static void DivScalarAvx2(double* a, double scalar, std::size_t size) {
    DivScalarAvx(a, scalar, size);
  }

  static void DivScalarAvx(double* a, double scalar, std::size_t size) {
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    __m256d      ymm0      = _mm256_set1_pd(scalar);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256d ymm1 = _mm256_loadu_pd(a + i);
      ymm1         = _mm256_div_pd(ymm1, ymm0);
      _mm256_storeu_pd(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  static void DivScalarSse42(double* a, double scalar, std::size_t size) {
    DivScalarSse3(a, scalar, size);
  }

  static void DivScalarSse41(double* a, double scalar, std::size_t size) {
    DivScalarSse3(a, scalar, size);
  }

  static void DivScalarSsse3(double* a, double scalar, std::size_t size) {
    DivScalarSse3(a, scalar, size);
  }

  static void DivScalarSse3(double* a, double scalar, std::size_t size) {
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    __m128d      xmm0      = _mm_set1_pd(scalar);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128d xmm1 = _mm_loadu_pd(a + i);
      xmm1         = _mm_div_pd(xmm1, xmm0);
      _mm_storeu_pd(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  static void DivScalarFallback(double* a, double scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  // END: division scalar
  //----------------------------------------------------------------------------

    // BEGIN: comparison array
  //----------------------------------------------------------------------------

  static int CmpAvx2(const double* a, const double* b, std::size_t size) {
    return CmpAvx(a, b, size);
  }

  static int CmpAvx(const double* a, const double* b, std::size_t size) {
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256d aVec      = _mm256_loadu_pd(a + i);
      __m256d bVec      = _mm256_loadu_pd(b + i);
      __m256d cmpResult = _mm256_cmp_pd(aVec, bVec, _CMP_LT_OQ);
      int     mask      = _mm256_movemask_pd(cmpResult);
      if (mask != 0) {
        return -1;
      }
      cmpResult = _mm256_cmp_pd(aVec, bVec, _CMP_GT_OQ);
      mask      = _mm256_movemask_pd(cmpResult);
      if (mask != 0) {
        return 1;
      }
      cmpResult = _mm256_cmp_pd(aVec, bVec, _CMP_NEQ_OQ);
      mask      = _mm256_movemask_pd(cmpResult);
      if (mask == 0) {
        return 0;
      }
    }

    // Handle any remainder
    for (; i < size; ++i) {
      if (a[i] < b[i]) {
        return -1;
      } else if (a[i] > b[i]) {
        return 1;
      }
    }
    return 0;
  }

  static int CmpSse42(const double* a, const double* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSse41(const double* a, const double* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSsse3(const double* a, const double* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSse3(const double* a, const double* b, std::size_t size) {
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128d aVec      = _mm_loadu_pd(a + i);
      __m128d bVec      = _mm_loadu_pd(b + i);
      __m128d cmpResult = _mm_cmplt_pd(aVec, bVec);
      int     mask      = _mm_movemask_pd(cmpResult);
      if (mask != 0) {
        return -1;
      }
      cmpResult = _mm_cmpgt_pd(aVec, bVec);
      mask      = _mm_movemask_pd(cmpResult);
      if (mask != 0) {
        return 1;
      }
      cmpResult = _mm_cmpeq_pd(aVec, bVec);
      mask      = _mm_movemask_pd(cmpResult);
      if (mask == 0x3) {
        return 0;
      }
    }

    // Handle any remainder
    for (; i < size; ++i) {
      if (a[i] < b[i]) {
        return -1;
      } else if (a[i] > b[i]) {
        return 1;
      }
    }
    return 0;
  }

  static int CmpFallback(const double* a, const double* b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      if (a[i] < b[i]) {
        return -1;
      } else if (a[i] > b[i]) {
        return 1;
      }
    }
    return 0;
  }

  // END: comparison array
  //----------------------------------------------------------------------------
};
}  // namespace math

#endif