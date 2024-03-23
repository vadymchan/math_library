/**
 * @file instruction_set_float.h
 */

#ifndef MATH_LIBRARY_INSTRUCTION_SET_FLOAT_H
#define MATH_LIBRARY_INSTRUCTION_SET_FLOAT_H

#include "../../options/options.h"
#include "../precompiled/simd_defines.h"

#include <immintrin.h>

namespace math {

template <typename T>
class InstructionSet;

template <>
class InstructionSet<float> {
  public:
  using AddFunc = void (*)(float*, const float*, std::size_t);

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

  using AddScalarFunc = void (*)(float*, float, std::size_t);

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

  using SubFunc = void (*)(float*, const float*, std::size_t);

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

  using SubScalarFunc = void (*)(float*, float, std::size_t);

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

  using NegFunc = void (*)(float*, std::size_t);

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
  using MulFunc = void (*)(float*,
                           const float*,
                           const float*,
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

  using MulScalarFunc = void (*)(float*, float, std::size_t);

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

  using DivScalarFunc = void (*)(float*, float, std::size_t);

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

  using CmpFunc = int (*)(const float*, const float*, std::size_t);

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
      = sizeof(__m256) / sizeof(float);  // 8
  static constexpr std::size_t s_kSseSimdWidth
      = sizeof(__m128) / sizeof(float);  // 4

  // BEGIN: add two arrays
  //----------------------------------------------------------------------------

  static void AddAvx2(float* a, const float* b, std::size_t size) {
    AddAvx(a, b, size);
  }

  static void AddAvx(float* a, const float* b, std::size_t size) {
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256 ymm1 = _mm256_loadu_ps(a + i);
      __m256 ymm2 = _mm256_loadu_ps(b + i);
      ymm1        = _mm256_add_ps(ymm1, ymm2);
      _mm256_storeu_ps(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += b[i];
    }
  }

  static void AddSse42(float* a, const float* b, std::size_t size) {
    AddSse3(a, b, size);
  }

  static void AddSse41(float* a, const float* b, std::size_t size) {
    AddSse3(a, b, size);
  }

  static void AddSsse3(float* a, const float* b, std::size_t size) {
    AddSse3(a, b, size);
  }

  static void AddSse3(float* a, const float* b, std::size_t size) {
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128 xmm1 = _mm_loadu_ps(a + i);
      __m128 xmm2 = _mm_loadu_ps(b + i);
      xmm1        = _mm_add_ps(xmm1, xmm2);
      _mm_storeu_ps(a + i, xmm1);
    }

    // Handling remaining elements
    for (; i < size; ++i) {
      a[i] += b[i];
    }
  }

  static void AddFallback(float* a, const float* b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] += b[i];
    }
  }

  // END: add two arrays
  //----------------------------------------------------------------------------

  // BEGIN: add scalar
  //----------------------------------------------------------------------------

  static void AddScalarAvx2(float* a, float scalar, std::size_t size) {
    AddScalarAvx(a, scalar, size);
  }

  static void AddScalarAvx(float* a, float scalar, std::size_t size) {
    __m256       ymm0      = _mm256_set1_ps(scalar);
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256 ymm1 = _mm256_loadu_ps(a + i);
      ymm1        = _mm256_add_ps(ymm1, ymm0);
      _mm256_storeu_ps(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void AddScalarSse42(float* a, float scalar, std::size_t size) {
    AddScalarSse3(a, scalar, size);
  }

  static void AddScalarSse41(float* a, float scalar, std::size_t size) {
    AddScalarSse3(a, scalar, size);
  }

  static void AddScalarSsse3(float* a, float scalar, std::size_t size) {
    AddScalarSse3(a, scalar, size);
  }

  static void AddScalarSse3(float* a, float scalar, std::size_t size) {
    __m128       xmm0      = _mm_set1_ps(scalar);
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128 xmm1 = _mm_loadu_ps(a + i);
      xmm1        = _mm_add_ps(xmm1, xmm0);
      _mm_storeu_ps(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void AddScalarFallback(float* a, float scalar, std::size_t size) {
    // no SIMD
    for (std::size_t i = 0; i < size; ++i) {
      a[i] += scalar;
    }
  }

  // END: add scalar
  //----------------------------------------------------------------------------

  // BEGIN: subtract two arrays
  //----------------------------------------------------------------------------

  static void SubAvx2(float* a, const float* b, std::size_t size) {
    SubAvx(a, b, size);
  }

  static void SubAvx(float* a, const float* b, std::size_t size) {
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256 ymm1 = _mm256_loadu_ps(a + i);
      __m256 ymm2 = _mm256_loadu_ps(b + i);
      ymm1        = _mm256_sub_ps(ymm1, ymm2);
      _mm256_storeu_ps(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  static void SubSse42(float* a, const float* b, std::size_t size) {
    SubSse3(a, b, size);
  }

  static void SubSse41(float* a, const float* b, std::size_t size) {
    SubSse3(a, b, size);
  }

  static void SubSsse3(float* a, const float* b, std::size_t size) {
    SubSse3(a, b, size);
  }

  static void SubSse3(float* a, const float* b, std::size_t size) {
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128 xmm1 = _mm_loadu_ps(a + i);
      __m128 xmm2 = _mm_loadu_ps(b + i);
      xmm1        = _mm_sub_ps(xmm1, xmm2);
      _mm_storeu_ps(a + i, xmm1);
    }

    // Handle any remaining elements
    for (; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  static void SubFallback(float* a, const float* b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  // END: subtract two arrays
  //----------------------------------------------------------------------------

  static void SubScalarAvx2(float* a, float scalar, std::size_t size) {
    SubScalarAvx(a, scalar, size);
  }

  static void SubScalarAvx(float* a, float scalar, std::size_t size) {
    __m256       ymm0      = _mm256_set1_ps(scalar);
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256 ymm1 = _mm256_loadu_ps(a + i);
      ymm1        = _mm256_sub_ps(ymm1, ymm0);
      _mm256_storeu_ps(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void SubScalarSse42(float* a, float scalar, std::size_t size) {
    SubScalarSse3(a, scalar, size);
  }

  static void SubScalarSse41(float* a, float scalar, std::size_t size) {
    SubScalarSse3(a, scalar, size);
  }

  static void SubScalarSsse3(float* a, float scalar, std::size_t size) {
    SubScalarSse3(a, scalar, size);
  }

  static void SubScalarSse3(float* a, float scalar, std::size_t size) {
    __m128       xmm0      = _mm_set1_ps(scalar);
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128 xmm1 = _mm_loadu_ps(a + i);
      xmm1        = _mm_sub_ps(xmm1, xmm0);
      _mm_storeu_ps(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void SubScalarFallback(float* a, float scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  // END: subtract scalar
  //----------------------------------------------------------------------------

  // BEGIN: negation array
  //----------------------------------------------------------------------------

  static void NegAvx2(float* a, std::size_t size) { NegAvx(a, size); }

  static void NegAvx(float* a, std::size_t size) {
    __m256       negZero   = _mm256_set1_ps(-0.0f);
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256 ymm1 = _mm256_loadu_ps(a + i);
      ymm1        = _mm256_xor_ps(ymm1, negZero);
      _mm256_storeu_ps(a + i, ymm1);
    }

    // Handle any remaining elements without SIMD
    for (; i < size; ++i) {
      a[i] = -a[i];
    }
  }

  static void NegSse42(float* a, std::size_t size) { NegSse3(a, size); }

  static void NegSse41(float* a, std::size_t size) { NegSse3(a, size); }

  static void NegSsse3(float* a, std::size_t size) { NegSse3(a, size); }

  static void NegSse3(float* a, std::size_t size) {
    __m128       negZero   = _mm_set1_ps(-0.0f);
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128 xmm1 = _mm_loadu_ps(a + i);
      xmm1        = _mm_xor_ps(xmm1, negZero);
      _mm_storeu_ps(a + i, xmm1);
    }

    // Handle any remaining elements without SIMD
    for (; i < size; ++i) {
      a[i] = -a[i];
    }
  }

  static void NegFallback(float* a, std::size_t size) {
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
  static inline auto LoadAAvx(const float* a,
                              const std::size_t kCurrentRowA,
                              const std::size_t kInnerIndex,
                              const std::size_t kRowsA,
                              const std::size_t kColsARowsB) -> __m256 {
    if constexpr (Option == Options::RowMajor) {
      return _mm256_loadu_ps(
          &a[IndexA<Option>(kCurrentRowA, kInnerIndex, kRowsA, kColsARowsB)]);
    } else {
      return _mm256_set_ps(
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 7, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 6, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 5, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 4, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 3, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 2, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 1, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex, kRowsA, kColsARowsB)]);
    }
  }

  template <Options Option>
  static inline auto LoadBAvx(const float* b,
                              const std::size_t kInnerIndex,
                              const std::size_t kCurrentColB,
                              const std::size_t kColsB,
                              const std::size_t kColsARowsB) -> __m256 {
    if constexpr (Option == Options::RowMajor) {
      return _mm256_set_ps(
          b[IndexB<Option>(kInnerIndex + 7, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 6, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 5, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 4, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 3, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 2, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 1, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]);
    } else {
      return _mm256_loadu_ps(
          &b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]);
    }
  }

  // END: AVX multiplication array utility functions

  // BEGIN: SSE multiplication array utility functions

  template <Options Option>
  static inline auto LoadASse(const float* a,
                              const std::size_t kCurrentRowA,
                              const std::size_t kInnerIndex,
                              const std::size_t kRowsA,
                              const std::size_t kColsARowsB) -> __m128 {
    if constexpr (Option == Options::RowMajor) {
      return _mm_loadu_ps(
          &a[IndexA<Option>(kCurrentRowA, kInnerIndex, kRowsA, kColsARowsB)]);
    } else {
      return _mm_set_ps(
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 3, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 2, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 1, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex, kRowsA, kColsARowsB)]);
    }
  }

  template <Options Option>
  static inline auto LoadBSse(const float* b,
                              const std::size_t kInnerIndex,
                              const std::size_t kCurrentColB,
                              const std::size_t kColsB,
                              const std::size_t kColsARowsB) -> __m128 {
    if constexpr (Option == Options::ColumnMajor) {
      return _mm_loadu_ps(
          &b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]);
    } else {
      return _mm_set_ps(
          b[IndexB<Option>(kInnerIndex + 3, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 2, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 1, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]);
    }
  }

  // END: SSE multiplication array utility functions

  // END: multiplication array utility functions

  template <Options Option>
  static void MulAvx2(float*       result,
                      const float* a,
                      const float* b,
                      const std::size_t kRowsA,
                      const std::size_t kColsB,
                      const std::size_t kColsARowsB) {
    MulAvx<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulAvx(float*       result,
                     const float* a,
                     const float* b,
                     const std::size_t kRowsA,
                     const std::size_t kColsB,
                     const std::size_t kColsARowsB) {
    for (std::size_t currentRowA = 0; currentRowA < kRowsA; ++currentRowA) {
      for (std::size_t currentColB = 0; currentColB < kColsB; ++currentColB) {
        __m256 sum        = _mm256_setzero_ps();
        std::size_t innerIndex = 0;
        for (; innerIndex + s_kAvxSimdWidth - 1 < kColsARowsB;
             innerIndex += s_kAvxSimdWidth) {
          __m256 aVec = LoadAAvx<Option>(
              a, currentRowA, innerIndex, kRowsA, kColsARowsB);
          __m256 bVec = LoadBAvx<Option>(
              b, innerIndex, currentColB, kColsB, kColsARowsB);

          sum = _mm256_fmadd_ps(aVec, bVec, sum);
        }
        float tmp[s_kAvxSimdWidth];
        _mm256_storeu_ps(tmp, sum);
        float finalSum = 0.0f;
        for (float i : tmp) {
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
  static void MulSse42(float*       result,
                       const float* a,
                       const float* b,
                       const std::size_t kRowsA,
                       const std::size_t kColsB,
                       const std::size_t kColsARowsB) {
    MulSse3<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSse41(float*       result,
                       const float* a,
                       const float* b,
                       const std::size_t kRowsA,
                       const std::size_t kColsB,
                       const std::size_t kColsARowsB) {
    MulSse3<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSsse3(float*       result,
                       const float* a,
                       const float* b,
                       const std::size_t kRowsA,
                       const std::size_t kColsB,
                       const std::size_t kColsARowsB) {
    MulSse3<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSse3(float*       result,
                      const float* a,
                      const float* b,
                      const std::size_t kRowsA,
                      const std::size_t kColsB,
                      const std::size_t kColsARowsB) {
    for (std::size_t currentRowA = 0; currentRowA < kRowsA; ++currentRowA) {
      for (std::size_t currentColB = 0; currentColB < kColsB; ++currentColB) {
        __m128 sum        = _mm_setzero_ps();
        std::size_t innerIndex = 0;
        for (; innerIndex + s_kSseSimdWidth - 1 < kColsARowsB;
             innerIndex += s_kSseSimdWidth) {
          __m128 aVec = LoadASse<Option>(
              a, currentRowA, innerIndex, kRowsA, kColsARowsB);
          __m128 bVec = LoadBSse<Option>(
              b, innerIndex, currentColB, kColsB, kColsARowsB);
          sum = _mm_add_ps(sum, _mm_mul_ps(aVec, bVec));
        }
        float tmp[s_kSseSimdWidth];
        _mm_storeu_ps(tmp, sum);
        float finalSum = 0.0f;
        for (float i : tmp) {
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
  static void MulFallback(float*       result,
                          const float* a,
                          const float* b,
                          const std::size_t kRowsA,
                          const std::size_t kColsB,
                          const std::size_t kColsARowsB) {
    for (std::size_t i = 0; i < kRowsA; ++i) {
      for (std::size_t j = 0; j < kColsB; ++j) {
        float sum = 0;
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

  static void MulScalarAvx2(float* a, float scalar, std::size_t size) {
    MulScalarAvx(a, scalar, size);
  }

  static void MulScalarAvx(float* a, float scalar, std::size_t size) {
    __m256       ymm0      = _mm256_set1_ps(scalar);
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256 ymm1 = _mm256_loadu_ps(a + i);
      ymm1        = _mm256_mul_ps(ymm1, ymm0);
      _mm256_storeu_ps(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void MulScalarSse42(float* a, float scalar, std::size_t size) {
    MulScalarSse3(a, scalar, size);
  }

  static void MulScalarSse41(float* a, float scalar, std::size_t size) {
    MulScalarSse3(a, scalar, size);
  }

  static void MulScalarSsse3(float* a, float scalar, std::size_t size) {
    MulScalarSse3(a, scalar, size);
  }

  static void MulScalarSse3(float* a, float scalar, std::size_t size) {
    __m128       xmm0      = _mm_set1_ps(scalar);
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128 xmm1 = _mm_loadu_ps(a + i);
      xmm1        = _mm_mul_ps(xmm1, xmm0);
      _mm_storeu_ps(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void MulScalarFallback(float* a, float scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  // END: multiplication scalar
  //----------------------------------------------------------------------------

  // BEGIN: division scalar
  //----------------------------------------------------------------------------

  static void DivScalarAvx2(float* a, float scalar, std::size_t size) {
    DivScalarAvx(a, scalar, size);
  }

  static void DivScalarAvx(float* a, float scalar, std::size_t size) {
    __m256       ymm0      = _mm256_set1_ps(scalar);
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256 ymm1 = _mm256_loadu_ps(a + i);
      ymm1        = _mm256_div_ps(ymm1, ymm0);
      _mm256_storeu_ps(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  static void DivScalarSse42(float* a, float scalar, std::size_t size) {
    DivScalarSse3(a, scalar, size);
  }

  static void DivScalarSse41(float* a, float scalar, std::size_t size) {
    DivScalarSse3(a, scalar, size);
  }

  static void DivScalarSsse3(float* a, float scalar, std::size_t size) {
    DivScalarSse3(a, scalar, size);
  }

  static void DivScalarSse3(float* a, float scalar, std::size_t size) {
    __m128       xmm0      = _mm_set1_ps(scalar);
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128 xmm1 = _mm_loadu_ps(a + i);
      xmm1        = _mm_div_ps(xmm1, xmm0);
      _mm_storeu_ps(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  static void DivScalarFallback(float* a, float scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  // END: division scalar
  //----------------------------------------------------------------------------

  // BEGIN: comparison array
  //----------------------------------------------------------------------------

  static int CmpAvx2(const float* a, const float* b, std::size_t size) {
    return CmpAvx(a, b, size);
  }

  static int CmpAvx(const float* a, const float* b, std::size_t size) {
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256 aVec      = _mm256_loadu_ps(a + i);
      __m256 bVec      = _mm256_loadu_ps(b + i);
      __m256 cmpResult = _mm256_cmp_ps(aVec, bVec, _CMP_LT_OQ);
      int    mask      = _mm256_movemask_ps(cmpResult);
      if (mask != 0) {
        return -1;
      }
      cmpResult = _mm256_cmp_ps(aVec, bVec, _CMP_GT_OQ);
      mask      = _mm256_movemask_ps(cmpResult);
      if (mask != 0) {
        return 1;
      }
      cmpResult = _mm256_cmp_ps(aVec, bVec, _CMP_NEQ_OQ);
      mask      = _mm256_movemask_ps(cmpResult);
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

  static int CmpSse42(const float* a, const float* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSse41(const float* a, const float* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSsse3(const float* a, const float* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSse3(const float* a, const float* b, std::size_t size) {
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128 aVec      = _mm_loadu_ps(a + i);
      __m128 bVec      = _mm_loadu_ps(b + i);
      __m128 cmpResult = _mm_cmplt_ps(aVec, bVec);
      int    mask      = _mm_movemask_ps(cmpResult);
      if (mask != 0) {
        return -1;
      }
      cmpResult = _mm_cmpgt_ps(aVec, bVec);
      mask      = _mm_movemask_ps(cmpResult);
      if (mask != 0) {
        return 1;
      }
      cmpResult = _mm_cmpeq_ps(aVec, bVec);
      mask      = _mm_movemask_ps(cmpResult);
      if (mask == 0xF) {
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

  static int CmpFallback(const float* a, const float* b, std::size_t size) {
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