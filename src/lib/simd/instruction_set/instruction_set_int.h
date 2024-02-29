/**
 * @file instruction_set_int.h
 */

#ifndef MATH_LIBRARY_INSTRUCTION_SET_INT_H
#define MATH_LIBRARY_INSTRUCTION_SET_INT_H

#include "../../options/options.h"
#include "../precompiled/simd_defines.h"

#include <immintrin.h>

namespace math {
template <typename T>
class InstructionSet;

template <>
class InstructionSet<int> {
  public:
  using AddFunc = void (*)(int*, const int*, size_t);

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

  using AddScalarFunc = void (*)(int*, int, size_t);

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

  using SubFunc = void (*)(int*, const int*, size_t);

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

  using SubScalarFunc = void (*)(int*, int, size_t);

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

  using NegFunc = void (*)(int*, size_t);

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
  using MulFunc = void (*)(
      int*, const int*, const int*, const size_t, const size_t, const size_t);

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

  using MulScalarFunc = void (*)(int*, int, size_t);

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

  using DivScalarFunc = void (*)(int*, int, size_t);

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
  static constexpr size_t s_kAvxSimdWidth = sizeof(__m256i) / sizeof(int);  // 8
  static constexpr size_t s_kSseSimdWidth = sizeof(__m128i) / sizeof(int);  // 4

  // BEGIN: add two arrays
  //----------------------------------------------------------------------------

  static void AddAvx2(int* a, const int* b, size_t size) { AddAvx(a, b, size); }

  static void AddAvx(int* a, const int* b, size_t size) {
    const size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
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

  static void AddSse42(int* a, const int* b, size_t size) {
    AddSse3(a, b, size);
  }

  static void AddSse41(int* a, const int* b, size_t size) {
    AddSse3(a, b, size);
  }

  static void AddSsse3(int* a, const int* b, size_t size) {
    AddSse3(a, b, size);
  }

  static void AddSse3(int* a, const int* b, size_t size) {
    const size_t kSseLimit = size - (size % s_kSseSimdWidth);
    size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
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

  static void AddFallback(int* a, const int* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] += b[i];
    }
  }

  // END: add two arrays
  //----------------------------------------------------------------------------

  // BEGIN: add scalar
  //----------------------------------------------------------------------------

  static void AddScalarAvx2(int* a, int scalar, size_t size) {
    AddScalarAvx(a, scalar, size);
  }

  static void AddScalarAvx(int* a, int scalar, size_t size) {
    __m256i      ymm0      = _mm256_set1_epi32(scalar);
    const size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256i ymm1 = _mm256_loadu_si256((const __m256i*)(a + i));
      ymm1         = _mm256_add_epi32(ymm1, ymm0);
      _mm256_storeu_si256((__m256i*)(a + i), ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void AddScalarSse42(int* a, int scalar, size_t size) {
    AddScalarSse3(a, scalar, size);
  }

  static void AddScalarSse41(int* a, int scalar, size_t size) {
    AddScalarSse3(a, scalar, size);
  }

  static void AddScalarSsse3(int* a, int scalar, size_t size) {
    AddScalarSse3(a, scalar, size);
  }

  static void AddScalarSse3(int* a, int scalar, size_t size) {
    __m128i      xmm0      = _mm_set1_epi32(scalar);
    const size_t kSseLimit = size - (size % s_kSseSimdWidth);
    size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128i xmm1 = _mm_loadu_si128((const __m128i*)(a + i));
      xmm1         = _mm_add_epi32(xmm1, xmm0);
      _mm_storeu_si128((__m128i*)(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void AddScalarFallback(int* a, int scalar, size_t size) {
    // no SIMD
    for (size_t i = 0; i < size; ++i) {
      a[i] += scalar;
    }
  }

  // END: add scalar
  //----------------------------------------------------------------------------

  // BEGIN: subtract two arrays
  //----------------------------------------------------------------------------

  static void SubAvx2(int* a, const int* b, size_t size) { SubAvx(a, b, size); }

  static void SubAvx(int* a, const int* b, size_t size) {
    const size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
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

  static void SubSse42(int* a, const int* b, size_t size) {
    SubSse3(a, b, size);
  }

  static void SubSse41(int* a, const int* b, size_t size) {
    SubSse3(a, b, size);
  }

  static void SubSsse3(int* a, const int* b, size_t size) {
    SubSse3(a, b, size);
  }

  static void SubSse3(int* a, const int* b, size_t size) {
    const size_t kSseLimit = size - (size % s_kSseSimdWidth);
    size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
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

  static void SubFallback(int* a, const int* b, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  // END: subtract two arrays
  //----------------------------------------------------------------------------

  static void SubScalarAvx2(int* a, int scalar, size_t size) {
    SubScalarAvx(a, scalar, size);
  }

  static void SubScalarAvx(int* a, int scalar, size_t size) {
    __m256i      ymm0      = _mm256_set1_epi32(scalar);
    const size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256i ymm1 = _mm256_loadu_si256((const __m256i*)(a + i));
      ymm1         = _mm256_sub_epi32(ymm1, ymm0);
      _mm256_storeu_si256((__m256i*)(a + i), ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void SubScalarSse42(int* a, int scalar, size_t size) {
    SubScalarSse3(a, scalar, size);
  }

  static void SubScalarSse41(int* a, int scalar, size_t size) {
    SubScalarSse3(a, scalar, size);
  }

  static void SubScalarSsse3(int* a, int scalar, size_t size) {
    SubScalarSse3(a, scalar, size);
  }

  static void SubScalarSse3(int* a, int scalar, size_t size) {
    __m128i      xmm0      = _mm_set1_epi32(scalar);
    const size_t kSseLimit = size - (size % s_kSseSimdWidth);
    size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128i xmm1 = _mm_loadu_si128((const __m128i*)(a + i));
      xmm1         = _mm_sub_epi32(xmm1, xmm0);
      _mm_storeu_si128((__m128i*)(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void SubScalarFallback(int* a, int scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  // END: subtract scalar
  //----------------------------------------------------------------------------

  // BEGIN: negation array
  //----------------------------------------------------------------------------

  static void NegAvx2(int* a, size_t size) { NegAvx(a, size); }

  static void NegAvx(int* a, size_t size) {
    const size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256i ymm1
          = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
      ymm1 = _mm256_sub_epi32(_mm256_setzero_si256(), ymm1);  // Negate
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i), ymm1);
    }

    // Handle any remaining elements without SIMD
    for (; i < size; ++i) {
      a[i] = -a[i];
    }
  }

  static void NegSse42(int* a, size_t size) { NegSse3(a, size); }

  static void NegSse41(int* a, size_t size) { NegSse3(a, size); }

  static void NegSsse3(int* a, size_t size) { NegSse3(a, size); }

  static void NegSse3(int* a, size_t size) {
    const size_t kSseLimit = size - (size % s_kSseSimdWidth);
    size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128i xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
      xmm1         = _mm_sub_epi32(_mm_setzero_si128(), xmm1);  // Negate
      _mm_storeu_si128(reinterpret_cast<__m128i*>(a + i), xmm1);
    }

    // Handle any remaining elements without SIMD
    for (; i < size; ++i) {
      a[i] = -a[i];
    }
  }

  static void NegFallback(int* a, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] = -a[i];
    }
  }

  // END: negation array
  //----------------------------------------------------------------------------

  // BEGIN: multiplication array
  //----------------------------------------------------------------------------

  // BEGIN: multiplication array utility functions

  template <Options Option>
  static inline auto IndexA(const size_t kCurrentRowA,
                            const size_t kInnerIndex,
                            const size_t kRowsA,
                            const size_t kColsARowsB) -> size_t {
    if constexpr (Option == Options::ColumnMajor) {
      return kCurrentRowA + kInnerIndex * kRowsA;
    } else if constexpr (Option == Options::RowMajor) {
      return kCurrentRowA * kColsARowsB + kInnerIndex;
    }
  }

  template <Options Option>
  static inline auto IndexB(const size_t kInnerIndex,
                            const size_t kCurrentColB,
                            const size_t kColsB,
                            const size_t kColsARowsB) -> size_t {
    if constexpr (Option == Options::ColumnMajor) {
      return kInnerIndex + kCurrentColB * kColsARowsB;
    } else if constexpr (Option == Options::RowMajor) {
      return kInnerIndex * kColsB + kCurrentColB;
    }
  }

  template <Options Option>
  static inline auto IndexResult(const size_t kCurrentRowA,
                                 const size_t kCurrentColB,
                                 const size_t kRowsA,
                                 const size_t kColsB) -> size_t {
    if constexpr (Option == Options::ColumnMajor) {
      return kCurrentRowA + kCurrentColB * kRowsA;
    } else if constexpr (Option == Options::RowMajor) {
      return kCurrentRowA * kColsB + kCurrentColB;
    }
  }

  // BEGIN: AVX multiplication array utility functions

  template <Options Option>
  static inline auto LoadAAvx(const int*   a,
                              const size_t kCurrentRowA,
                              const size_t kInnerIndex,
                              const size_t kRowsA,
                              const size_t kColsARowsB) -> __m256i {
    if constexpr (Option == Options::RowMajor) {
      return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          &a[IndexA<Option>(kCurrentRowA, kInnerIndex, kRowsA, kColsARowsB)]));
    } else {
      return _mm256_set_epi32(
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
  static inline auto LoadB(const int*   b,
                           const size_t kInnerIndex,
                           const size_t kCurrentColB,
                           const size_t kColsB,
                           const size_t kColsARowsB) -> __m256 {
    if constexpr (Option == Options::RowMajor) {
      return _mm256_set_epi32(
          b[IndexB<Option>(kInnerIndex + 7, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 6, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 5, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 4, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 3, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 2, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 1, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]);
    } else {
      return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          &b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]));
    }
  }

  // END: AVX multiplication array utility functions

  // BEGIN: SSE multiplication array utility functions

  template <Options Option>
  static inline auto LoadASse(const int*   a,
                              const size_t kCurrentRowA,
                              const size_t kInnerIndex,
                              const size_t kRowsA,
                              const size_t kColsARowsB) -> __m128i {
    if constexpr (Option == Options::RowMajor) {
      return _mm_loadu_si128(reinterpret_cast<const __m128i*>(
          &a[IndexA<Option>(kCurrentRowA, kInnerIndex, kRowsA, kColsARowsB)]));
    } else {
      return _mm_set_epi32(
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 3, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 2, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex + 1, kRowsA, kColsARowsB)],
          a[IndexA<Option>(kCurrentRowA, kInnerIndex, kRowsA, kColsARowsB)]);
    }
  }

  template <Options Option>
  static inline auto LoadBSse(const int*   b,
                              const size_t kInnerIndex,
                              const size_t kCurrentColB,
                              const size_t kColsB,
                              const size_t kColsARowsB) -> __m128 {
    if constexpr (Option == Options::ColumnMajor) {
      return _mm_loadu_si128(reinterpret_cast<const __m128i*>(
          &b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]));
    } else {
      return _mm_set_epi32(
          b[IndexB<Option>(kInnerIndex + 3, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 2, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex + 1, kCurrentColB, kColsB, kColsARowsB)],
          b[IndexB<Option>(kInnerIndex, kCurrentColB, kColsB, kColsARowsB)]);
    }
  }

  // END: SSE multiplication array utility functions

  template <Options Option>
  static void MulAvx2(int*         result,
                      const int*   a,
                      const int*   b,
                      const size_t kRowsA,
                      const size_t kColsB,
                      const size_t kColsARowsB) {
    for (size_t currentRowA = 0; currentRowA < kRowsA; ++currentRowA) {
      for (size_t currentColB = 0; currentColB < kColsB; ++currentColB) {
        __m256i sum        = _mm256_setzero_si256();
        size_t  innerIndex = 0;
        for (; innerIndex + s_kAvxSimdWidth - 1 < kColsARowsB;
             innerIndex += s_kAvxSimdWidth) {
          __m256i aVec = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(&a[IndexA<Option>(
                  currentRowA, innerIndex, kRowsA, kColsARowsB)]));
          __m256i bVec = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(&b[IndexB<Option>(
                  innerIndex, currentColB, kColsB, kColsARowsB)]));

          sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(aVec, bVec));
        }
        int tmp[s_kAvxSimdWidth];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), sum);
        int finalSum = 0;
        for (int i : tmp) {
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
  static void MulAvx(int*         result,
                     const int*   a,
                     const int*   b,
                     const size_t kRowsA,
                     const size_t kColsB,
                     const size_t kColsARowsB) {
    // downgrade to SSE 4.2 since AVX does not support direct multiplication of
    // 32-bit integers
    MulSse42<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSse42(int*         result,
                       const int*   a,
                       const int*   b,
                       const size_t kRowsA,
                       const size_t kColsB,
                       const size_t kColsARowsB) {
    MulSse41<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSse41(int*         result,
                       const int*   a,
                       const int*   b,
                       const size_t kRowsA,
                       const size_t kColsB,
                       const size_t kColsARowsB) {
    for (size_t currentRowA = 0; currentRowA < kRowsA; ++currentRowA) {
      for (size_t currentColB = 0; currentColB < kColsB; ++currentColB) {
        __m128i sum        = _mm_setzero_si128();
        size_t  innerIndex = 0;
        for (; innerIndex + s_kSseSimdWidth - 1 < kColsARowsB;
             innerIndex += s_kSseSimdWidth) {
          __m128i a_vec = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(&a[IndexA<Option>(
                  currentRowA, innerIndex, kRowsA, kColsARowsB)]));
          __m128i b_vec = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(&b[IndexB<Option>(
                  innerIndex, currentColB, kColsB, kColsARowsB)]));

          sum = _mm_add_epi32(sum, _mm_mullo_epi32(a_vec, b_vec));
        }
        int tmp[s_kSseSimdWidth];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), sum);
        int finalSum = 0;
        for (int i : tmp) {
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
  static void MulSsse3(int*         result,
                       const int*   a,
                       const int*   b,
                       const size_t kRowsA,
                       const size_t kColsB,
                       const size_t kColsARowsB) {
    // SSSE3 does not include instructions for multiplying a vector of 32-bit
    // integers by a scalar.
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSse3(int*         result,
                      const int*   a,
                      const int*   b,
                      const size_t kRowsA,
                      const size_t kColsB,
                      const size_t kColsARowsB) {
    // SSE3 does not include instructions for multiplying a vector of 32-bit
    // integers by a scalar.
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulFallback(int*         result,
                          const int*   a,
                          const int*   b,
                          const size_t kRowsA,
                          const size_t kColsB,
                          const size_t kColsARowsB) {
    for (size_t i = 0; i < kRowsA; ++i) {
      for (size_t j = 0; j < kColsB; ++j) {
        int sum = 0;
        for (size_t k = 0; k < kColsARowsB; ++k) {
          sum += a[indexA<Option>(i, k, kRowsA, kColsARowsB)]
               * b[indexB<Option>(k, j, kColsB, kColsARowsB)];
        }
        result[indexResult<Option>(i, j, kRowsA, kColsB)] = sum;
      }
    }
  }

  // END: multiplication array
  //----------------------------------------------------------------------------

  // BEGIN: multiplication scalar
  //----------------------------------------------------------------------------

  static void MulScalarAvx2(int* a, int scalar, size_t size) {
    __m256i      ymm0       = _mm256_set1_epi32(scalar);
    const size_t kAvx2Limit = size - (size % s_kAvxSimdWidth);
    size_t       i          = 0;

    for (; i < kAvx2Limit; i += s_kAvxSimdWidth) {
      __m256i ymm1 = _mm256_loadu_si256((__m256i*)(a + i));
      ymm1         = _mm256_mullo_epi32(ymm1, ymm0);
      _mm256_storeu_si256((__m256i*)(a + i), ymm1);
    }

    // Handle any remaining values that didn't fit into the last AVX2 vector
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void MulScalarAvx(int* a, int scalar, size_t size) {
    MulScalarSse42(a, scalar, size);  // downgrade to SSE 4.2 since AVX does
                                      // not support direct multiplication of
                                      // 32-bit integers
  }

  static void MulScalarSse42(int* a, int scalar, size_t size) {
    MulScalarSse41(a, scalar, size);
  }

  static void MulScalarSse41(int* a, int scalar, size_t size) {
    __m128i      xmm0      = _mm_set1_epi32(scalar);
    const size_t kSseLimit = size - (size % s_kSseSimdWidth);
    size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128i xmm1 = _mm_loadu_si128((const __m128i*)(a + i));
      xmm1         = _mm_mullo_epi32(xmm1, xmm0);
      _mm_storeu_si128((__m128i*)(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void MulScalarSsse3(int* a, int scalar, size_t size) {
    // SSSE3 does not include instructions for multiplying a vector of 32-bit
    // integers by a scalar.
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarSse3(int* a, int scalar, size_t size) {
    // SSE3 does not include instructions for multiplying a vector of 32-bit
    // integers by a scalar.
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarFallback(int* a, int scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  // END: multiplication scalar
  //----------------------------------------------------------------------------

  // BEGIN: division scalar
  //----------------------------------------------------------------------------

  static void DivScalarAvx2(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    DivScalarAvx(a, scalar, size);
#else
    DivScalarFallback(a, scalar, size);
#endif
  }

  static void DivScalarAvx(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    const size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    __m256i      ymm0      = _mm256_set1_epi32(scalar);
    size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256i ymm1 = _mm256_loadu_epi32(a + i);
      ymm1         = _mm256_div_epi32(ymm1, ymm0);
      _mm256_storeu_epi32(a + i, ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] /= scalar;
    }
#else
    DivScalarFallback(a, scalar, size);
#endif
  }

  static void DivScalarSse42(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    DivScalarSse3(a, scalar, size);
#else
    DivScalarFallback(a, scalar, size);
#endif
  }

  static void DivScalarSse41(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    DivScalarSse3(a, scalar, size);
#else
    DivScalarFallback(a, scalar, size);
#endif
  }

  static void DivScalarSsse3(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    DivScalarSse3(a, scalar, size);
#else
    DivScalarFallback(a, scalar, size);
#endif
  }

  static void DivScalarSse3(int* a, int scalar, size_t size) {
#ifdef SUPPORTS_SVML
    const size_t kSseLimit = size - (size % s_kSseSimdWidth);
    __m128i      xmm0      = _mm_set1_epi32(scalar);
    size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128i xmm1 = _mm_loadu_epi32(a + i);
      xmm1         = _mm_div_epi32(xmm1, xmm0);
      _mm_storeu_epi32(a + i, xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] /= scalar;
    }
#else
    DivScalarFallback(a, scalar, size);
#endif
  }

  static void DivScalarFallback(int* a, int scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  // END: division scalar
  //----------------------------------------------------------------------------
};

}  // namespace math

#endif
