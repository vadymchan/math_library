/**
 * @file instruction_set_uint32.h
 */

#ifndef MATH_LIBRARY_INSTRUCTION_SET_UINT32_H
#define MATH_LIBRARY_INSTRUCTION_SET_UINT32_H

#include "../../options/options.h"
#include "../precompiled/simd_defines.h"

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace math {

template <typename T>
class InstructionSet;

template <>
class InstructionSet<std::uint32_t> {
  public:
  using AddFunc = void (*)(std::uint32_t*, const std::uint32_t*, std::size_t);

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

  using AddScalarFunc = void (*)(std::uint32_t*, std::uint32_t, std::size_t);

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

  using SubFunc = void (*)(std::uint32_t*, const std::uint32_t*, std::size_t);

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

  using SubScalarFunc = void (*)(std::uint32_t*, std::uint32_t, std::size_t);

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

  using NegFunc = void (*)(std::uint32_t*, std::size_t);

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
  using MulFunc = void (*)(std::uint32_t*,
                           const std::uint32_t*,
                           const std::uint32_t*,
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

  using MulScalarFunc = void (*)(std::uint32_t*, std::uint32_t, std::size_t);

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

  using DivScalarFunc = void (*)(std::uint32_t*, std::uint32_t, std::size_t);

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

  using CmpFunc
      = int (*)(const std::uint32_t*, const std::uint32_t*, std::size_t);

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
      = sizeof(__m256i) / sizeof(std::uint32_t);  // 8
  static constexpr std::size_t s_kSseSimdWidth
      = sizeof(__m128i) / sizeof(std::uint32_t);  // 4

  // BEGIN: add two arrays
  //----------------------------------------------------------------------------

  static void AddAvx2(std::uint32_t*       a,
                      const std::uint32_t* b,
                      std::size_t          size) {
    AddAvx(a, b, size);
  }

  static void AddAvx(std::uint32_t*       a,
                     const std::uint32_t* b,
                     std::size_t          size) {
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256i ymm1
          = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
      __m256i ymm2
          = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
      // Note: _mm256_add_epi32 works for both signed and unsigned
      ymm1 = _mm256_add_epi32(ymm1, ymm2);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i), ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += b[i];
    }
  }

  static void AddSse42(std::uint32_t*       a,
                       const std::uint32_t* b,
                       std::size_t          size) {
    AddSse3(a, b, size);
  }

  static void AddSse41(std::uint32_t*       a,
                       const std::uint32_t* b,
                       std::size_t          size) {
    AddSse3(a, b, size);
  }

  static void AddSsse3(std::uint32_t*       a,
                       const std::uint32_t* b,
                       std::size_t          size) {
    AddSse3(a, b, size);
  }

  static void AddSse3(std::uint32_t*       a,
                      const std::uint32_t* b,
                      std::size_t          size) {
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128i xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
      __m128i xmm2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
      // Note: _mm_add_epi32 works for both signed and unsigned
      xmm1 = _mm_add_epi32(xmm1, xmm2);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += b[i];
    }
  }

  static void AddFallback(std::uint32_t*       a,
                          const std::uint32_t* b,
                          std::size_t          size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] += b[i];
    }
  }

  // END: add two arrays
  //----------------------------------------------------------------------------

  // BEGIN: add scalar
  //----------------------------------------------------------------------------

  static void AddScalarAvx2(std::uint32_t* a,
                            std::uint32_t  scalar,
                            std::size_t    size) {
    AddScalarAvx(a, scalar, size);
  }

  static void AddScalarAvx(std::uint32_t* a,
                           std::uint32_t  scalar,
                           std::size_t    size) {
    __m256i           ymm0      = _mm256_set1_epi32(scalar);
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256i ymm1
          = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
      ymm1 = _mm256_add_epi32(ymm1, ymm0);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i), ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void AddScalarSse42(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
    AddScalarSse3(a, scalar, size);
  }

  static void AddScalarSse41(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
    AddScalarSse3(a, scalar, size);
  }

  static void AddScalarSsse3(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
    AddScalarSse3(a, scalar, size);
  }

  static void AddScalarSse3(std::uint32_t* a,
                            std::uint32_t  scalar,
                            std::size_t    size) {
    __m128i xmm0 = _mm_set1_epi32(
        scalar);  // Same intrinsic can be used for std::uint32_t
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128i xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
      xmm1         = _mm_add_epi32(xmm1, xmm0);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] += scalar;
    }
  }

  static void AddScalarFallback(std::uint32_t* a,
                                std::uint32_t  scalar,
                                std::size_t    size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] += scalar;
    }
  }

  // END: add scalar
  //----------------------------------------------------------------------------

  // BEGIN: subtract two arrays
  //----------------------------------------------------------------------------
  static void SubAvx2(std::uint32_t*       a,
                      const std::uint32_t* b,
                      std::size_t          size) {
    SubAvx(a, b, size);
  }

  static void SubAvx(std::uint32_t*       a,
                     const std::uint32_t* b,
                     std::size_t          size) {
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256i ymm1
          = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
      __m256i ymm2
          = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
      ymm1 = _mm256_sub_epi32(ymm1, ymm2);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i), ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  static void SubSse42(std::uint32_t*       a,
                       const std::uint32_t* b,
                       std::size_t          size) {
    SubSse3(a, b, size);
  }

  static void SubSse41(std::uint32_t*       a,
                       const std::uint32_t* b,
                       std::size_t          size) {
    SubSse3(a, b, size);
  }

  static void SubSsse3(std::uint32_t*       a,
                       const std::uint32_t* b,
                       std::size_t          size) {
    SubSse3(a, b, size);
  }

  static void SubSse3(std::uint32_t*       a,
                      const std::uint32_t* b,
                      std::size_t          size) {
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128i xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
      __m128i xmm2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
      xmm1         = _mm_sub_epi32(xmm1, xmm2);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  static void SubFallback(std::uint32_t*       a,
                          const std::uint32_t* b,
                          std::size_t          size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  // END: subtract two arrays
  //----------------------------------------------------------------------------

  // BEGIN: subtract scalar
  //----------------------------------------------------------------------------

  static void SubScalarAvx2(std::uint32_t* a,
                            std::uint32_t  scalar,
                            std::size_t    size) {
    SubScalarAvx(a, scalar, size);
  }

  static void SubScalarAvx(std::uint32_t* a,
                           std::uint32_t  scalar,
                           std::size_t    size) {
    __m256i           ymm0      = _mm256_set1_epi32(scalar);
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256i ymm1
          = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
      ymm1 = _mm256_sub_epi32(ymm1, ymm0);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i), ymm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void SubScalarSse42(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
    SubScalarSse3(a, scalar, size);
  }

  static void SubScalarSse41(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
    SubScalarSse3(a, scalar, size);
  }

  static void SubScalarSsse3(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
    SubScalarSse3(a, scalar, size);
  }

  static void SubScalarSse3(std::uint32_t* a,
                            std::uint32_t  scalar,
                            std::size_t    size) {
    __m128i           xmm0      = _mm_set1_epi32(scalar);
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128i xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
      xmm1         = _mm_sub_epi32(xmm1, xmm0);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  static void SubScalarFallback(std::uint32_t* a,
                                std::uint32_t  scalar,
                                std::size_t    size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  // END: subtract scalar
  //----------------------------------------------------------------------------

  // BEGIN: negation array fallback
  //----------------------------------------------------------------------------

  static void NegAvx2(std::uint32_t* a, std::size_t size) {
    NegFallback(a, size);
  }

  static void NegAvx(std::uint32_t* a, std::size_t size) {
    NegFallback(a, size);
  }

  static void NegSse42(std::uint32_t* a, std::size_t size) {
    NegFallback(a, size);
  }

  static void NegSse41(std::uint32_t* a, std::size_t size) {
    NegFallback(a, size);
  }

  static void NegSsse3(std::uint32_t* a, std::size_t size) {
    NegFallback(a, size);
  }

  static void NegSse3(std::uint32_t* a, std::size_t size) {
    NegFallback(a, size);
  }

  static void NegFallback(std::uint32_t* a, std::size_t size) {
    assert(false && "Negation for std::uint32_t is not supported.");
  }

  // END: negation array fallback
  //----------------------------------------------------------------------------

  // BEGIN: multiplication array
  //----------------------------------------------------------------------------

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

#ifdef FEATURE_LOAD_SIMD_INT

  // BEGIN: AVX multiplication array utility functions

  template <Options Option>
  static inline auto LoadAAvx(const std::uint32_t* a,
                              const std::size_t    kCurrentRowA,
                              const std::size_t    kInnerIndex,
                              const std::size_t    kRowsA,
                              const std::size_t    kColsARowsB) -> __m256i {
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
  static inline auto LoadBAvx(const std::uint32_t* b,
                              const std::size_t    kInnerIndex,
                              const std::size_t    kCurrentColB,
                              const std::size_t    kColsB,
                              const std::size_t    kColsARowsB) -> __m256i {
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
  static inline auto LoadASse(const std::uint32_t* a,
                              const std::size_t    kCurrentRowA,
                              const std::size_t    kInnerIndex,
                              const std::size_t    kRowsA,
                              const std::size_t    kColsARowsB) -> __m128i {
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
  static inline auto LoadBSse(const std::uint32_t* b,
                              const std::size_t    kInnerIndex,
                              const std::size_t    kCurrentColB,
                              const std::size_t    kColsB,
                              const std::size_t    kColsARowsB) -> __m128i {
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

#endif

  template <Options Option>
  static void MulAvx2(std::uint32_t*       result,
                      const std::uint32_t* a,
                      const std::uint32_t* b,
                      const std::size_t    kRowsA,
                      const std::size_t    kColsB,
                      const std::size_t    kColsARowsB) {
    for (std::size_t currentRowA = 0; currentRowA < kRowsA; ++currentRowA) {
      for (std::size_t currentColB = 0; currentColB < kColsB; ++currentColB) {
        __m256i     sum        = _mm256_setzero_si256();
        std::size_t innerIndex = 0;
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
        std::uint32_t tmp[s_kAvxSimdWidth];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), sum);
        std::uint32_t finalSum = 0;
        for (std::uint32_t i : tmp) {
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
  static void MulAvx(std::uint32_t*       result,
                     const std::uint32_t* a,
                     const std::uint32_t* b,
                     std::size_t          kRowsA,
                     std::size_t          kColsB,
                     std::size_t          kColsARowsB) {
    for (std::size_t currentRowA = 0; currentRowA < kRowsA; ++currentRowA) {
      for (std::size_t currentColB = 0; currentColB < kColsB; ++currentColB) {
        __m256i     sum        = _mm256_setzero_si256();
        std::size_t innerIndex = 0;

        for (; innerIndex + s_kAvxSimdWidth - 1 < kColsARowsB;
             innerIndex += s_kAvxSimdWidth) {
          __m256i aVec = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(&a[IndexA<Option>(
                  currentRowA, innerIndex, kRowsA, kColsARowsB)]));
          __m256i bVec = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(&b[IndexB<Option>(
                  innerIndex, currentColB, kColsB, kColsARowsB)]));

          __m128i aLo = _mm256_castsi256_si128(aVec);
          __m128i aHi = _mm256_extracti128_si256(aVec, 1);
          __m128i bLo = _mm256_castsi256_si128(bVec);
          __m128i bHi = _mm256_extracti128_si256(bVec, 1);

          __m128i mulLo = _mm_mullo_epi32(aLo, bLo);
          __m128i mulHi = _mm_mullo_epi32(aHi, bHi);

          __m256i mulResult = _mm256_castsi128_si256(mulLo);
          mulResult         = _mm256_insertf128_si256(mulResult, mulHi, 1);

          sum = _mm256_add_epi32(sum, mulResult);
        }

        std::uint32_t tmp[s_kAvxSimdWidth];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), sum);
        std::uint32_t finalSum = 0;
        for (std::uint32_t i : tmp) {
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
  static void MulSse42(std::uint32_t*       result,
                       const std::uint32_t* a,
                       const std::uint32_t* b,
                       std::size_t          kRowsA,
                       std::size_t          kColsB,
                       std::size_t          kColsARowsB) {
    MulSse41<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  // MulSse41 - SSE4.1 implementation
  template <Options Option>
  static void MulSse41(std::uint32_t*       result,
                       const std::uint32_t* a,
                       const std::uint32_t* b,
                       std::size_t          kRowsA,
                       std::size_t          kColsB,
                       std::size_t          kColsARowsB) {
    for (std::size_t currentRowA = 0; currentRowA < kRowsA; ++currentRowA) {
      for (std::size_t currentColB = 0; currentColB < kColsB; ++currentColB) {
        __m128i     sum        = _mm_setzero_si128();
        std::size_t innerIndex = 0;
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
        std::uint32_t tmp[s_kSseSimdWidth];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), sum);
        std::uint32_t finalSum = 0;
        for (std::uint32_t i : tmp) {
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
  static void MulSsse3(std::uint32_t*       result,
                       const std::uint32_t* a,
                       const std::uint32_t* b,
                       const std::size_t    kRowsA,
                       const std::size_t    kColsB,
                       const std::size_t    kColsARowsB) {
    // SSSE3 does not include instructions for multiplying a vector of 32-bit
    // integers by a scalar.
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSse3(std::uint32_t*       result,
                      const std::uint32_t* a,
                      const std::uint32_t* b,
                      const std::size_t    kRowsA,
                      const std::size_t    kColsB,
                      const std::size_t    kColsARowsB) {
    // SSE3 does not include instructions for multiplying a vector of 32-bit
    // integers by a scalar.
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulFallback(std::uint32_t*       result,
                          const std::uint32_t* a,
                          const std::uint32_t* b,
                          const std::size_t    kRowsA,
                          const std::size_t    kColsB,
                          const std::size_t    kColsARowsB) {
    for (std::size_t i = 0; i < kRowsA; ++i) {
      for (std::size_t j = 0; j < kColsB; ++j) {
        std::uint32_t sum = 0;
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

  static void MulScalarAvx2(std::uint32_t* a,
                            std::uint32_t  scalar,
                            std::size_t    size) {
    __m256i           ymm0       = _mm256_set1_epi32(scalar);
    const std::size_t kAvx2Limit = size - (size % s_kAvxSimdWidth);
    std::size_t       i          = 0;

    for (; i < kAvx2Limit; i += s_kAvxSimdWidth) {
      __m256i ymm1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i));
      ymm1         = _mm256_mullo_epi32(ymm1, ymm0);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i), ymm1);
    }

    // Handle any remaining values that didn't fit into the last AVX2 vector
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void MulScalarAvx(std::uint32_t* a,
                           std::uint32_t  scalar,
                           std::size_t    size) {
    MulScalarSse42(a, scalar, size);  // downgrade to SSE 4.2 since AVX does
                                      // not support direct multiplication of
                                      // 32-bit integers
  }

  static void MulScalarSse42(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
    MulScalarSse41(a, scalar, size);
  }

  static void MulScalarSse41(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
    __m128i           xmm0      = _mm_set1_epi32(scalar);
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128i xmm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
      xmm1         = _mm_mullo_epi32(xmm1, xmm0);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(a + i), xmm1);
    }

    // Handle any remainder
    for (; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  static void MulScalarSsse3(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
    // SSSE3 does not include instructions for multiplying a vector of 32-bit
    // integers by a scalar.
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarSse3(std::uint32_t* a,
                            std::uint32_t  scalar,
                            std::size_t    size) {
    // SSE3 does not include instructions for multiplying a vector of 32-bit
    // integers by a scalar.
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarFallback(std::uint32_t* a,
                                std::uint32_t  scalar,
                                std::size_t    size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  // END: multiplication scalar
  //----------------------------------------------------------------------------

  // BEGIN: division scalar
  //----------------------------------------------------------------------------

  static void DivScalarAvx2(std::uint32_t* a,
                            std::uint32_t  scalar,
                            std::size_t    size) {
#ifdef SUPPORTS_SVML
    DivScalarAvx(a, scalar, size);
#else
    DivScalarFallback(a, scalar, size);
#endif
  }

  static void DivScalarAvx(std::uint32_t* a,
                           std::uint32_t  scalar,
                           std::size_t    size) {
#ifdef SUPPORTS_SVML
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    __m256i           ymm0      = _mm256_set1_epi32(scalar);
    std::size_t       i         = 0;

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

  static void DivScalarSse42(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
#ifdef SUPPORTS_SVML
    DivScalarSse3(a, scalar, size);
#else
    DivScalarFallback(a, scalar, size);
#endif
  }

  static void DivScalarSse41(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
#ifdef SUPPORTS_SVML
    DivScalarSse3(a, scalar, size);
#else
    DivScalarFallback(a, scalar, size);
#endif
  }

  static void DivScalarSsse3(std::uint32_t* a,
                             std::uint32_t  scalar,
                             std::size_t    size) {
#ifdef SUPPORTS_SVML
    DivScalarSse3(a, scalar, size);
#else
    DivScalarFallback(a, scalar, size);
#endif
  }

  static void DivScalarSse3(std::uint32_t* a,
                            std::uint32_t  scalar,
                            std::size_t    size) {
#ifdef SUPPORTS_SVML
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    __m128i           xmm0      = _mm_set1_epi32(scalar);
    std::size_t       i         = 0;

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

  static void DivScalarFallback(std::uint32_t* a,
                                std::uint32_t  scalar,
                                std::size_t    size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  // END: division scalar
  //----------------------------------------------------------------------------

  // BEGIN: comparison array
  //----------------------------------------------------------------------------

  static int CmpAvx2(const std::uint32_t* a,
                     const std::uint32_t* b,
                     std::size_t          size) {
    return CmpAvx(a, b, size);
  }

  static int CmpAvx(const std::uint32_t* a,
                    const std::uint32_t* b,
                    std::size_t          size) {
    const std::size_t kAvxLimit = size - (size % s_kAvxSimdWidth);
    std::size_t       i         = 0;

    for (; i < kAvxLimit; i += s_kAvxSimdWidth) {
      __m256i aVec = _mm256_loadu_si256((__m256i*)(a + i));
      __m256i bVec = _mm256_loadu_si256((__m256i*)(b + i));

      __m256i signBit  = _mm256_set1_epi32(0x80'00'00'00);
      __m256i aVecSign = _mm256_xor_si256(aVec, signBit);
      __m256i bVecSign = _mm256_xor_si256(bVec, signBit);

      __m256i cmpResult = _mm256_cmpgt_epi32(bVecSign, aVecSign);
      int     mask      = _mm256_movemask_ps(_mm256_castsi256_ps(cmpResult));
      if (mask != 0) {
        return -1;
      }

      cmpResult = _mm256_cmpgt_epi32(aVecSign, bVecSign);
      mask      = _mm256_movemask_ps(_mm256_castsi256_ps(cmpResult));
      if (mask != 0) {
        return 1;
      }

      cmpResult = _mm256_cmpeq_epi32(aVec, bVec);
      mask      = _mm256_movemask_ps(_mm256_castsi256_ps(cmpResult));
      if (mask == 0xFF) {
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

  static int CmpSse42(const std::uint32_t* a,
                      const std::uint32_t* b,
                      std::size_t          size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSse41(const std::uint32_t* a,
                      const std::uint32_t* b,
                      std::size_t          size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSsse3(const std::uint32_t* a,
                      const std::uint32_t* b,
                      std::size_t          size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSse3(const std::uint32_t* a,
                     const std::uint32_t* b,
                     std::size_t          size) {
    const std::size_t kSseLimit = size - (size % s_kSseSimdWidth);
    std::size_t       i         = 0;

    for (; i < kSseLimit; i += s_kSseSimdWidth) {
      __m128i aVec = _mm_loadu_si128((__m128i*)(a + i));
      __m128i bVec = _mm_loadu_si128((__m128i*)(b + i));

      __m128i signBit  = _mm_set1_epi32(0x80'00'00'00);
      __m128i aVecSign = _mm_xor_si128(aVec, signBit);
      __m128i bVecSign = _mm_xor_si128(bVec, signBit);

      __m128i cmpResult = _mm_cmpgt_epi32(bVecSign, aVecSign);
      int     mask      = _mm_movemask_ps(_mm_castsi128_ps(cmpResult));
      if (mask != 0) {
        return -1;
      }

      cmpResult = _mm_cmpgt_epi32(aVecSign, bVecSign);
      mask      = _mm_movemask_ps(_mm_castsi128_ps(cmpResult));
      if (mask != 0) {
        return 1;
      }

      cmpResult = _mm_cmpeq_epi32(aVec, bVec);
      mask      = _mm_movemask_ps(_mm_castsi128_ps(cmpResult));
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

  static int CmpFallback(const std::uint32_t* a,
                         const std::uint32_t* b,
                         std::size_t          size) {
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

#endif  // MATH_LIBRARY_INSTRUCTION_SET_UINT32_H
