/**
 * @file instruction_set.h
 */

#ifndef MATH_LIBRARY_INSTRUCTION_SET_H
#define MATH_LIBRARY_INSTRUCTION_SET_H

#include "../../options/options.h"
#include "../precompiled/simd_defines.h"

#include <immintrin.h>

#include <type_traits>

namespace math {

template <typename T>
concept SimdSupportedType
    = std::is_same_v<T, std::int32_t> || std::is_same_v<T, float>
   || std::is_same_v<T, double> || std::is_same_v<T, std::uint32_t>;

template <typename T>
class InstructionSet {
  public:
#ifdef MATH_LIBRARY_ENABLE_SIMD_TYPE_CHECK
  static_assert(SimdSupportedType<T>,
                "InstructionSet supports only int, float, and double types");
#endif

  using AddFunc = void (*)(T*, const T*, std::size_t);

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

  using AddScalarFunc = void (*)(T*, T, std::size_t);

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

  using SubFunc = void (*)(T*, const T*, std::size_t);

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

  using SubScalarFunc = void (*)(T*, T, std::size_t);

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

  using NegFunc = void (*)(T*, std::size_t);

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
  using MulFunc = void (*)(T*,
                           const T*,
                           const T*,
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

  using MulScalarFunc = void (*)(T*, T, std::size_t);

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

  using DivScalarFunc = void (*)(T*, T, std::size_t);

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

  using CmpFunc = int (*)(const T*, const T*, std::size_t);

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
  // BEGIN: add two arrays
  //----------------------------------------------------------------------------

  static void AddAvx2(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  static void AddAvx(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  static void AddSse42(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  static void AddSse41(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  static void AddSsse3(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  static void AddSse3(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  static void AddFallback(T* a, const T* b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] += b[i];
    }
  }

  // END: add two arrays
  //----------------------------------------------------------------------------

  // BEGIN: add scalar
  //----------------------------------------------------------------------------

  static void AddScalarAvx2(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarAvx(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarSse42(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarSse41(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarSsse3(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarSse3(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  static void AddScalarFallback(T* a, T scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] += scalar;
    }
  }

  // END: add scalar
  //----------------------------------------------------------------------------

  // BEGIN: subtraction array
  //----------------------------------------------------------------------------

  static void SubAvx2(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  static void SubAvx(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  static void SubSse42(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  static void SubSse41(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  static void SubSsse3(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  static void SubSse3(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  static void SubFallback(T* a, const T* b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  // END: subtraction array
  //----------------------------------------------------------------------------

  // BEGIN: subtraction scalar
  //----------------------------------------------------------------------------

  static void SubScalarAvx2(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarAvx(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarSse42(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarSse41(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarSsse3(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarSse3(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  static void SubScalarFallback(T* a, T scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  // END: subtraction scalar
  //----------------------------------------------------------------------------

  // BEGIN: negation array
  //----------------------------------------------------------------------------

  static void NegAvx2(T* a, std::size_t size) { NegFallback(a, size); }

  static void NegAvx(T* a, std::size_t size) { NegFallback(a, size); }

  static void NegSse42(T* a, std::size_t size) { NegFallback(a, size); }

  static void NegSse41(T* a, std::size_t size) { NegFallback(a, size); }

  static void NegSsse3(T* a, std::size_t size) { NegFallback(a, size); }

  static void NegSse3(T* a, std::size_t size) { NegFallback(a, size); }

  static void NegFallback(T* a, std::size_t size) {
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

  // END: multiplication array utility functions

  template <Options Option>
  static void MulAvx2(T*                result,
                      const T*          a,
                      const T*          b,
                      const std::size_t kRowsA,
                      const std::size_t kColsB,
                      const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulAvx(T*                result,
                     const T*          a,
                     const T*          b,
                     const std::size_t kRowsA,
                     const std::size_t kColsB,
                     const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSse42(T*                result,
                       const T*          a,
                       const T*          b,
                       const std::size_t kRowsA,
                       const std::size_t kColsB,
                       const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSse41(T*                result,
                       const T*          a,
                       const T*          b,
                       const std::size_t kRowsA,
                       const std::size_t kColsB,
                       const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSsse3(T*                result,
                       const T*          a,
                       const T*          b,
                       const std::size_t kRowsA,
                       const std::size_t kColsB,
                       const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulSse3(T*                result,
                      const T*          a,
                      const T*          b,
                      const std::size_t kRowsA,
                      const std::size_t kColsB,
                      const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  template <Options Option>
  static void MulFallback(T*                result,
                          const T*          a,
                          const T*          b,
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

  static void MulScalarAvx2(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  /**
   * @brief Multiplies each element of an array by a scalar using AVX SIMD
  static void MulScalarAvx(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }
  static void MulScalarSse42(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarSse41(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarSsse3(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarSse3(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  static void MulScalarFallback(T* a, T scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  // END: multiplication scalar
  //----------------------------------------------------------------------------

  // BEGIN: division scalar
  //----------------------------------------------------------------------------

  static void DivScalarAvx2(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarAvx(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarSse42(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarSse41(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarSsse3(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarSse3(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  static void DivScalarFallback(T* a, T scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  // END: division scalar
  //----------------------------------------------------------------------------

  // BEGIN: comparison array
  //----------------------------------------------------------------------------

  static int CmpAvx2(const T* a, const T* b, std::size_t size) {
    return CmpAvx(a, b, size);
  }

  static int CmpAvx(const T* a, const T* b, std::size_t size) {
    return CmpFallback(a, b, size);
  }

  static int CmpSse42(const T* a, const T* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSse41(const T* a, const T* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSsse3(const T* a, const T* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  static int CmpSse3(const T* a, const T* b, std::size_t size) {
    return CmpFallback(a, b, size);
  }

  static int CmpFallback(const T* a, const T* b, std::size_t size) {
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

#include "instruction_set_double.h"
#include "instruction_set_float.h"
#include "instruction_set_int32.h"
#include "instruction_set_uint32.h"

#endif