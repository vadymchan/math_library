/**
 * @file instruction_set.h
 * @brief Provides SIMD-based instruction set functionalities for generalized
 * data types.
 *
 * This file contains templates and functions that utilize SIMD instructions to
 * perform operations such as addition, subtraction, multiplication, division,
 * and comparison. The implementation adapts to the supported SIMD instruction
 * set available at runtime, such as AVX2, AVX, SSE, or fallback implementations
 * for platforms without SIMD support.
 */

#ifndef MATH_LIBRARY_INSTRUCTION_SET_H
#define MATH_LIBRARY_INSTRUCTION_SET_H

#include "../../options/options.h"
#include "../precompiled/simd_defines.h"

#include <immintrin.h>

#include <type_traits>

namespace math {

/**
 * @brief Concept to check if the type is supported by SIMD instructions.
 *
 * This concept ensures that the type is either a `std::int32_t`,
 * `std::uint32_t`, `float`, or `double`, which are the types supported by the
 * SIMD instruction set in this library.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept SimdSupportedType
    = std::is_same_v<T, std::int32_t> || std::is_same_v<T, float>
   || std::is_same_v<T, double> || std::is_same_v<T, std::uint32_t>;

/**
 * @brief Class that provides SIMD instruction set functionalities for supported
 * types.
 *
 * This class contains methods for performing SIMD-optimized operations such as
 * addition, subtraction, multiplication, negation, and comparison. Depending on
 * the available SIMD instruction set (AVX, SSE, etc.), it selects the optimal
 * function for each operation.
 *
 * @tparam T The data type (must satisfy `SimdSupportedType` concept).
 */
template <typename T>
class InstructionSet {
  public:
#ifdef MATH_LIBRARY_ENABLE_SIMD_TYPE_CHECK
  static_assert(SimdSupportedType<T>,
                "InstructionSet supports only int, float, and double types");
#endif

  /**
   * @brief Function pointer type for array addition operations.
   */
  using AddFunc = void (*)(T*, const T*, std::size_t);

  /**
   * @brief Returns the appropriate function for adding two arrays based on the
   * supported SIMD set.
   *
   * This function selects the most efficient SIMD implementation available for
   * adding two arrays of type `T`.
   *
   * @return A function pointer to the SIMD implementation for array addition.
   */
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

  /**
   * @brief Function pointer type for scalar addition operations.
   */
  using AddScalarFunc = void (*)(T*, T, std::size_t);

  /**
   * @brief Returns the appropriate function for adding a scalar to an array
   * based on the supported SIMD set.
   *
   * This function dynamically selects the most efficient SIMD implementation
   * available for adding a scalar value to an array of type `T`.
   *
   * @return A function pointer to the SIMD implementation for scalar addition.
   */
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

  /**
   * @brief Function pointer type for array subtraction operations.
   */
  using SubFunc = void (*)(T*, const T*, std::size_t);

  /**
   * @brief Returns the appropriate function for subtracting one array from
   * another based on the supported SIMD set.
   *
   * This function dynamically selects the most efficient SIMD implementation
   * available for subtracting one array from another.
   *
   * @return A function pointer to the SIMD implementation for array
   * subtraction.
   */
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

  /**
   * @brief Function pointer type for scalar subtraction operations.
   */
  using SubScalarFunc = void (*)(T*, T, std::size_t);

  /**
   * @brief Returns the appropriate function for subtracting a scalar from an
   * array based on the supported SIMD set.
   *
   * This function dynamically selects the most efficient SIMD implementation
   * available for subtracting a scalar from an array.
   *
   * @return A function pointer to the SIMD implementation for scalar
   * subtraction.
   */
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

  /**
   * @brief Function pointer type for negating an array.
   */
  using NegFunc = void (*)(T*, std::size_t);

  /**
   * @brief Returns the appropriate function for negating an array based on the
   * supported SIMD set.
   *
   * This function dynamically selects the most efficient SIMD implementation
   * available for negating an array of type `T`.
   *
   * @return A function pointer to the SIMD implementation for negation.
   */
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

  /**
   * @brief Function pointer type for matrix multiplication.
   *
   * This type defines a function pointer for performing matrix multiplication
   * on arrays. The function multiplies two matrices and stores the result in an
   * output array. It supports different memory layout options.
   *
   * @tparam Option The memory layout option, either row-major or column-major.
   */
  template <Options Option>
  using MulFunc = void (*)(T*,
                           const T*,
                           const T*,
                           const std::size_t,
                           const std::size_t,
                           const std::size_t);

  /**
   * @brief Returns the appropriate function for matrix multiplication based on
   * the supported SIMD set.
   *
   * This function dynamically selects the most efficient SIMD implementation
   * available for multiplying two matrices, depending on the available
   * instruction set (e.g., AVX2, AVX, SSE).
   *
   * @tparam Option The memory layout option (row-major or column-major).
   * @return A function pointer to the SIMD implementation for matrix
   * multiplication.
   */
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

  /**
   * @brief Function pointer type for scalar multiplication.
   *
   * This type defines a function pointer for multiplying each element in an
   * array by a scalar value.
   */
  using MulScalarFunc = void (*)(T*, T, std::size_t);

  /**
   * @brief Returns the appropriate function for scalar multiplication based on
   * the supported SIMD set.
   *
   * This function dynamically selects the most efficient SIMD implementation
   * available for multiplying each element in an array by a scalar.
   *
   * @return A function pointer to the SIMD implementation for scalar
   * multiplication.
   */
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

  /**
   * @brief Function pointer type for scalar division.
   *
   * This type defines a function pointer for dividing each element in an array
   * by a scalar value.
   */
  using DivScalarFunc = void (*)(T*, T, std::size_t);

  /**
   * @brief Returns the appropriate function for scalar division based on the
   * supported SIMD set.
   *
   * This function dynamically selects the most efficient SIMD implementation
   * available for dividing each element in an array by a scalar.
   *
   * @return A function pointer to the SIMD implementation for scalar division.
   */
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

  /**
   * @brief Function pointer type for array comparison.
   *
   * This type defines a function pointer for comparing two arrays. It returns
   * an integer indicating whether the first array is less than, equal to, or
   * greater than the second array.
   */
  using CmpFunc = int (*)(const T*, const T*, std::size_t);

  /**
   * @brief Returns the appropriate function for comparing two arrays based on
   * the supported SIMD set.
   *
   * This function dynamically selects the most efficient SIMD implementation
   * available for comparing two arrays.
   *
   * @return A function pointer to the SIMD implementation for array comparison.
   */
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

  /**
   * @brief Adds two arrays using AVX2 SIMD instructions.
   *
   * This function adds two arrays element-wise using AVX2 SIMD instructions.
   * If AVX2 is not supported, it falls back to the generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be added.
   * @param size The number of elements in the arrays.
   */
  static void AddAvx2(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  /**
   * @brief Adds two arrays using AVX SIMD instructions.
   *
   * This function adds two arrays element-wise using AVX SIMD instructions.
   * If AVX is not supported, it falls back to the generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be added.
   * @param size The number of elements in the arrays.
   */
  static void AddAvx(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  /**
   * @brief Adds two arrays using SSE4.2 SIMD instructions.
   *
   * This function adds two arrays element-wise using SSE4.2 SIMD instructions.
   * If SSE4.2 is not supported, it falls back to the generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be added.
   * @param size The number of elements in the arrays.
   */
  static void AddSse42(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  /**
   * @brief Adds two arrays using SSE4.1 SIMD instructions.
   *
   * This function adds two arrays element-wise using SSE4.1 SIMD instructions.
   * If SSE4.1 is not supported, it falls back to the generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be added.
   * @param size The number of elements in the arrays.
   */
  static void AddSse41(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  /**
   * @brief Adds two arrays using SSSE3 SIMD instructions.
   *
   * This function adds two arrays element-wise using SSSE3 SIMD instructions.
   * If SSSE3 is not supported, it falls back to the generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be added.
   * @param size The number of elements in the arrays.
   */
  static void AddSsse3(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  /**
   * @brief Adds two arrays using SSE3 SIMD instructions.
   *
   * This function adds two arrays element-wise using SSE3 SIMD instructions.
   * If SSE3 is not supported, it falls back to the generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be added.
   * @param size The number of elements in the arrays.
   */
  static void AddSse3(T* a, const T* b, std::size_t size) {
    AddFallback(a, b, size);
  }

  /**
   * @brief Adds two arrays element-wise in a generic way.
   *
   * This is the fallback function for adding two arrays element-wise without
   * SIMD support.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be added.
   * @param size The number of elements in the arrays.
   */
  static void AddFallback(T* a, const T* b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] += b[i];
    }
  }

  // END: add two arrays
  //----------------------------------------------------------------------------

  // BEGIN: add scalar
  //----------------------------------------------------------------------------

  /**
   * @brief Adds a scalar to each element of an array using AVX2 SIMD
   * instructions.
   *
   * This function adds a scalar value to each element in the array using AVX2
   * SIMD instructions. If AVX2 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The destination array (modified in-place).
   * @param scalar The scalar value to be added to each element.
   * @param size The number of elements in the array.
   */
  static void AddScalarAvx2(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  /**
   * @brief Adds a scalar to each element of an array using AVX SIMD
   * instructions.
   *
   * This function adds a scalar value to each element in the array using AVX
   * SIMD instructions. If AVX is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be modified (each element will be incremented by the
   * scalar value).
   * @param scalar The scalar value to be added to each element of the array.
   * @param size The number of elements in the array.
   */
  static void AddScalarAvx(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  /**
   * @brief Adds a scalar to each element of an array using SSE4.2 SIMD
   * instructions.
   *
   * This function adds a scalar value to each element in the array using SSE4.2
   * SIMD instructions. If SSE4.2 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be modified (each element will be incremented by the
   * scalar value).
   * @param scalar The scalar value to be added to each element of the array.
   * @param size The number of elements in the array.
   */
  static void AddScalarSse42(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  /**
   * @brief Adds a scalar to each element of an array using SSE4.1 SIMD
   * instructions.
   *
   * This function adds a scalar value to each element in the array using SSE4.1
   * SIMD instructions. If SSE4.1 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be modified (each element will be incremented by the
   * scalar value).
   * @param scalar The scalar value to be added to each element of the array.
   * @param size The number of elements in the array.
   */
  static void AddScalarSse41(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  /**
   * @brief Adds a scalar to each element of an array using SSSE3 SIMD
   * instructions.
   *
   * This function adds a scalar value to each element in the array using SSSE3
   * SIMD instructions. If SSSE3 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be modified (each element will be incremented by the
   * scalar value).
   * @param scalar The scalar value to be added to each element of the array.
   * @param size The number of elements in the array.
   */
  static void AddScalarSsse3(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  /**
   * @brief Adds a scalar to each element of an array using SSE3 SIMD
   * instructions.
   *
   * This function adds a scalar value to each element in the array using SSE3
   * SIMD instructions. If SSE3 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be modified (each element will be incremented by the
   * scalar value).
   * @param scalar The scalar value to be added to each element of the array.
   * @param size The number of elements in the array.
   */
  static void AddScalarSse3(T* a, T scalar, std::size_t size) {
    AddScalarFallback(a, scalar, size);
  }

  /**
   * @brief Adds a scalar to each element of an array in a generic way.
   *
   * This is the fallback function for adding a scalar to each element in the
   * array without SIMD support.
   *
   * @param a The destination array (modified in-place).
   * @param scalar The scalar value to be added to each element.
   * @param size The number of elements in the array.
   */
  static void AddScalarFallback(T* a, T scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] += scalar;
    }
  }

  // END: add scalar
  //----------------------------------------------------------------------------

  // BEGIN: subtraction array
  //----------------------------------------------------------------------------

  /**
   * @brief Subtracts two arrays using AVX2 SIMD instructions.
   *
   * This function subtracts the second array from the first array element-wise
   * using AVX2 SIMD instructions. If AVX2 is not supported, it falls back to
   * the generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be subtracted.
   * @param size The number of elements in the arrays.
   */
  static void SubAvx2(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  /**
   * @brief Subtracts two arrays using AVX SIMD instructions.
   *
   * This function subtracts the second array from the first array element-wise
   * using AVX SIMD instructions. If AVX is not supported, it falls back to the
   * generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be subtracted.
   * @param size The number of elements in the arrays.
   */
  static void SubAvx(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  /**
   * @brief Subtracts two arrays using SSE4.2 SIMD instructions.
   *
   * This function subtracts the second array from the first array element-wise
   * using SSE4.2 SIMD instructions. If SSE4.2 is not supported, it falls back
   * to the generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be subtracted.
   * @param size The number of elements in the arrays.
   */
  static void SubSse42(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  /**
   * @brief Subtracts two arrays using SSE4.1 SIMD instructions.
   *
   * This function subtracts the second array from the first array element-wise
   * using SSE4.1 SIMD instructions. If SSE4.1 is not supported, it falls back
   * to the generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be subtracted.
   * @param size The number of elements in the arrays.
   */
  static void SubSse41(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  /**
   * @brief Subtracts two arrays using SSSE3 SIMD instructions.
   *
   * This function subtracts the second array from the first array element-wise
   * using SSSE3 SIMD instructions. If SSSE3 is not supported, it falls back to
   * the generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be subtracted.
   * @param size The number of elements in the arrays.
   */
  static void SubSsse3(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  /**
   * @brief Subtracts two arrays using SSE3 SIMD instructions.
   *
   * This function subtracts the second array from the first array element-wise
   * using SSE3 SIMD instructions. If SSE3 is not supported, it falls back to
   * the generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be subtracted.
   * @param size The number of elements in the arrays.
   */
  static void SubSse3(T* a, const T* b, std::size_t size) {
    SubFallback(a, b, size);
  }

  /**
   * @brief Subtracts two arrays element-wise in a generic way.
   *
   * This is the fallback function for subtracting two arrays element-wise
   * without SIMD support.
   *
   * @param a The destination array (modified in-place).
   * @param b The source array to be subtracted.
   * @param size The number of elements in the arrays.
   */
  static void SubFallback(T* a, const T* b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] -= b[i];
    }
  }

  // END: subtraction array
  //----------------------------------------------------------------------------

  // BEGIN: subtraction scalar
  //----------------------------------------------------------------------------

  /**
   * @brief Subtracts a scalar from each element of an array using AVX2 SIMD
   * instructions.
   *
   * This function subtracts a scalar value from each element in the array using
   * AVX2 SIMD instructions. If AVX2 is not supported, it falls back to the
   * generic implementation.
   *
   * @param a The destination array (modified in-place).
   * @param scalar The scalar value to be subtracted from each element.
   * @param size The number of elements in the array.
   */
  static void SubScalarAvx2(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  /**
   * @brief Subtracts a scalar from each element of an array using AVX SIMD
   * instructions.
   *
   * This function subtracts a scalar value from each element in the array using
   * AVX SIMD instructions. If AVX is not supported, it falls back to the
   * generic implementation.
   *
   * @param a The array to be modified (in-place).
   * @param scalar The scalar value to subtract from each element.
   * @param size The number of elements in the array.
   */
  static void SubScalarAvx(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  /**
   * @brief Subtracts a scalar from each element of an array using SSE4.2 SIMD
   * instructions.
   *
   * This function subtracts a scalar value from each element in the array using
   * SSE4.2 SIMD instructions. If SSE4.2 is not supported, it falls back to the
   * generic implementation.
   *
   * @param a The array to be modified (in-place).
   * @param scalar The scalar value to subtract from each element.
   * @param size The number of elements in the array.
   */
  static void SubScalarSse42(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  /**
   * @brief Subtracts a scalar from each element of an array using SSE4.1 SIMD
   * instructions.
   *
   * This function subtracts a scalar value from each element in the array using
   * SSE4.1 SIMD instructions. If SSE4.1 is not supported, it falls back to the
   * generic implementation.
   *
   * @param a The array to be modified (in-place).
   * @param scalar The scalar value to subtract from each element.
   * @param size The number of elements in the array.
   */
  static void SubScalarSse41(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  /**
   * @brief Subtracts a scalar from each element of an array using SSSE3 SIMD
   * instructions.
   *
   * This function subtracts a scalar value from each element in the array using
   * SSSE3 SIMD instructions. If SSSE3 is not supported, it falls back to the
   * generic implementation.
   *
   * @param a The array to be modified (in-place).
   * @param scalar The scalar value to subtract from each element.
   * @param size The number of elements in the array.
   */
  static void SubScalarSsse3(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  /**
   * @brief Subtracts a scalar from each element of an array using SSE3 SIMD
   * instructions.
   *
   * This function subtracts a scalar value from each element in the array using
   * SSE3 SIMD instructions. If SSE3 is not supported, it falls back to the
   * generic implementation.
   *
   * @param a The array to be modified (in-place).
   * @param scalar The scalar value to subtract from each element.
   * @param size The number of elements in the array.
   */
  static void SubScalarSse3(T* a, T scalar, std::size_t size) {
    SubScalarFallback(a, scalar, size);
  }

  /**
   * @brief Subtracts a scalar from each element of an array in a generic way.
   *
   * This is the fallback function for subtracting a scalar from each element in
   * the array without SIMD support.
   *
   * @param a The destination array (modified in-place).
   * @param scalar The scalar value to be subtracted from each element.
   * @param size The number of elements in the array.
   */
  static void SubScalarFallback(T* a, T scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] -= scalar;
    }
  }

  // END: subtraction scalar
  //----------------------------------------------------------------------------

  // BEGIN: negation array
  //----------------------------------------------------------------------------

  /**
   * @brief Negates each element in an array using AVX2 SIMD instructions.
   *
   * This function negates each element in the array using AVX2 SIMD
   * instructions. If AVX2 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be negated (modified in-place).
   * @param size The number of elements in the array.
   */
  static void NegAvx2(T* a, std::size_t size) { NegFallback(a, size); }

  /**
   * @brief Negates each element in an array using AVX SIMD instructions.
   *
   * This function negates each element in the array using AVX SIMD
   * instructions. If AVX is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be negated (modified in-place).
   * @param size The number of elements in the array.
   */
  static void NegAvx(T* a, std::size_t size) { NegFallback(a, size); }

  /**
   * @brief Negates each element in an array using SSE4.2 SIMD instructions.
   *
   * This function negates each element in the array using SSE4.2 SIMD
   * instructions. If SSE4.2 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be negated (modified in-place).
   * @param size The number of elements in the array.
   */
  static void NegSse42(T* a, std::size_t size) { NegFallback(a, size); }

  /**
   * @brief Negates each element in an array using SSE4.1 SIMD instructions.
   *
   * This function negates each element in the array using SSE4.1 SIMD
   * instructions. If SSE4.1 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be negated (modified in-place).
   * @param size The number of elements in the array.
   */
  static void NegSse41(T* a, std::size_t size) { NegFallback(a, size); }

  /**
   * @brief Negates each element in an array using SSSE3 SIMD instructions.
   *
   * This function negates each element in the array using SSSE3 SIMD
   * instructions. If SSSE3 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be negated (modified in-place).
   * @param size The number of elements in the array.
   */
  static void NegSsse3(T* a, std::size_t size) { NegFallback(a, size); }

  /**
   * @brief Negates each element in an array using SSE3 SIMD instructions.
   *
   * This function negates each element in the array using SSE3 SIMD
   * instructions. If SSE3 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be negated (modified in-place).
   * @param size The number of elements in the array.
   */
  static void NegSse3(T* a, std::size_t size) { NegFallback(a, size); }

  /**
   * @brief Negates each element in an array in a generic way.
   *
   * This is the fallback function for negating each element in an array without
   * SIMD support.
   *
   * @param a The array to be negated (modified in-place).
   * @param size The number of elements in the array.
   */
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

  /**
   * @brief Computes the index of an element in matrix A for matrix
   * multiplication.
   *
   * This function calculates the index of an element in matrix A based on the
   * current row, the inner index for matrix multiplication, and the matrix
   * layout (row-major or column-major).
   *
   * @tparam Option Specifies the memory layout of the matrix (either
   * `Options::RowMajor` or `Options::ColumnMajor`).
   * @param kCurrentRowA The current row in matrix A being processed.
   * @param kInnerIndex The inner loop index for matrix multiplication.
   * @param kRowsA The total number of rows in matrix A.
   * @param kColsARowsB The shared inner dimension between matrix A and matrix B
   * (number of columns in A or rows in B).
   * @return The computed index of the element in matrix A.
   */
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

  /**
   * @brief Computes the index for accessing elements in matrix B based on
   * memory layout.
   *
   * This function calculates the index for accessing elements in matrix B
   * during matrix multiplication. It supports both row-major and column-major
   * memory layouts, depending on the specified `Option`.
   *
   * @tparam Option The memory layout option (either `Options::RowMajor` or
   * `Options::ColumnMajor`).
   * @param kInnerIndex The index in the inner loop (shared dimension between
   * matrices A and B).
   * @param kCurrentColB The current column in matrix B being processed.
   * @param kColsB The number of columns in matrix B.
   * @param kColsARowsB The shared dimension size between matrices A and B.
   * @return The computed index for accessing matrix B.
   */
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

  /**
   * @brief Computes the index for accessing elements in the result matrix based
   * on memory layout.
   *
   * This function calculates the index for accessing elements in the result
   * matrix during matrix multiplication. It supports both row-major and
   * column-major memory layouts, depending on the specified `Option`.
   *
   * @tparam Option The memory layout option (either `Options::RowMajor` or
   * `Options::ColumnMajor`).
   * @param kCurrentRowA The current row in matrix A being processed.
   * @param kCurrentColB The current column in matrix B being processed.
   * @param kRowsA The number of rows in matrix A.
   * @param kColsB The number of columns in matrix B.
   * @return The computed index for accessing the result matrix.
   */
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

  /**
   * @brief Multiplies two matrices using AVX2 SIMD instructions.
   *
   * This function multiplies two matrices and stores the result in an output
   * array using AVX2 SIMD instructions. If AVX2 is not supported, it falls back
   * to the generic implementation.
   *
   * @tparam Option The memory layout option (row-major or column-major).
   * @param result The output array where the result will be stored.
   * @param a The first matrix (left-hand side).
   * @param b The second matrix (right-hand side).
   * @param kRowsA The number of rows in the first matrix.
   * @param kColsB The number of columns in the second matrix.
   * @param kColsARowsB The inner dimension shared by both matrices.
   */
  template <Options Option>
  static void MulAvx2(T*                result,
                      const T*          a,
                      const T*          b,
                      const std::size_t kRowsA,
                      const std::size_t kColsB,
                      const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  /**
   * @brief Multiplies two matrices using AVX SIMD instructions.
   *
   * This function multiplies two matrices and stores the result in an output
   * array using AVX SIMD instructions. If AVX is not supported, it falls back
   * to the generic implementation.
   *
   * @tparam Option The memory layout option (row-major or column-major).
   * @param result The output array where the result will be stored.
   * @param a The first matrix (left-hand side).
   * @param b The second matrix (right-hand side).
   * @param kRowsA The number of rows in the first matrix.
   * @param kColsB The number of columns in the second matrix.
   * @param kColsARowsB The inner dimension shared by both matrices.
   */
  template <Options Option>
  static void MulAvx(T*                result,
                     const T*          a,
                     const T*          b,
                     const std::size_t kRowsA,
                     const std::size_t kColsB,
                     const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  /**
   * @brief Multiplies two matrices using SSE4.2 SIMD instructions.
   *
   * This function multiplies two matrices and stores the result in an output
   * array using SSE4.2 SIMD instructions. If SSE4.2 is not supported, it falls
   * back to the generic implementation.
   *
   * @tparam Option The memory layout option (row-major or column-major).
   * @param result The output array where the result will be stored.
   * @param a The first matrix (left-hand side).
   * @param b The second matrix (right-hand side).
   * @param kRowsA The number of rows in the first matrix.
   * @param kColsB The number of columns in the second matrix.
   * @param kColsARowsB The inner dimension shared by both matrices.
   */
  template <Options Option>
  static void MulSse42(T*                result,
                       const T*          a,
                       const T*          b,
                       const std::size_t kRowsA,
                       const std::size_t kColsB,
                       const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  /**
   * @brief Multiplies two matrices using SSE4.1 SIMD instructions.
   *
   * This function multiplies two matrices and stores the result in an output
   * array using SSE4.1 SIMD instructions. If SSE4.1 is not supported, it falls
   * back to the generic implementation.
   *
   * @tparam Option The memory layout option (row-major or column-major).
   * @param result The output array where the result will be stored.
   * @param a The first matrix (left-hand side).
   * @param b The second matrix (right-hand side).
   * @param kRowsA The number of rows in the first matrix.
   * @param kColsB The number of columns in the second matrix.
   * @param kColsARowsB The inner dimension shared by both matrices.
   */
  template <Options Option>
  static void MulSse41(T*                result,
                       const T*          a,
                       const T*          b,
                       const std::size_t kRowsA,
                       const std::size_t kColsB,
                       const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  /**
   * @brief Multiplies two matrices using SSSE3 SIMD instructions.
   *
   * This function multiplies two matrices and stores the result in an output
   * array using SSSE3 SIMD instructions. If SSSE3 is not supported, it falls
   * back to the generic implementation.
   *
   * @tparam Option The memory layout option (row-major or column-major).
   * @param result The output array where the result will be stored.
   * @param a The first matrix (left-hand side).
   * @param b The second matrix (right-hand side).
   * @param kRowsA The number of rows in the first matrix.
   * @param kColsB The number of columns in the second matrix.
   * @param kColsARowsB The inner dimension shared by both matrices.
   */
  template <Options Option>
  static void MulSsse3(T*                result,
                       const T*          a,
                       const T*          b,
                       const std::size_t kRowsA,
                       const std::size_t kColsB,
                       const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  /**
   * @brief Multiplies two matrices using SSE3 SIMD instructions.
   *
   * This function multiplies two matrices and stores the result in an output
   * array using SSE3 SIMD instructions. If SSE3 is not supported, it falls back
   * to the generic implementation.
   *
   * @tparam Option The memory layout option (row-major or column-major).
   * @param result The output array where the result will be stored.
   * @param a The first matrix (left-hand side).
   * @param b The second matrix (right-hand side).
   * @param kRowsA The number of rows in the first matrix.
   * @param kColsB The number of columns in the second matrix.
   * @param kColsARowsB The inner dimension shared by both matrices.
   */
  template <Options Option>
  static void MulSse3(T*                result,
                      const T*          a,
                      const T*          b,
                      const std::size_t kRowsA,
                      const std::size_t kColsB,
                      const std::size_t kColsARowsB) {
    MulFallback<Option>(result, a, b, kRowsA, kColsB, kColsARowsB);
  }

  /**
   * @brief Multiplies two matrices in a generic way.
   *
   * This is the fallback function for multiplying two matrices without SIMD
   * support.
   *
   * @tparam Option The memory layout option (row-major or column-major).
   * @param result The output array where the result will be stored.
   * @param a The first matrix (left-hand side).
   * @param b The second matrix (right-hand side).
   * @param kRowsA The number of rows in the first matrix.
   * @param kColsB The number of columns in the second matrix.
   * @param kColsARowsB The inner dimension shared by both matrices.
   */
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

  /**
   * @brief Multiplies each element of an array by a scalar using AVX2 SIMD
   * instructions.
   *
   * This function multiplies each element of the array by a scalar using AVX2
   * SIMD instructions. If AVX2 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be multiplied (modified in-place).
   * @param scalar The scalar value to multiply each element by.
   * @param size The number of elements in the array.
   */
  static void MulScalarAvx2(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  /**
   * @brief Multiplies each element of an array by a scalar using AVX SIMD
   * instructions.
   *
   * This function multiplies each element of the array by a scalar using AVX
   * SIMD instructions. If AVX is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be multiplied (modified in-place).
   * @param scalar The scalar value to multiply each element by.
   * @param size The number of elements in the array.
   */
  static void MulScalarAvx(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  /**
   * @brief Multiplies each element of an array by a scalar using SSE4.2 SIMD
   * instructions.
   *
   * This function multiplies each element of the array by a scalar using SSE4.2
   * SIMD instructions. If SSE4.2 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be multiplied (modified in-place).
   * @param scalar The scalar value to multiply each element by.
   * @param size The number of elements in the array.
   */
  static void MulScalarSse42(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  /**
   * @brief Multiplies each element of an array by a scalar using SSE4.1 SIMD
   * instructions.
   *
   * This function multiplies each element of the array by a scalar using SSE4.1
   * SIMD instructions. If SSE4.1 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be multiplied (modified in-place).
   * @param scalar The scalar value to multiply each element by.
   * @param size The number of elements in the array.
   */
  static void MulScalarSse41(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  /**
   * @brief Multiplies each element of an array by a scalar using SSSE3 SIMD
   * instructions.
   *
   * This function multiplies each element of the array by a scalar using SSSE3
   * SIMD instructions. If SSSE3 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be multiplied (modified in-place).
   * @param scalar The scalar value to multiply each element by.
   * @param size The number of elements in the array.
   */
  static void MulScalarSsse3(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  /**
   * @brief Multiplies each element of an array by a scalar using SSE3 SIMD
   * instructions.
   *
   * This function multiplies each element of the array by a scalar using SSE3
   * SIMD instructions. If SSE3 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be multiplied (modified in-place).
   * @param scalar The scalar value to multiply each element by.
   * @param size The number of elements in the array.
   */
  static void MulScalarSse3(T* a, T scalar, std::size_t size) {
    MulScalarFallback(a, scalar, size);
  }

  /**
   * @brief Multiplies each element of an array by a scalar in a generic way.
   *
   * This is the fallback function for multiplying each element of an array by a
   * scalar without SIMD support.
   *
   * @param a The array to be multiplied (modified in-place).
   * @param scalar The scalar value to multiply each element by.
   * @param size The number of elements in the array.
   */
  static void MulScalarFallback(T* a, T scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] *= scalar;
    }
  }

  // END: multiplication scalar
  //----------------------------------------------------------------------------

  // BEGIN: division scalar
  //----------------------------------------------------------------------------

  /**
   * @brief Divides each element of an array by a scalar using AVX2 SIMD
   * instructions.
   *
   * This function divides each element of the array by a scalar using AVX2 SIMD
   * instructions. If AVX2 is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be divided (modified in-place).
   * @param scalar The scalar value to divide each element by.
   * @param size The number of elements in the array.
   */
  static void DivScalarAvx2(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  /**
   * @brief Divides each element of an array by a scalar using AVX SIMD
   * instructions.
   *
   * This function divides each element in the array by a scalar value using AVX
   * SIMD instructions. If AVX is not supported, it falls back to the generic
   * implementation.
   *
   * @param a The array to be divided (modified in-place).
   * @param scalar The scalar value to divide each element by.
   * @param size The number of elements in the array.
   */
  static void DivScalarAvx(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  /**
   * @brief Divides each element of an array by a scalar using SSE4.2 SIMD
   * instructions.
   *
   * This function divides each element in the array by a scalar value using
   * SSE4.2 SIMD instructions. If SSE4.2 is not supported, it falls back to the
   * generic implementation.
   *
   * @param a The array to be divided (modified in-place).
   * @param scalar The scalar value to divide each element by.
   * @param size The number of elements in the array.
   */
  static void DivScalarSse42(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  /**
   * @brief Divides each element of an array by a scalar using SSE4.1 SIMD
   * instructions.
   *
   * This function divides each element in the array by a scalar value using
   * SSE4.1 SIMD instructions. If SSE4.1 is not supported, it falls back to the
   * generic implementation.
   *
   * @param a The array to be divided (modified in-place).
   * @param scalar The scalar value to divide each element by.
   * @param size The number of elements in the array.
   */
  static void DivScalarSse41(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  /**
   * @brief Divides each element of an array by a scalar using SSSE3 SIMD
   * instructions.
   *
   * This function divides each element in the array by a scalar value using
   * SSSE3 SIMD instructions. If SSSE3 is not supported, it falls back to the
   * generic implementation.
   *
   * @param a The array to be divided (modified in-place).
   * @param scalar The scalar value to divide each element by.
   * @param size The number of elements in the array.
   */
  static void DivScalarSsse3(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  /**
   * @brief Divides each element of an array by a scalar using SSE3 SIMD
   * instructions.
   *
   * This function divides each element in the array by a scalar value using
   * SSE3 SIMD instructions. If SSE3 is not supported, it falls back to the
   * generic implementation.
   *
   * @param a The array to be divided (modified in-place).
   * @param scalar The scalar value to divide each element by.
   * @param size The number of elements in the array.
   */
  static void DivScalarSse3(T* a, T scalar, std::size_t size) {
    DivScalarFallback(a, scalar, size);
  }

  /**
   * @brief Divides each element of an array by a scalar in a generic way.
   *
   * This is the fallback function for dividing each element of an array by a
   * scalar without SIMD support.
   *
   * @param a The array to be divided (modified in-place).
   * @param scalar The scalar value to divide each element by.
   * @param size The number of elements in the array.
   */
  static void DivScalarFallback(T* a, T scalar, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      a[i] /= scalar;
    }
  }

  // END: division scalar
  //----------------------------------------------------------------------------

  // BEGIN: comparison array
  //----------------------------------------------------------------------------

  /**
   * @brief Compares two arrays using AVX2 SIMD instructions.
   *
   * This function compares two arrays element-wise using AVX2 SIMD
   * instructions. It returns an integer indicating whether the first array is
   * less than, equal to, or greater than the second array. If AVX2 is not
   * supported, it falls back to the generic implementation.
   *
   * @param a The first array to be compared.
   * @param b The second array to be compared.
   * @param size The number of elements in the arrays.
   * @return -1 if `a < b`, 0 if `a == b`, 1 if `a > b`.
   */
  static int CmpAvx2(const T* a, const T* b, std::size_t size) {
    return CmpAvx(a, b, size);
  }

  /**
   * @brief Compares two arrays using AVX SIMD instructions.
   *
   * This function compares two arrays element-wise using AVX SIMD instructions.
   * It returns an integer indicating whether the first array is less than,
   * equal to, or greater than the second array. If AVX is not supported, it
   * falls back to the generic implementation.
   *
   * @param a The first array to be compared.
   * @param b The second array to be compared.
   * @param size The number of elements in the arrays.
   * @return -1 if `a < b`, 0 if `a == b`, 1 if `a > b`.
   */
  static int CmpAvx(const T* a, const T* b, std::size_t size) {
    return CmpFallback(a, b, size);
  }

  /**
   * @brief Compares two arrays using SSE4.2 SIMD instructions.
   *
   * This function compares two arrays element-wise using SSE4.2 SIMD
   * instructions. It returns an integer indicating whether the first array is
   * less than, equal to, or greater than the second array. If SSE4.2 is not
   * supported, it falls back to the SSE3 implementation.
   *
   * @param a The first array to be compared.
   * @param b The second array to be compared.
   * @param size The number of elements in the arrays.
   * @return -1 if `a < b`, 0 if `a == b`, 1 if `a > b`.
   */
  static int CmpSse42(const T* a, const T* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  /**
   * @brief Compares two arrays using SSE4.1 SIMD instructions.
   *
   * This function compares two arrays element-wise using SSE4.1 SIMD
   * instructions. It returns an integer indicating whether the first array is
   * less than, equal to, or greater than the second array. If SSE4.1 is not
   * supported, it falls back to the SSE3 implementation.
   *
   * @param a The first array to be compared.
   * @param b The second array to be compared.
   * @param size The number of elements in the arrays.
   * @return -1 if `a < b`, 0 if `a == b`, 1 if `a > b`.
   */
  static int CmpSse41(const T* a, const T* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  /**
   * @brief Compares two arrays using SSSE3 SIMD instructions.
   *
   * This function compares two arrays element-wise using SSSE3 SIMD
   * instructions. It returns an integer indicating whether the first array is
   * less than, equal to, or greater than the second array. If SSSE3 is not
   * supported, it falls back to the SSE3 implementation.
   *
   * @param a The first array to be compared.
   * @param b The second array to be compared.
   * @param size The number of elements in the arrays.
   * @return -1 if `a < b`, 0 if `a == b`, 1 if `a > b`.
   */
  static int CmpSsse3(const T* a, const T* b, std::size_t size) {
    return CmpSse3(a, b, size);
  }

  /**
   * @brief Compares two arrays using SSE3 SIMD instructions.
   *
   * This function compares two arrays element-wise using SSE3 SIMD
   * instructions. It returns an integer indicating whether the first array is
   * less than, equal to, or greater than the second array. If SSE3 is not
   * supported, it falls back to the generic implementation.
   *
   * @param a The first array to be compared.
   * @param b The second array to be compared.
   * @param size The number of elements in the arrays.
   * @return -1 if `a < b`, 0 if `a == b`, 1 if `a > b`.
   */
  static int CmpSse3(const T* a, const T* b, std::size_t size) {
    return CmpFallback(a, b, size);
  }

  /**
   * @brief Compares two arrays element-wise in a generic way.
   *
   * This is the fallback function for comparing two arrays element-wise without
   * SIMD support.
   *
   * @param a The first array to be compared.
   * @param b The second array to be compared.
   * @param size The number of elements in the arrays.
   * @return -1 if `a < b`, 0 if `a == b`, 1 if `a > b`.
   */
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