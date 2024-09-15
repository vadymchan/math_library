/**
 * @file utils.h
 * @brief Utility functions and constants for mathematical operations.
 *
 * This file contains various utility functions and constants, including
 * conversions between radians and degrees, comparison functions with tolerance,
 * and an absolute value function. It is a part of the math library, providing
 * essential mathematical helpers for handling floating-point operations.
 */

#ifndef MATH_LIBRARY_UTILS_H
#define MATH_LIBRARY_UTILS_H

#include <cmath>

namespace math {

/**
 * @brief Constant representing the value of Pi.
 *
 * This constant holds the value of Pi, which is used in various
 * mathematical operations involving angles, circles, and trigonometry.
 */
constexpr float g_kPi               = 3.14159265358979323846f;

/**
 * @brief Default tolerance value for floating-point comparisons.
 *
 * This constant defines the default tolerance level used when comparing two
 * floating-point numbers for near equality. A small tolerance is necessary due
 * to precision issues in floating-point arithmetic.
 */
constexpr float g_kDefaultTolerance = 1e-8f;

/**
 * @brief Converts radians to degrees.
 *
 * This function converts an angle in radians to its equivalent in degrees.
 *
 * @tparam T The floating-point type (e.g., `float`, `double`) of the input.
 * @param radian The angle in radians to be converted.
 * @return The equivalent angle in degrees.
 */
template <typename T>
  requires std::floating_point<T>
constexpr auto g_radianToDegree(T radian) -> T {
  constexpr T kToDegree = static_cast<T>(180) / g_kPi;
  return radian * kToDegree;
}

/**
 * @brief Converts degrees to radians.
 *
 * This function converts an angle in degrees to its equivalent in radians.
 *
 * @tparam T The floating-point type (e.g., `float`, `double`) of the input.
 * @param degree The angle in degrees to be converted.
 * @return The equivalent angle in radians.
 */
template <typename T>
  requires std::floating_point<T>
constexpr auto g_degreeToRadian(T degree) -> T {
  constexpr T kToRadian = g_kPi / static_cast<T>(180);
  return degree * kToRadian;
}

/**
 * @brief Computes the absolute value of a number.
 *
 * This function returns the absolute value of the input. For integral types, it
 * returns the value itself since the standard `abs` function does not apply.
 * For floating-point types, it returns the result of `std::abs`.
 *
 * @tparam T The type of the input (either integral or floating-point).
 * @param value The value for which the absolute value is to be computed.
 * @return The absolute value of the input.
 */
template <typename T>
auto g_abs(T value) -> T {
  if constexpr (std::is_integral_v<T>) {
    return value;
  } else {
    return std::abs(value);
  }
}

/**
 * @brief Checks if two floating-point numbers are nearly equal.
 *
 * This function compares two floating-point numbers to see if they are nearly
 * equal, within a specified tolerance. This is useful in cases where direct
 * equality checks may fail due to precision issues.
 *
 * @tparam T The floating-point type (e.g., `float`, `double`) of the input.
 * @param a The first number.
 * @param b The second number.
 * @param tolerance The allowable tolerance for comparison (default is
 * `g_kDefaultTolerance`).
 * @return `true` if the numbers are nearly equal, `false` otherwise.
 */
template <typename T>
  requires std::floating_point<T>
constexpr auto g_isNearlyEqual(T a, T b, T tolerance = g_kDefaultTolerance)
    -> bool {
  return g_abs(a - b) <= tolerance;
}

/**
 * @brief Checks if a floating-point number is nearly zero.
 *
 * This function checks if a floating-point number is close to zero, within a
 * specified tolerance.
 *
 * @tparam T The floating-point type (e.g., `float`, `double`) of the input.
 * @param a The number to check.
 * @param tolerance The allowable tolerance for comparison (default is
 * `g_kDefaultTolerance`).
 * @return `true` if the number is nearly zero, `false` otherwise.
 */
template <typename T>
  requires std::floating_point<T>
constexpr auto g_isNearlyZero(T a, T tolerance = g_kDefaultTolerance) -> bool {
  return g_abs(a) <= tolerance;
}

}  // namespace math

#endif  // MATH_LIBRARY_UTILS_H