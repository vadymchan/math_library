/**
 * @file utils.h
 */

#ifndef MATH_LIBRARY_UTILS_H
#define MATH_LIBRARY_UTILS_H

namespace math {

constexpr float g_kPi               = 3.14159265358979323846f;
constexpr float g_kDefaultTolerance = 1e-8f;

template <typename T>
  requires std::floating_point<T>
constexpr auto g_radianToDegree(T radian) -> T {
  constexpr T kToDegree = static_cast<T>(180) / g_kPi;
  return radian * kToDegree;
}

template <typename T>
  requires std::floating_point<T>
constexpr auto g_degreeToRadian(T degree) -> T {
  constexpr T kToRadian = g_kPi / static_cast<T>(180);
  return degree * kToRadian;
}

template <typename T>
auto g_abs(T value) -> T {
  if constexpr (std::is_integral_v<T>) {
    return value;
  } else {
    return std::abs(value);
  }
}

template <typename T>
  requires std::floating_point<T>
constexpr auto g_isNearlyEqual(T a, T b, T tolerance = g_kDefaultTolerance)
    -> bool {
  return g_abs(a - b) <= tolerance;
}

template <typename T>
  requires std::floating_point<T>
constexpr auto g_isNearlyZero(T a, T tolerance = g_kDefaultTolerance) -> bool {
  return g_abs(a) <= tolerance;
}

}  // namespace math

#endif  // MATH_LIBRARY_UTILS_H