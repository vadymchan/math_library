/**
 * @file utils.h
 */

#ifndef MATH_LIBRARY_UTILS_H
#define MATH_LIBRARY_UTILS_H

namespace math {

static constexpr float g_kPi = 3.141592653f;

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

}  // namespace math

#endif  // MATH_LIBRARY_UTILS_H