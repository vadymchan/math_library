/**
 * @file utils.h
 */

#ifndef MATH_LIBRARY_UTILS_H
#define MATH_LIBRARY_UTILS_H

namespace math {

template <typename T>
auto abs(T value) -> T {
  if constexpr (std::is_integral<T>::value) {
    return value;
  } else {
    return std::abs(value);
  }
}

}  // namespace math

#endif  // MATH_LIBRARY_UTILS_H