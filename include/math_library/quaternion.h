/**
 * @file quaternion.h
 */

#ifndef MATH_LIBRARY_QUATERNION_H
#define MATH_LIBRARY_QUATERNION_H

#include "matrix.h"
#include "vector.h"

namespace math {

template <typename T>
class Quaternion {
  public:

  private:

  // m_data_ is row vector under the hood
  Vector4D<T> m_data_;
};

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

}  // namespace math

#endif