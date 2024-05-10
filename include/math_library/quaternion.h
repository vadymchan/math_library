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
  Quaternion()
      : m_data_(0, 0, 0, 1) {}

  Quaternion(const T x, const T y, const T z, const T w)
      : m_data_(x, y, z, w) {}

  explicit Quaternion(const Vector3D<T>& v, const T w)
      : m_data_(v.x(), v.y(), v.z(), w) {}

  explicit Quaternion(const Vector4D<T>& v)
      : m_data_(v) {}

  Quaternion(const Quaternion& other)
      : m_data_(other.m_data_) {}


  private:

  // m_data_ is row vector under the hood
  Vector4D<T> m_data_;
};

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

}  // namespace math

#endif