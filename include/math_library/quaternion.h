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
};

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

}  // namespace math

#endif