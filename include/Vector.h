/**
 * @file Vector.h
 */

#ifndef MATH_LIBRARY_VECTOR_H
#define MATH_LIBRARY_VECTOR_H


#include "Matrix.h"

namespace math {

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
class Vector {
  public:

// Vector of doubles
template <unsigned int Size, Options Option = Options::RowMajor>
using VectorXd = Matrix<double, Size, 1, Option>;

// Vector of ints
template <unsigned int Size, Options Option = Options::RowMajor>
using VectorXi = Matrix<int, Size, 1, Option>;

// Row vector of floats
template <unsigned int Size, Options Option = Options::RowMajor>
using RowVectorXf = Matrix<float, 1, Size, Option>;

// Row vector of doubles
template <unsigned int Size, Options Option = Options::RowMajor>
using RowVectorXd = Matrix<double, 1, Size, Option>;

// Row vector of ints
template <unsigned int Size, Options Option = Options::RowMajor>
using RowVectorXi = Matrix<int, 1, Size, Option>;

// Specific size vectors
using Vector2f    = VectorXf<2>;
using Vector3f    = VectorXf<3>;
using Vector4f    = VectorXf<4>;
using RowVector2f = RowVectorXf<2>;
using RowVector3f = RowVectorXf<3>;
using RowVector4f = RowVectorXf<4>;

  private:
  using UnderlyingType = std::conditional_t<Option == Options::RowMajor,
                                            Matrix<T, Size, 1, Option>,
                                            Matrix<T, 1, Size, Option>>;

  UnderlyingType m_dataStorage_;
};

}  // namespace math

#endif