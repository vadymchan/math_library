/**
 * @file graphics.h
 */

#ifndef MATH_LIBRARY_GRAPHICS_H
#define MATH_LIBRARY_GRAPHICS_H

#include "matrix.h"

namespace math {

// lookAtRH(eyePosition, centerPosition, upDirection)
// lookAtLH(eyePosition, centerPosition, upDirection)

template <typename T, Options Option = Options::RowMajor>
auto g_lookAtRH(const Matrix<T, 3, 1>& eye,
                const Matrix<T, 3, 1>& target,
                const Matrix<T, 3, 1>& worldUp) -> Matrix<T, 4, 4> {
  Matrix<T, 3, 1> forward = (eye - target).normalize();
  Matrix<T, 3, 1> right   = worldUp.cross(forward).normalize();
  Matrix<T, 3, 1> up      = forward.cross(right);

  Matrix<T, 4, 4> viewMatrix = Matrix<T, 4, 4>::Identity();

  if constexpr (Option == Options::RowMajor) {
    viewMatrix(0, 0) = right(0, 0);
    viewMatrix(1, 0) = right(1, 0);
    viewMatrix(2, 0) = right(2, 0);
    viewMatrix(0, 3) = -right.dot(eye);

    viewMatrix(1, 1) = up(0, 0);
    viewMatrix(2, 1) = up(1, 0);
    viewMatrix(3, 1) = up(2, 0);
    viewMatrix(1, 3) = -up.dot(eye);

    viewMatrix(1, 2) = forward(0, 0);
    viewMatrix(2, 2) = forward(1, 0);
    viewMatrix(3, 2) = forward(2, 0);
    viewMatrix(2, 3) = -forward.dot(eye);
  } else if constexpr (Option == Options::ColumnMajor) {
    viewMatrix(0, 0) = right(0, 0);
    viewMatrix(0, 1) = right(1, 0);
    viewMatrix(0, 2) = right(2, 0);
    viewMatrix(3, 0) = -right.dot(eye);

    viewMatrix(1, 0) = up(0, 0);
    viewMatrix(1, 1) = up(1, 0);
    viewMatrix(1, 2) = up(2, 0);
    viewMatrix(3, 1) = -up.dot(eye);

    viewMatrix(2, 0) = forward(0, 0);
    viewMatrix(2, 1) = forward(1, 0);
    viewMatrix(2, 2) = forward(2, 0);
    viewMatrix(3, 2) = -forward.dot(eye);
  }

  return viewMatrix;
}

template <typename T, Options Option = Options::RowMajor>
auto g_lookAtLh(const Matrix<T, 3, 1>& eye,
                const Matrix<T, 3, 1>& target,
                const Matrix<T, 3, 1>& worldUp) -> Matrix<T, 4, 4> {
  Matrix<T, 3, 1> forward = (target - eye).normalize();
  Matrix<T, 3, 1> right   = worldUp.cross(forward).normalize();
  Matrix<T, 3, 1> up      = forward.cross(right);

  Matrix<T, 4, 4> viewMatrix = Matrix<T, 4, 4>::Identity();

  if constexpr (Option == Options::RowMajor) {
    viewMatrix(0, 0) = right(0, 0);
    viewMatrix(0, 1) = right(1, 0);
    viewMatrix(0, 2) = right(2, 0);
    viewMatrix(0, 3) = -right.dot(eye);

    viewMatrix(1, 0) = up(0, 0);
    viewMatrix(1, 1) = up(1, 0);
    viewMatrix(1, 2) = up(2, 0);
    viewMatrix(1, 3) = -up.dot(eye);

    viewMatrix(2, 0) = forward(0, 0);
    viewMatrix(2, 1) = forward(1, 0);
    viewMatrix(2, 2) = forward(2, 0);
    viewMatrix(2, 3) = -forward.dot(eye);
  } else  // COLUMN_MAJOR
  {
    viewMatrix(0, 0) = right(0, 0);
    viewMatrix(1, 0) = right(1, 0);
    viewMatrix(2, 0) = right(2, 0);
    viewMatrix(3, 0) = -right.dot(eye);

    viewMatrix(0, 1) = up(0, 0);
    viewMatrix(1, 1) = up(1, 0);
    viewMatrix(2, 1) = up(2, 0);
    viewMatrix(3, 1) = -up.dot(eye);

    viewMatrix(0, 2) = forward(0, 0);
    viewMatrix(1, 2) = forward(1, 0);
    viewMatrix(2, 2) = forward(2, 0);
    viewMatrix(3, 2) = -forward.dot(eye);
  }

  return viewMatrix;
}

// lookToRH(eyePosition, centerPosition, upDirection)
// lookToLH(eyePosition, centerPosition, upDirection)

template <typename T, Options Option>
auto g_lookToRh(const Matrix<T, 3, 1>& eye,
                const Matrix<T, 3, 1>& direction,
                const Matrix<T, 3, 1>& worldUp) -> Matrix<T, 4, 4> {
  Matrix<T, 3, 1> forward = direction.normalize();
  Matrix<T, 3, 1> right   = worldUp.cross(forward).normalize();
  Matrix<T, 3, 1> up      = forward.cross(right);

  Matrix<T, 4, 4> viewMatrix = Matrix<T, 4, 4>::Identity();

  if constexpr (Option == Options::RowMajor) {
    viewMatrix(0, 0) = right(0, 0);
    viewMatrix(0, 1) = right(1, 0);
    viewMatrix(0, 2) = right(2, 0);
    viewMatrix(0, 3) = -right.dot(eye);

    viewMatrix(1, 0) = up(0, 0);
    viewMatrix(1, 1) = up(1, 0);
    viewMatrix(1, 2) = up(2, 0);
    viewMatrix(1, 3) = -up.dot(eye);

    viewMatrix(2, 0) = -forward(0, 0);
    viewMatrix(2, 1) = -forward(1, 0);
    viewMatrix(2, 2) = -forward(2, 0);
    viewMatrix(2, 3) = forward.dot(eye);
  } else if constexpr (Option == Options::ColumnMajor) {
    viewMatrix(0, 0) = right(0, 0);
    viewMatrix(1, 0) = right(1, 0);
    viewMatrix(2, 0) = right(2, 0);
    viewMatrix(3, 0) = -right.dot(eye);

    viewMatrix(0, 1) = up(0, 0);
    viewMatrix(1, 1) = up(1, 0);
    viewMatrix(2, 1) = up(2, 0);
    viewMatrix(3, 1) = -up.dot(eye);

    viewMatrix(0, 2) = -forward(0, 0);
    viewMatrix(1, 2) = -forward(1, 0);
    viewMatrix(2, 2) = -forward(2, 0);
    viewMatrix(3, 2) = forward.dot(eye);
  }

  return viewMatrix;
}

template <typename T, Options Option = Options::RowMajor>
auto g_lookToLh(const Matrix<T, 3, 1>& eye,
                const Matrix<T, 3, 1>& direction,
                const Matrix<T, 3, 1>& worldUp) -> Matrix<T, 4, 4> {
  Matrix<T, 3, 1> forward = direction.normalize();
  Matrix<T, 3, 1> right   = worldUp.cross(forward).normalize();
  Matrix<T, 3, 1> up      = forward.cross(right);

  Matrix<T, 4, 4> viewMatrix = Matrix<T, 4, 4>::Identity();

  if constexpr (Option == Options::RowMajor) {
    viewMatrix(0, 0) = right(0, 0);
    viewMatrix(0, 1) = right(1, 0);
    viewMatrix(0, 2) = right(2, 0);
    viewMatrix(0, 3) = -right.dot(eye);

    viewMatrix(1, 0) = up(0, 0);
    viewMatrix(1, 1) = up(1, 0);
    viewMatrix(1, 2) = up(2, 0);
    viewMatrix(1, 3) = -up.dot(eye);

    viewMatrix(2, 0) = forward(0, 0);
    viewMatrix(2, 1) = forward(1, 0);
    viewMatrix(2, 2) = forward(2, 0);
    viewMatrix(2, 3) = forward.dot(eye);
  } else if constexpr (Option == Options::ColumnMajor) {
    viewMatrix(0, 0) = right(0, 0);
    viewMatrix(1, 0) = right(1, 0);
    viewMatrix(2, 0) = right(2, 0);
    viewMatrix(3, 0) = -right.dot(eye);

    viewMatrix(0, 1) = up(0, 0);
    viewMatrix(1, 1) = up(1, 0);
    viewMatrix(2, 1) = up(2, 0);
    viewMatrix(3, 1) = -up.dot(eye);

    viewMatrix(0, 2) = forward(0, 0);
    viewMatrix(1, 2) = forward(1, 0);
    viewMatrix(2, 2) = forward(2, 0);
    viewMatrix(3, 2) = forward.dot(eye);
  }

  return viewMatrix;
}

// perspectiveFovRH(fov, aspectRatio, nearPlane, farPlane)
// perspectiveFovLH(fov, aspectRatio, nearPlane, farPlane)
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveFovRh(T fov, T aspectRatio, T nearPlane, T farPlane)
    -> Matrix<T, 4, 4> {
  assert(abs(aspectRatio - std::numeric_limits<T>::epsilon())
         > static_cast<T>(0));

  T tanHalfFovy = tan(fov / static_cast<T>(2));

  Matrix<T, 4, 4> result(0);
  result(0, 0) = static_cast<T>(1) / (aspectRatio * tanHalfFovy);
  result(1, 1) = static_cast<T>(1) / (tanHalfFovy);
  result(2, 2) = farPlane / (nearPlane - farPlane);

  if constexpr (Option == Options::RowMajor) {
    result(2, 3) = -1;
    result(3, 2) = -(farPlane * nearPlane) / (farPlane - nearPlane);
  } else if constexpr (Option == Options::ColumnMajor) {
    result(2, 3) = -(farPlane * nearPlane) / (farPlane - nearPlane);
    result(3, 2) = -1;
  }

  return result;
}

template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveFovLh(T fov, T aspectRatio, T nearPlane, T farPlane)
    -> Matrix<T, 4, 4> {
  assert(abs(aspectRatio - std::numeric_limits<T>::epsilon())
         > static_cast<T>(0));

  T tanHalfFovy = tan(fov / static_cast<T>(2));

  Matrix<T, 4, 4> result(0);
  result(0, 0) = static_cast<T>(1) / (aspectRatio * tanHalfFovy);
  result(1, 1) = static_cast<T>(1) / (tanHalfFovy);
  result(2, 2) = farPlane / (farPlane - nearPlane);

  if constexpr (Option == Options::RowMajor) {
    result(2, 3) = 1;
    result(3, 2) = -(farPlane * nearPlane) / (farPlane - nearPlane);
  } else if constexpr (Option == Options::ColumnMajor) {
    result(3, 2) = 1;
    result(2, 3) = -(farPlane * nearPlane) / (farPlane - nearPlane);
  }

  return result;
}

// perspectiveOffCenter(left, right, bottom, top, nearPlane, farPlane)

template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveOffCenter(
    T left, T right, T bottom, T top, T nearPlane, T farPlane)
    -> Matrix<T, 4, 4> {
  assert(abs(right - left) > std::numeric_limits<T>::epsilon()
         && "Right and left values cannot be equal");
  assert(abs(top - bottom) > std::numeric_limits<T>::epsilon()
         && "Top and bottom values cannot be equal");
  assert(abs(farPlane - nearPlane) > std::numeric_limits<T>::epsilon()
         && "Far and near plane values cannot be equal");

  Matrix<T, 4, 4> result(0);
  result(0, 0) = (static_cast<T>(2) * nearPlane) / (right - left);
  result(1, 1) = (static_cast<T>(2) * nearPlane) / (top - bottom);
  result(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);

  if constexpr (Option == Options::RowMajor) {
    result(2, 0) = (right + left) / (right - left);
    result(2, 1) = (top + bottom) / (top - bottom);
    result(2, 3) = -1;
    result(3, 2)
        = -(static_cast<T>(2) * farPlane * nearPlane) / (farPlane - nearPlane);
  } else if constexpr (Option == Options::ColumnMajor) {
    result(0, 2) = (right + left) / (right - left);
    result(1, 2) = (top + bottom) / (top - bottom);
    result(3, 2) = -1;
    result(2, 3)
        = -(static_cast<T>(2) * farPlane * nearPlane) / (farPlane - nearPlane);
  }

  return result;
}

// orthoRH(left, right, bottom, top, nearPlane, farPlane)
// orthoLH(left, right, bottom, top, nearPlane, farPlane)

template <typename T, Options Option = Options::RowMajor>
auto g_orthoRh(T left, T right, T bottom, T top, T nearPlane, T farPlane)
    -> Matrix<T, 4, 4> {
  assert(abs(right - left) > std::numeric_limits<T>::epsilon()
         && "Right and left values cannot be equal");
  assert(abs(top - bottom) > std::numeric_limits<T>::epsilon()
         && "Top and bottom values cannot be equal");
  assert(abs(farPlane - nearPlane) > std::numeric_limits<T>::epsilon()
         && "Far and near plane values cannot be equal");

  Matrix<T, 4, 4> result = Matrix<T, 4, 4>::Identity();
  result(0, 0)           = static_cast<T>(2) / (right - left);
  result(1, 1)           = static_cast<T>(2) / (top - bottom);
  result(2, 2)           = -static_cast<T>(2) / (farPlane - nearPlane);

  if constexpr (Option == Options::RowMajor) {
    result(0, 3) = -(right + left) / (right - left);
    result(1, 3) = -(top + bottom) / (top - bottom);
    result(2, 3) = -(farPlane + nearPlane) / (farPlane - nearPlane);
  } else if constexpr (Option == Options::ColumnMajor) {
    result(3, 0) = -(right + left) / (right - left);
    result(3, 1) = -(top + bottom) / (top - bottom);
    result(3, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
  }

  return result;
}

template <typename T, Options Option = Options::RowMajor>
auto g_orthoLh(T left, T right, T bottom, T top, T nearPlane, T farPlane)
    -> Matrix<T, 4, 4> {
  assert(abs(right - left) > std::numeric_limits<T>::epsilon()
         && "Right and left values cannot be equal");
  assert(abs(top - bottom) > std::numeric_limits<T>::epsilon()
         && "Top and bottom values cannot be equal");
  assert(abs(farPlane - nearPlane) > std::numeric_limits<T>::epsilon()
         && "Far and near plane values cannot be equal");

  Matrix<T, 4, 4> result = Matrix<T, 4, 4>::Identity();
  result(0, 0)           = static_cast<T>(2) / (right - left);
  result(1, 1)           = static_cast<T>(2) / (top - bottom);
  result(2, 2)           = static_cast<T>(2) / (farPlane - nearPlane);

  if constexpr (Option == Options::RowMajor) {
    result(0, 3) = -(right + left) / (right - left);
    result(1, 3) = -(top + bottom) / (top - bottom);
    result(2, 3) = -(farPlane + nearPlane) / (farPlane - nearPlane);
  } else if constexpr (Option == Options::ColumnMajor) {
    result(3, 0) = -(right + left) / (right - left);
    result(3, 1) = -(top + bottom) / (top - bottom);
    result(3, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
  }

  return result;
}

// TODO: consider moving directional vector variables to a separate file

// TODO: consider use another approach for defining these vectors (e.g. static methods, functions)

// TODO: make these constexpr
template <typename T, Options Option = Options::RowMajor>
const Vector<T, 3, Option> g_kUpVector = {0, 1, 0};

template <typename T, Options Option = Options::RowMajor>
const Vector<T, 3, Option> g_kDownVector = {0, -1, 0};

template <typename T, Options Option = Options::RowMajor>
const Vector<T, 3, Option> g_kForwardVector = {0, 0, 1};

template <typename T, Options Option = Options::RowMajor>
const Vector<T, 3, Option> g_kBackwardVector = {0, 0, -1};

template <typename T, Options Option = Options::RowMajor>
const Vector<T, 3, Option> g_kRightVector = {1, 0, 0};

template <typename T, Options Option = Options::RowMajor>
const Vector<T, 3, Option> g_kLeftVector = {-1, 0, 0};

template <typename T, Options Option = Options::RowMajor>
const auto g_kZeroVector = Vector<T, 3, Option>(0);

template <typename T, Options Option = Options::RowMajor>
const auto g_kOneVector = Vector<T, 3, Option>(1);

/**
 * @brief Normalized one vector.
 */
template <typename T, Options Option = Options::RowMajor>
const auto g_kUnitVector = Vector<T, 3, Option>(0.57735026919);

// TODO: type postfix naming is not the best. Maybe use a different approach

// I wanted to use this approach, but because of 'static initialization order
// fiasco' it wasn't possible. template <Options Option = Options::RowMajor>
// const auto& g_kUpVectorf = g_kUpVector<float, Option>();
// template <Options Option = Options::RowMajor>
// const auto g_kDownVectorf = g_kDownVector<float, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kForwardVectorf = g_kForwardVector<float, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kBackwardVectorf = g_kBackwardVector<float, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kRightVectorf = g_kRightVector<float, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kLeftVectorf = g_kLeftVector<float, Option>;
//
// template <Options Option = Options::RowMajor>
// const auto g_kUpVectord = g_kUpVector<double, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kDownVectord = g_kDownVector<double, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kForwardVectord = g_kForwardVector<double, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kBackwardVectord = g_kBackwardVector<double, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kRightVectord = g_kRightVector<double, Option>;
//
// template <Options Option = Options::RowMajor>
// const auto g_kUpVectori = g_kUpVector<int, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kDownVectori = g_kDownVector<int, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kForwardVectori = g_kForwardVector<int, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kBackwardVectori = g_kBackwardVector<int, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kRightVectori = g_kRightVector<int, Option>;
// template <Options Option = Options::RowMajor>
// const auto g_kLeftVectori = g_kLeftVector<int, Option>;

// BTW, if you want to know how to resolve this problem, here's the solution (it
// just didn't fit to my vision) template <typename T, Options Option =
// Options::RowMajor> const Vector<T, 3, Option>& GetUpVector() {
//   static const Vector<T, 3, Option> instance{0, 1, 0};
//   return instance;
// }
//
// template <Options Option = Options::RowMajor>
// const auto& g_kUpVectorf = GetUpVector<float, Option>();

}  // namespace math

#endif