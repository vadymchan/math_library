/**
 * @file graphics.h
 */

#ifndef MATH_LIBRARY_GRAPHICS_H
#define MATH_LIBRARY_GRAPHICS_H

#include "matrix.h"

namespace math {

// clang-format off

// BEGIN: rotation matrix creation functions
// ----------------------------------------------------------------------------

template <typename T, Options Option = Options::RowMajor>
auto g_rotateRhX(T angle) -> Matrix<T, 4, 4, Option> {
  const T cosAngle = std::cos(angle);
  const T sinAngle = std::sin(angle);

  Matrix<T, 4, 4, Option> rotateMat;

  if constexpr (Option == Options::RowMajor) {
    rotateMat << 1,   0,          0,          0,
                 0,   cosAngle,   sinAngle,   0,
                 0,  -sinAngle,   cosAngle,   0,
                 0,   0,          0,          1;
  } else {
    rotateMat << 1,   0,          0,          0,
                 0,   cosAngle,  -sinAngle,   0,
                 0,   sinAngle,   cosAngle,   0,
                 0,   0,          0,          1;
  }
  return rotateMat;
}

template <typename T, Options Option = Options::RowMajor>
auto g_rotateRhY(T angle) -> Matrix<T, 4, 4, Option> {
  const T cosAngle = std::cos(angle);
  const T sinAngle = std::sin(angle);

  Matrix<T, 4, 4, Option> rotateMat;

  if constexpr (Option == Options::RowMajor) {
    rotateMat <<  cosAngle,   0,  -sinAngle,   0,
                  0,          1,   0,          0,
                  sinAngle,   0,   cosAngle,   0,
                  0,          0,   0,          1;
  } else {
    rotateMat <<  cosAngle,   0,   sinAngle,   0,
                  0,          1,   0,          0,
                 -sinAngle,   0,   cosAngle,   0,
                  0,          0,   0,          1;
  }
  return rotateMat;
}

template <typename T, Options Option = Options::RowMajor>
auto g_rotateRhZ(T angle) -> Matrix<T, 4, 4, Option> {
  const T cosAngle = std::cos(angle);
  const T sinAngle = std::sin(angle);

  Matrix<T, 4, 4, Option> rotateMat;

  if constexpr (Option == Options::RowMajor) {
    rotateMat <<  cosAngle,     sinAngle,  0,  0,
                 -sinAngle,     cosAngle,  0,  0,
                  0,            0,         1,  0,
                  0,            0,         0,  1;
  } else {
    rotateMat <<  cosAngle,    -sinAngle,  0,  0,
                  sinAngle,     cosAngle,  0,  0,
                  0,            0,         1,  0,
                  0,            0,         0,  1;
  }
  return rotateMat;
}

template <typename T, Options Option = Options::RowMajor>
auto g_rotateLhX(T angle) -> Matrix<T, 4, 4, Option> {
  return g_rotateRhX<T, Option>(-angle);
}

template <typename T, Options Option = Options::RowMajor>
auto g_rotateLhY(T angle) -> Matrix<T, 4, 4, Option> {
  return g_rotateRhY<T, Option>(-angle);
}

template <typename T, Options Option = Options::RowMajor>
auto g_rotateLhZ(T angle) -> Matrix<T, 4, 4, Option> {
  return g_rotateRhZ<T, Option>(-angle);
}

// END: rotation matrix creation functions
// ----------------------------------------------------------------------------

// clang-format on

// BEGIN: view matrix creation functions
// ----------------------------------------------------------------------------

template <typename T, Options Option = Options::RowMajor>
auto g_lookAtRh(const Vector3D<T, Option>& eye,
                const Vector3D<T, Option>& target,
                const Vector3D<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  auto forward = target - eye;
  forward.normalize();
  auto right = worldUp.cross(forward);
  right.normalize();
#else
  auto forward = (target - eye).normalize();
  auto right   = worldUp.cross(forward).normalize();
#endif
  auto up = forward.cross(right);

  auto viewMatrix = Matrix<T, 4, 4, Option>::Identity();

  if constexpr (Option == Options::RowMajor) {
    viewMatrix(0, 0) = right.x();
    viewMatrix(0, 1) = right.y();
    viewMatrix(0, 2) = right.z();
    viewMatrix(3, 0) = -right.dot(eye);

    viewMatrix(1, 0) = up.x();
    viewMatrix(1, 1) = up.y();
    viewMatrix(1, 2) = up.z();
    viewMatrix(3, 1) = -up.dot(eye);

    // forward - depends on handness
    viewMatrix(2, 0) = -forward.x();
    viewMatrix(2, 1) = -forward.y();
    viewMatrix(2, 2) = -forward.z();
    viewMatrix(3, 2) = -forward.dot(eye);
  } else if constexpr (Option == Options::ColumnMajor) {
    viewMatrix(0, 0) = right.x();
    viewMatrix(1, 0) = right.y();
    viewMatrix(2, 0) = right.z();
    viewMatrix(0, 3) = -right.dot(eye);

    viewMatrix(0, 1) = up.x();
    viewMatrix(1, 1) = up.y();
    viewMatrix(2, 1) = up.z();
    viewMatrix(1, 3) = -up.dot(eye);

    // forward - depends on handness
    viewMatrix(0, 2) = -forward.x();
    viewMatrix(1, 2) = -forward.y();
    viewMatrix(2, 2) = -forward.z();
    viewMatrix(2, 3) = -forward.dot(eye);
  }

  return viewMatrix;
}

template <typename T, Options Option = Options::RowMajor>
auto g_lookAtLh(const Vector3D<T, Option>& eye,
                const Vector3D<T, Option>& target,
                const Vector3D<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  auto forward = target - eye;
  forward.normalize();
  auto right = worldUp.cross(forward);
  right.normalize();
#else
  auto forward = (target - eye).normalize();
  auto right   = worldUp.cross(forward).normalize();
#endif

  auto up = forward.cross(right);

  auto viewMatrix = Matrix<T, 4, 4, Option>::Identity();

  if constexpr (Option == Options::RowMajor) {
    viewMatrix(0, 0) = right.x();
    viewMatrix(0, 1) = right.y();
    viewMatrix(0, 2) = right.z();
    viewMatrix(3, 0) = -right.dot(eye);

    viewMatrix(1, 0) = up.x();
    viewMatrix(1, 1) = up.y();
    viewMatrix(1, 2) = up.z();
    viewMatrix(3, 1) = -up.dot(eye);

    // forward - depends on handness
    viewMatrix(2, 0) = forward.x();
    viewMatrix(2, 1) = forward.y();
    viewMatrix(2, 2) = forward.z();
    viewMatrix(3, 2) = -forward.dot(eye);
  } else if constexpr (Option == Options::ColumnMajor) {
    viewMatrix(0, 0) = right.x();
    viewMatrix(1, 0) = right.y();
    viewMatrix(2, 0) = right.z();
    viewMatrix(0, 3) = -right.dot(eye);

    viewMatrix(0, 1) = up.x();
    viewMatrix(1, 1) = up.y();
    viewMatrix(2, 1) = up.z();
    viewMatrix(1, 3) = -up.dot(eye);

    // forward - depends on handness
    viewMatrix(0, 2) = forward.x();
    viewMatrix(1, 2) = forward.y();
    viewMatrix(2, 2) = forward.z();
    viewMatrix(2, 3) = -forward.dot(eye);
  }

  return viewMatrix;
}

template <typename T, Options Option>
auto g_lookToRh(const Vector3D<T, Option>& eye,
                const Vector3D<T, Option>& direction,
                const Vector3D<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  auto forward = direction;
  forward.normalize();
  auto right = worldUp.cross(forward);
  right.normalize();
#else
  auto forward = direction.normalize();
  auto right   = worldUp.cross(forward).normalize();
#endif
  auto up = forward.cross(right);

  auto viewMatrix = Matrix<T, 4, 4, Option>::Identity();

  if constexpr (Option == Options::RowMajor) {
    viewMatrix(0, 0) = right.x();
    viewMatrix(0, 1) = right.y();
    viewMatrix(0, 2) = right.z();
    viewMatrix(3, 0) = -right.dot(eye);

    viewMatrix(1, 0) = up.x();
    viewMatrix(1, 1) = up.y();
    viewMatrix(1, 2) = up.z();
    viewMatrix(3, 1) = -up.dot(eye);

    // forward - depends on handness
    viewMatrix(2, 0) = -forward.x();
    viewMatrix(2, 1) = -forward.y();
    viewMatrix(2, 2) = -forward.z();
    viewMatrix(3, 2) = -forward.dot(eye);
  } else if constexpr (Option == Options::ColumnMajor) {
    viewMatrix(0, 0) = right.x();
    viewMatrix(1, 0) = right.y();
    viewMatrix(2, 0) = right.z();
    viewMatrix(0, 3) = -right.dot(eye);

    viewMatrix(0, 1) = up.x();
    viewMatrix(1, 1) = up.y();
    viewMatrix(2, 1) = up.z();
    viewMatrix(1, 3) = -up.dot(eye);

    // forward - depends on handness
    viewMatrix(0, 2) = -forward.x();
    viewMatrix(1, 2) = -forward.y();
    viewMatrix(2, 2) = -forward.z();
    viewMatrix(2, 3) = -forward.dot(eye);
  }

  return viewMatrix;
}

template <typename T, Options Option = Options::RowMajor>
auto g_lookToLh(const Vector3D<T, Option>& eye,
                const Vector3D<T, Option>& direction,
                const Vector3D<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  auto forward = direction;
  forward.normalize();
  auto right = worldUp.cross(forward);
  right.normalize();
#else
  auto forward = direction.normalize();
  auto right   = worldUp.cross(forward).normalize();
#endif
  auto up = forward.cross(right);

  auto viewMatrix = Matrix<T, 4, 4, Option>::Identity();

  if constexpr (Option == Options::RowMajor) {
    viewMatrix(0, 0) = right.x();
    viewMatrix(0, 1) = right.y();
    viewMatrix(0, 2) = right.z();
    viewMatrix(3, 0) = -right.dot(eye);

    viewMatrix(1, 0) = up.x();
    viewMatrix(1, 1) = up.y();
    viewMatrix(1, 2) = up.z();
    viewMatrix(3, 1) = -up.dot(eye);

    // forward - depends on handness
    viewMatrix(2, 0) = forward.x();
    viewMatrix(2, 1) = forward.y();
    viewMatrix(2, 2) = forward.z();
    viewMatrix(3, 2) = -forward.dot(eye);
  } else if constexpr (Option == Options::ColumnMajor) {
    viewMatrix(0, 0) = right.x();
    viewMatrix(1, 0) = right.y();
    viewMatrix(2, 0) = right.z();
    viewMatrix(0, 3) = -right.dot(eye);

    viewMatrix(0, 1) = up.x();
    viewMatrix(1, 1) = up.y();
    viewMatrix(2, 1) = up.z();
    viewMatrix(1, 3) = -up.dot(eye);

    // forward - depends on handness
    viewMatrix(0, 2) = forward.x();
    viewMatrix(1, 2) = forward.y();
    viewMatrix(2, 2) = forward.z();
    viewMatrix(2, 3) = -forward.dot(eye);
  }

  return viewMatrix;
}

// END: view matrix creation functions
// ----------------------------------------------------------------------------

// clang-format off

// BEGIN: perspective projection creation matrix
// ----------------------------------------------------------------------------

/**
 * Generates a right-handed perspective projection matrix with a depth range of zero to one.
 * @note RH-ZO - Right-Handed, Zero to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhZo(T fovY, T aspect, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  // validate aspect ratio to prevent division by zero
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspeciveMatrix(0);
  perspeciveMatrix(0, 0) = static_cast<T>(1) / (aspect * tanHalfFovY);
  perspeciveMatrix(1, 1) = static_cast<T>(1) / (tanHalfFovY);
  perspeciveMatrix(2, 2) = zFar / (zNear - zFar);  // not the same (depends on handness + NO / LO)
  if constexpr (Option == Options::RowMajor) {
    perspeciveMatrix(3, 2) = -(zFar * zNear) / (zFar - zNear);  // depends on NO / LO
    perspeciveMatrix(2, 3) = -static_cast<T>(1);                // depends on handness (-z)
  } else if (Option == Options::ColumnMajor) {
    perspeciveMatrix(2, 3) = -(zFar * zNear) / (zFar - zNear);  // depends on handness (-z)
    perspeciveMatrix(3, 2) = -static_cast<T>(1);                // depends on NO / LO
  }
  return perspeciveMatrix;
}

/**
 * Generates a right-handed perspective projection matrix with a depth range of negative one to one.
 * @note RH-NO - Right-Handed, Negative One to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhNo(T fovY, T aspect, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  // validate aspect ratio to prevent division by zero
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspeciveMatrix(0);
  perspeciveMatrix(0, 0) = static_cast<T>(1) / (aspect * tanHalfFovY);
  perspeciveMatrix(1, 1) = static_cast<T>(1) / (tanHalfFovY);
  perspeciveMatrix(2, 2) = -(zFar + zNear) / (zFar - zNear); // not the same (depends on handness + NO / LO)
  if constexpr (Option == Options::RowMajor) {
    perspeciveMatrix(3, 2) = -(static_cast<T>(2) * zFar * zNear) / (zFar - zNear); // depends on NO / LO
    perspeciveMatrix(2, 3) = -static_cast<T>(1);                                   // depends on handness (-z)
  } else if (Option == Options::ColumnMajor) {
    perspeciveMatrix(2, 3) = -(static_cast<T>(2) * zFar * zNear) / (zFar - zNear); // depends on handness (-z)
    perspeciveMatrix(3, 2) = -static_cast<T>(1);                                   // depends on NO / LO
  }
  return perspeciveMatrix;
}

/**
 * Generates a left-handed perspective projection matrix with a depth range of zero to one.
 * @note LH-ZO - Left-Handed, Zero to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhZo(T fovY, T aspect, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspeciveMatrix(0);
  perspeciveMatrix(0, 0) = static_cast<T>(1) / (aspect * tanHalfFovY);
  perspeciveMatrix(1, 1) = static_cast<T>(1) / (tanHalfFovY);
  perspeciveMatrix(2, 2) = zFar / (zFar - zNear);  // not the same (depends on handness + NO / LO)
  if constexpr (Option == Options::RowMajor) {
    perspeciveMatrix(3, 2) = (zFar * zNear) / (zFar - zNear); // depends on NO / LO
    perspeciveMatrix(2, 3) = static_cast<T>(1);               // depends on handness (z, not -z)
  } else if (Option == Options::ColumnMajor) {
    perspeciveMatrix(2, 3) = (zFar * zNear) / (zFar - zNear); // depends on NO / LO
    perspeciveMatrix(3, 2) = static_cast<T>(1);               // depends on handness (z, not -z)
  }
  return perspeciveMatrix;
}

/**
 * Generates a left-handed perspective projection matrix with a depth range of negative one to one.
 * @note LH-NO - Left-Handed, Negative One to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhNo(T fovY, T aspect, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspeciveMatrix(0);
  perspeciveMatrix(0, 0) = static_cast<T>(1) / (aspect * tanHalfFovY);
  perspeciveMatrix(1, 1) = static_cast<T>(1) / (tanHalfFovY);
  perspeciveMatrix(2, 2) = (zFar + zNear)    / (zFar - zNear);  // not the same (depends on handness + NO / LO)
  if constexpr (Option == Options::RowMajor) {
    perspeciveMatrix(3, 2) = -(static_cast<T>(2) * zFar * zNear) / (zFar - zNear); // depends on NO / LO
    perspeciveMatrix(2, 3) = static_cast<T>(1);  // depends on handness (z, not -z)
  } else if (Option == Options::ColumnMajor) {
    perspeciveMatrix(2, 3) = -(static_cast<T>(2) * zFar * zNear) / (zFar - zNear); // depends on NO / LO
    perspeciveMatrix(3, 2) = static_cast<T>(1);  // depends on handness (z, not -z)
  }
  return perspeciveMatrix;
}

/**
 * Generates a right-handed perspective projection matrix based on field of
 * view, width, and height with a depth range of zero to one.
 * @note RH-ZO - Right-Handed, Zero to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhZo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveRhZo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

/**
 * Generates a right-handed perspective projection matrix based on field of
 * view, width, and height with a depth range of negative one to one.
 * @note RH-NO - Right-Handed, Negative One to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhNo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveRhNo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

/**
 * Generates a left-handed perspective projection matrix based on field of view,
 * width, and height with a depth range of zero to one.
 * @note LH-ZO - Left-Handed, Zero to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhZo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveLhZo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

/**
 * Generates a left-handed perspective projection matrix based on field of view,
 * width, and height with a depth range of negative one to one.
 * @note LH-NO - Left-Handed, Negative One to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhNo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveLhNo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

// END: perspective projection creation matrix
// ----------------------------------------------------------------------------

// BEGIN: frustrum (perspective projection matrix that off center) creation functions
// ----------------------------------------------------------------------------

/**
 * Generates a right-handed frustum projection matrix with a depth range of zero
 * to one.
 * @note RH-ZO - Right-Handed, Zero to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_frustumRhZo(T left, T right, T bottom, T top, T nearVal, T farVal)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> frustrum(0);
  frustrum(0, 0) = (static_cast<T>(2) * nearVal) / (right - left);
  frustrum(1, 1) = (static_cast<T>(2) * nearVal) / (top - bottom);
  frustrum(2, 2) = farVal / (nearVal - farVal);        // depends on NO / ZO

  if constexpr (Option == Options::RowMajor) {
    frustrum(2, 0) = (right + left) / (right - left);  // depends on handness
    frustrum(2, 1) = (top + bottom) / (top - bottom);  // depends on handness
    frustrum(3, 2) = -(farVal * nearVal) / (farVal - nearVal); // depends on NO / ZO
    frustrum(2, 3) = -static_cast<T>(1);               // depends on handness
  } else if (Option == Options::ColumnMajor) {
    frustrum(0, 2) = (right + left) / (right - left);  // depends on handness
    frustrum(1, 2) = (top + bottom) / (top - bottom);  // depends on handness
    frustrum(2, 3) = -(farVal * nearVal) / (farVal - nearVal); // depends on NO / ZO
    frustrum(3, 2) = -static_cast<T>(1);               // depends on handness
  }
  return frustrum;
}

/**
 * Generates a right-handed frustum projection matrix with a depth range of
 * negative one to one.
 * @note RH-NO - Right-Handed, Negative One to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_frustumRhNo(T left, T right, T bottom, T top, T nearVal, T farVal)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> frustrum(0);
  frustrum(0, 0) = (static_cast<T>(2) * nearVal) / (right - left);
  frustrum(1, 1) = (static_cast<T>(2) * nearVal) / (top - bottom);
  frustrum(2, 2)
      = -(farVal + nearVal) / (farVal - nearVal);    // depends on NO / ZO

  if constexpr (Option == Options::RowMajor) {
    frustrum(2, 0) = (right + left) / (right - left);  // depends on handness
    frustrum(2, 1) = (top + bottom) / (top - bottom);  // depends on handness
    frustrum(3, 2) = -(static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal); // depends on NO / ZO
    frustrum(2, 3) = -static_cast<T>(1);               // depends on handness
  } else if (Option == Options::ColumnMajor) {
    frustrum(0, 2) = (right + left) / (right - left);  // depends on handness
    frustrum(1, 2) = (top + bottom) / (top - bottom);  // depends on handness
    frustrum(2, 3) = -(static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal); // depends on NO / ZO
    frustrum(3, 2) = -static_cast<T>(1);               // depends on handness
  }
  return frustrum;
}

/**
 * Generates a left-handed frustum projection matrix with a depth range of zero
 * to one.
 * @note LH-ZO - Left-Handed, Zero to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_frustumLhZo(T left, T right, T bottom, T top, T nearVal, T farVal)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> frustrum(0);
  frustrum(0, 0) = (static_cast<T>(2) * nearVal) / (right - left);
  frustrum(1, 1) = (static_cast<T>(2) * nearVal) / (top - bottom);
  frustrum(2, 2) = farVal / (farVal - nearVal);         // depends on NO / ZO

  if constexpr (Option == Options::RowMajor) {
    frustrum(2, 0) = -(right + left) / (right - left);  // depends on handness
    frustrum(2, 1) = -(top + bottom) / (top - bottom);  // depends on handness
    frustrum(3, 2) = -(farVal * nearVal) / (farVal - nearVal); // depends on NO / ZO
    frustrum(2, 3) = static_cast<T>(1);                 // depends on handness
  } else if (Option == Options::ColumnMajor) {
    frustrum(0, 2) = -(right + left) / (right - left);  // depends on handness
    frustrum(1, 2) = -(top + bottom) / (top - bottom);  // depends on handness
    frustrum(2, 3) = -(farVal * nearVal) / (farVal - nearVal); // depends on NO / ZO
    frustrum(3, 2) = static_cast<T>(1);                 // depends on handness
  }
  return frustrum;
}

/**
 * Generates a left-handed frustum projection matrix with a depth range of
 * negative one to one.
 * @note LH-NO - Left-Handed, Negative One to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_frustumLhNo(T left, T right, T bottom, T top, T nearVal, T farVal)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> frustrum(0);
  frustrum(0, 0) = (static_cast<T>(2) * nearVal) / (right - left);
  frustrum(1, 1) = (static_cast<T>(2) * nearVal) / (top - bottom);
  frustrum(2, 2) = (farVal + nearVal) / (farVal - nearVal);  // depends on NO / ZO

  if constexpr (Option == Options::RowMajor) {
    frustrum(2, 0) = -(right + left) / (right - left);  // depends on handness
    frustrum(2, 1) = -(top + bottom) / (top - bottom);  // depends on handness
    frustrum(3, 2) = -(static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal); // depends on NO / ZO
    frustrum(2, 3) = static_cast<T>(1);                 // depends on handness
  } else if (Option == Options::ColumnMajor) {
    frustrum(0, 2) = -(right + left) / (right - left);  // depends on handness
    frustrum(1, 2) = -(top + bottom) / (top - bottom);  // depends on handness
    frustrum(2, 3) = -(static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal); // depends on NO / ZO
    frustrum(3, 2) = static_cast<T>(1);                 // depends on handness
  }

  return frustrum;
}

// END: frustrum (perspective projection matrix that off center) creation functions
// ----------------------------------------------------------------------------

// BEGIN: orthographic projection creation matrix
// ----------------------------------------------------------------------------

// TODO: add ortho functions (LH/RH) that takes (left, right, bottom, top) w/o near / far  

/**
 * Generates a left-handed orthographic projection matrix with a depth range of
 * zero to one.
 * @note LH-ZO - Left-Handed, Zero to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoLhZo(T left, T right, T bottom, T top, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto orthographicMat  = Matrix<T, 4, 4, Option>::Identity();
  orthographicMat(0, 0) = static_cast<T>(2) / (right - left);
  orthographicMat(1, 1) = static_cast<T>(2) / (top - bottom);
  orthographicMat(2, 2) = static_cast<T>(1) / (zFar - zNear);  // depends on handness
  if constexpr (Option == Options::RowMajor) {
    orthographicMat(3, 0) = -(right + left) / (right - left);
    orthographicMat(3, 1) = -(top + bottom) / (top - bottom);
    orthographicMat(3, 2) = -zNear / (zFar - zNear);  // depends on ZO / NO
  } else if (Option == Options::ColumnMajor) {
    orthographicMat(0, 3) = -(right + left) / (right - left);
    orthographicMat(1, 3) = -(top + bottom) / (top - bottom);
    orthographicMat(2, 3) = -zNear / (zFar - zNear);  // depends on ZO / NO
  }
  return orthographicMat;
}

/**
 * Generates a left-handed orthographic projection matrix with a depth range of
 * negative one to one.
 * @note LH-NO - Left-Handed, Negative One to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoLhNo(T left, T right, T bottom, T top, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto orthographicMat  = Matrix<T, 4, 4, Option>::Identity();
  orthographicMat(0, 0) = static_cast<T>(2) / (right - left);
  orthographicMat(1, 1) = static_cast<T>(2) / (top - bottom);
  orthographicMat(2, 2) = static_cast<T>(2) / (zFar - zNear);  // depends on handness
  if constexpr (Option == Options::RowMajor) {
    orthographicMat(3, 0) = -(right + left) / (right - left);
    orthographicMat(3, 1) = -(top + bottom) / (top - bottom);
    orthographicMat(3, 2) = -(zFar + zNear) / (zFar - zNear);  // depends on ZO / NO
  } else if (Option == Options::ColumnMajor) {
    orthographicMat(0, 3) = -(right + left) / (right - left);
    orthographicMat(1, 3) = -(top + bottom) / (top - bottom);
    orthographicMat(2, 3) = -(zFar + zNear) / (zFar - zNear);  // depends on ZO / NO
  }
  return orthographicMat;
}

/**
 * Generates a right-handed orthographic projection matrix with a depth range of
 * zero to one.
 * @note RH-ZO - Right-Handed, Zero to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoRhZo(T left, T right, T bottom, T top, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto orthographicMat  = Matrix<T, 4, 4, Option>::Identity();
  orthographicMat(0, 0) = static_cast<T>(2) / (right - left);
  orthographicMat(1, 1) = static_cast<T>(2) / (top - bottom);
  orthographicMat(2, 2) = -static_cast<T>(1) / (zFar - zNear);  // depends on handness
  if constexpr (Option == Options::RowMajor) {
    orthographicMat(3, 0) = -(right + left) / (right - left);
    orthographicMat(3, 1) = -(top + bottom) / (top - bottom);
    orthographicMat(3, 2) = -zNear / (zFar - zNear);  // depends on ZO / NO
  } else if (Option == Options::ColumnMajor) {
    orthographicMat(0, 3) = -(right + left) / (right - left);
    orthographicMat(1, 3) = -(top + bottom) / (top - bottom);
    orthographicMat(2, 3) = -zNear / (zFar - zNear);  // depends on ZO / NO
  }
  return orthographicMat;
}

/**
 * Generates a right-handed orthographic projection matrix with a depth range of
 * negative one to one.
 * @note RH-NO - Right-Handed, Negative One to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoRhNo(T left, T right, T bottom, T top, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto orthographicMat  = Matrix<T, 4, 4, Option>::Identity();
  orthographicMat(0, 0) = static_cast<T>(2) / (right - left);
  orthographicMat(1, 1) = static_cast<T>(2) / (top - bottom);
  orthographicMat(2, 2) = -static_cast<T>(2) / (zFar - zNear);  // depends on handness
  if constexpr (Option == Options::RowMajor) {
    orthographicMat(3, 0) = -(right + left) / (right - left);
    orthographicMat(3, 1) = -(top + bottom) / (top - bottom);
    orthographicMat(3, 2) = -(zFar + zNear) / (zFar - zNear);  // depends on ZO / NO
  } else if (Option == Options::ColumnMajor) {
    orthographicMat(0, 3) = -(right + left) / (right - left);
    orthographicMat(1, 3) = -(top + bottom) / (top - bottom);
    orthographicMat(2, 3) = -(zFar + zNear) / (zFar - zNear);  // depends on ZO / NO
  }
  return orthographicMat;
}

// END: orthographic projection creation matrix
// ----------------------------------------------------------------------------

// clang-format on

// BEGIN: global util vector objects
// ----------------------------------------------------------------------------

// TODO: consider moving directional vector variables to a separate file

// TODO: make these constexpr

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_upVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.y() = 1;
  return vec;
}

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_downVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.y() = -1;
  return vec;
}

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_rightVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.x() = 1;
  return vec;
}

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_leftVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.x() = -1;
  return vec;
}

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 3>
auto g_forwardVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.z() = 1;
  return vec;
}

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 3>
auto g_backwardVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.z() = -1;
  return vec;
}

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
auto g_zeroVector() -> const Vector<T, Size, Option> {
  return Vector<T, Size, Option>(0);
}

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
auto g_oneVector() -> const Vector<T, Size, Option> {
  return Vector<T, Size, Option>(1);
}

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
auto g_unitVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(1);
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  vec.normalize();
  return vec;
#else
  return vec.normalize();
#endif  // MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
}

// END: global util vector objects
// ----------------------------------------------------------------------------

}  // namespace math

#endif