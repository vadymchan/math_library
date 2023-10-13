/**
 * @file Graphics.h
 */

#pragma once

#include "Matrix.h"

namespace math
{

	//lookAtRH(eyePosition, centerPosition, upDirection)
	//lookAtLH(eyePosition, centerPosition, upDirection)

    template<typename T, Options Option = Options::RowMajor>
    Matrix<T, 4, 4> lookAtRH(const Matrix<T, 3, 1>& eye, const Matrix<T, 3, 1>& target, const Matrix<T, 3, 1>& worldUp)
    {
        Matrix<T, 3, 1> forward = (eye - target).normalize();
        Matrix<T, 3, 1> right = worldUp.cross(forward).normalize();
        Matrix<T, 3, 1> up = forward.cross(right);

        Matrix<T, 4, 4> viewMatrix = Matrix<T, 4, 4>::Identity();

        if constexpr (Option == Options::RowMajor)
        {
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
        }
        else if constexpr (Option == Options::ColumnMajor)
        {
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


    template<typename T, Options Option = Options::RowMajor>
    Matrix<T, 4, 4> lookAtLH(const Matrix<T, 3, 1>& eye, const Matrix<T, 3, 1>& target, const Matrix<T, 3, 1>& worldUp)
    {
        Matrix<T, 3, 1> forward = (target - eye).normalize();
        Matrix<T, 3, 1> right = worldUp.cross(forward).normalize();
        Matrix<T, 3, 1> up = forward.cross(right);

        Matrix<T, 4, 4> viewMatrix = Matrix<T, 4, 4>::Identity();

        if constexpr (Option == Options::RowMajor)
        {
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
        }
        else // COLUMN_MAJOR
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



	//lookToRH(eyePosition, centerPosition, upDirection)
	//lookToLH(eyePosition, centerPosition, upDirection)
    
    template<typename T, Options Option>
    Matrix<T, 4, 4> lookToRH(const Matrix<T, 3, 1>& eye, const Matrix<T, 3, 1>& direction, const Matrix<T, 3, 1>& worldUp)
    {
        Matrix<T, 3, 1> forward = direction.normalize();
        Matrix<T, 3, 1> right = worldUp.cross(forward).normalize();
        Matrix<T, 3, 1> up = forward.cross(right);

        Matrix<T, 4, 4> viewMatrix = Matrix<T, 4, 4>::Identity();

        if constexpr (Option == Options::RowMajor)
        {
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
        }
        else if constexpr (Option == Options::ColumnMajor)
        {
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


    template<typename T, Options Option = Options::RowMajor>
    Matrix<T, 4, 4> lookToLH(const Matrix<T, 3, 1>& eye, const Matrix<T, 3, 1>& direction, const Matrix<T, 3, 1>& worldUp)
    {
        Matrix<T, 3, 1> forward = direction.normalize();
        Matrix<T, 3, 1> right = worldUp.cross(forward).normalize();
        Matrix<T, 3, 1> up = forward.cross(right);

        Matrix<T, 4, 4> viewMatrix = Matrix<T, 4, 4>::Identity();

        if constexpr (Option == Options::RowMajor)
        {
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
        }
        else if constexpr (Option == Options::ColumnMajor)
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
            viewMatrix(3, 2) = forward.dot(eye);
        }

        return viewMatrix;
    }


	//perspectiveFovRH(fov, aspectRatio, nearPlane, farPlane)
	//perspectiveFovLH(fov, aspectRatio, nearPlane, farPlane)
    template<typename T, Options Option = Options::RowMajor>
    Matrix<T, 4, 4> perspectiveFovRH(T fov, T aspectRatio, T nearPlane, T farPlane)
    {
        assert(abs(aspectRatio - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

        T tanHalfFovy = tan(fov / static_cast<T>(2));

        Matrix<T, 4, 4> result(0);
        result(0, 0) = static_cast<T>(1) / (aspectRatio * tanHalfFovy);
        result(1, 1) = static_cast<T>(1) / (tanHalfFovy);
        result(2, 2) = farPlane / (nearPlane - farPlane);

        if constexpr (Option == Options::RowMajor)
        {
            result(2, 3) = -1;
            result(3, 2) = -(farPlane * nearPlane) / (farPlane - nearPlane);
        }
        else if constexpr (Option == Options::ColumnMajor)
        {
            result(2, 3) = -(farPlane * nearPlane) / (farPlane - nearPlane);
            result(3, 2) = -1;
        }

        return result;
    }


    template<typename T, Options Option = Options::RowMajor>
    Matrix<T, 4, 4> perspectiveFovLH(T fov, T aspectRatio, T nearPlane, T farPlane)
    {
        assert(abs(aspectRatio - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

        T tanHalfFovy = tan(fov / static_cast<T>(2));

        Matrix<T, 4, 4> result(0);
        result(0, 0) = static_cast<T>(1) / (aspectRatio * tanHalfFovy);
        result(1, 1) = static_cast<T>(1) / (tanHalfFovy);
        result(2, 2) = farPlane / (farPlane - nearPlane);

        if constexpr (Option == Options::RowMajor)
        {
            result(2, 3) = 1;
            result(3, 2) = -(farPlane * nearPlane) / (farPlane - nearPlane);
        }
        else if constexpr (Option == Options::ColumnMajor)
        {
            result(3, 2) = 1;
            result(2, 3) = -(farPlane * nearPlane) / (farPlane - nearPlane);
        }

        return result;
    }



    //perspectiveOffCenter(left, right, bottom, top, nearPlane, farPlane)
	
    template<typename T, Options Option = Options::RowMajor>
    Matrix<T, 4, 4> perspectiveOffCenter(T left, T right, T bottom, T top, T nearPlane, T farPlane)
    {
        assert(abs(right - left) > std::numeric_limits<T>::epsilon() && "Right and left values cannot be equal");
        assert(abs(top - bottom) > std::numeric_limits<T>::epsilon() && "Top and bottom values cannot be equal");
        assert(abs(farPlane - nearPlane) > std::numeric_limits<T>::epsilon() && "Far and near plane values cannot be equal");

        Matrix<T, 4, 4> result(0);
        result(0, 0) = (static_cast<T>(2) * nearPlane) / (right - left);
        result(1, 1) = (static_cast<T>(2) * nearPlane) / (top - bottom);
        result(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);

        if constexpr (Option == Options::RowMajor)
        {
            result(2, 0) = (right + left) / (right - left);
            result(2, 1) = (top + bottom) / (top - bottom);
            result(2, 3) = -1;
            result(3, 2) = -(static_cast<T>(2) * farPlane * nearPlane) / (farPlane - nearPlane);
        }
        else if constexpr (Option == Options::ColumnMajor)
        {
            result(0, 2) = (right + left) / (right - left);
            result(1, 2) = (top + bottom) / (top - bottom);
            result(3, 2) = -1;
            result(2, 3) = -(static_cast<T>(2) * farPlane * nearPlane) / (farPlane - nearPlane);
        }

        return result;
    }



    //orthoRH(left, right, bottom, top, nearPlane, farPlane)
	//orthoLH(left, right, bottom, top, nearPlane, farPlane)

    template<typename T, Options Option = Options::RowMajor>
    Matrix<T, 4, 4> orthoRH(T left, T right, T bottom, T top, T nearPlane, T farPlane)
    {
        assert(abs(right - left) > std::numeric_limits<T>::epsilon() && "Right and left values cannot be equal");
        assert(abs(top - bottom) > std::numeric_limits<T>::epsilon() && "Top and bottom values cannot be equal");
        assert(abs(farPlane - nearPlane) > std::numeric_limits<T>::epsilon() && "Far and near plane values cannot be equal");

        Matrix<T, 4, 4> result = Matrix<T, 4, 4>::Identity();
        result(0, 0) = static_cast<T>(2) / (right - left);
        result(1, 1) = static_cast<T>(2) / (top - bottom);
        result(2, 2) = -static_cast<T>(2) / (farPlane - nearPlane);

        if constexpr (Option == Options::RowMajor)
        {
            result(0, 3) = -(right + left) / (right - left);
            result(1, 3) = -(top + bottom) / (top - bottom);
            result(2, 3) = -(farPlane + nearPlane) / (farPlane - nearPlane);
        }
        else if constexpr (Option == Options::ColumnMajor)
        {
            result(3, 0) = -(right + left) / (right - left);
            result(3, 1) = -(top + bottom) / (top - bottom);
            result(3, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
        }

        return result;
    }

    template<typename T, Options Option = Options::RowMajor>
    Matrix<T, 4, 4> orthoLH(T left, T right, T bottom, T top, T nearPlane, T farPlane)
    {
        assert(abs(right - left) > std::numeric_limits<T>::epsilon() && "Right and left values cannot be equal");
        assert(abs(top - bottom) > std::numeric_limits<T>::epsilon() && "Top and bottom values cannot be equal");
        assert(abs(farPlane - nearPlane) > std::numeric_limits<T>::epsilon() && "Far and near plane values cannot be equal");

        Matrix<T, 4, 4> result = Matrix<T, 4, 4>::Identity();
        result(0, 0) = static_cast<T>(2) / (right - left);
        result(1, 1) = static_cast<T>(2) / (top - bottom);
        result(2, 2) = static_cast<T>(2) / (farPlane - nearPlane);

        if constexpr (Option == Options::RowMajor)
        {
            result(0, 3) = -(right + left) / (right - left);
            result(1, 3) = -(top + bottom) / (top - bottom);
            result(2, 3) = -(farPlane + nearPlane) / (farPlane - nearPlane);
        }
        else if constexpr (Option == Options::ColumnMajor)
        {
            result(3, 0) = -(right + left) / (right - left);
            result(3, 1) = -(top + bottom) / (top - bottom);
            result(3, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
        }

        return result;
    }



} // namespace math