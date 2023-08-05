/**
 * @file Matrix.h
 */


#pragma once

#include "../src/lib/options/Options.h"
#include "../src/lib/simd/instruction_set/InstructionSet.h"

#include <iterator>
#include <cassert>
#include <algorithm>
#include <type_traits>
#include <ranges>

constexpr unsigned int STACK_ALLOCATION_LIMIT = 16; // 4 by 4 matrix

namespace math
{
	template<typename T, unsigned int Rows, unsigned int Columns, Options Option = Options::ROW_MAJOR>
	class Matrix
	{
	public:

		static const bool UseHeap = Rows * Columns > STACK_ALLOCATION_LIMIT;

		Matrix()
		{
			if constexpr (UseHeap) {
				m_data_ = new T[Rows * Columns];
			}
		}

		Matrix(const T& element)
		{
			if constexpr (UseHeap) {
				m_data_ = new T[Rows * Columns];
			}
			std::fill_n(m_data_, Rows * Columns, element);
		}

		Matrix(const Matrix& other)
		{
			if constexpr (UseHeap) {
				m_data_ = new T[Rows * Columns];
			}

			std::copy_n(other.m_data_, Rows * Columns, m_data_);
		}

		Matrix& operator=(const Matrix& other)
		{
			if (this != &other) {
				if constexpr (UseHeap) {
					delete[] m_data_;
					m_data_ = new T[Rows * Columns];
				}
				std::copy_n(other.m_data_, Rows * Columns, m_data_);
			}
			return *this;
		}

		Matrix(Matrix&& other) noexcept
		{
			if constexpr (UseHeap) {
				m_data_ = other.m_data_;
				other.m_data_ = nullptr;
			}
			else {
				std::move(std::begin(other.m_data_), std::end(other.m_data_), std::begin(m_data_));
			}
		}

		Matrix& operator=(Matrix&& other) noexcept
		{
			if (this != &other) {
				if constexpr (UseHeap) {
					delete[] m_data_;
					m_data_ = other.m_data_;
					other.m_data_ = nullptr;
				}
				else {
					std::move(std::begin(other.m_data_), std::end(other.m_data_), std::begin(m_data_));
				}
			}
			return *this;
		}

		template <std::input_iterator InputIt>
		Matrix(InputIt first, InputIt last)
		{
			assert(std::distance(first, last) == Rows * Columns);
			if constexpr (UseHeap) {
				m_data_ = new T[Rows * Columns];
			}
			std::copy(first, last, m_data_);
		}
		static constexpr Matrix Identity()
		{
			Matrix m(0);
			constexpr unsigned int kMin = std::min(Rows, Columns);
			for (unsigned int i = 0; i < kMin; ++i) {
				m(i, i) = 1;
			}
			return m;
		}

		~Matrix()
		{
			if constexpr (UseHeap) {
				delete[] m_data_;
			}
		}

		T& operator()(unsigned int row, unsigned int col)
		{
			if constexpr (Option == Options::ROW_MAJOR) {
				return m_data_[row * Columns + col];
			}
			else {
				return m_data_[col * Rows + row];
			}
		}

		const T& operator()(unsigned int row, unsigned int col) const
		{
			if constexpr (Option == Options::ROW_MAJOR) {
				return m_data_[row * Columns + col];
			}
			else {
				return m_data_[col * Rows + row];
			}
		}

		const T& coeff(unsigned int row, unsigned int col) const
		{
			assert(row < Rows && col < Columns && "Index out of bounds");
			return operator()(row, col);
		}

		T& coeffRef(unsigned int row, unsigned int col)
		{
			assert(row < Rows && col < Columns && "Index out of bounds");
			return operator()(row, col);
		}

		constexpr unsigned int getRows() const
		{
			return Rows;
		}

		constexpr unsigned int getColumns() const
		{
			return Columns;
		}
		T determinant() const
		{
			static_assert(Rows == Columns, "Determinant is only defined for square matrices");
			//TODO: implement
			return;
		}

		Matrix inverse() const
		{
			static_assert(Rows == Columns, "Inverse is only defined for square matrices");
			//TODO: implement
			return;
		}
		Matrix operator+(const Matrix& other) const
		{
			Matrix result = *this;
			auto addFunc = InstructionSet<T>::getAddFunc();
			addFunc(result.m_data_, other.m_data_, Rows * Columns);
			return result;
		}

		Matrix& operator+=(const Matrix& other)
		{
			auto addFunc = InstructionSet<T>::getAddFunc();
			addFunc(m_data_, other.m_data_, Rows * Columns);
			return *this;
		}

		Matrix operator+(const T& scalar) const
		{
			Matrix result = *this;
			auto addScalarFunc = InstructionSet<T>::getAddScalarFunc();
			addScalarFunc(result.m_data_, scalar, Rows * Columns);
			return result;
		}

		Matrix& operator+=(const T& scalar)
		{
			auto addScalarFunc = InstructionSet<T>::getAddScalarFunc();
			addScalarFunc(m_data_, scalar, Rows * Columns);
			return *this;
		}

		Matrix operator-(const Matrix& other) const
		{
			Matrix result = *this;
			auto subFunc = InstructionSet<T>::getSubFunc();
			subFunc(result.m_data_, other.m_data_, Rows * Columns);
			return result;
		}

		Matrix& operator-=(const Matrix& other)
		{
			auto subFunc = InstructionSet<T>::getSubFunc();
			subFunc(m_data_, other.m_data_, Rows * Columns);
			return *this;
		}

		Matrix operator-(const T& scalar) const
		{
			Matrix result = *this;
			auto subScalarFunc = InstructionSet<T>::getSubScalarFunc();
			subScalarFunc(result.m_data_, scalar, Rows * Columns);
			return result;
		}

		Matrix& operator-=(const T& scalar)
		{
			auto subScalarFunc = InstructionSet<T>::getSubScalarFunc();
			subScalarFunc(m_data_, scalar, Rows * Columns);
			return *this;
		}

		Matrix operator*(const Matrix& other) const
		{
			//static_assert(Columns == other.Rows, "Matrix dimensions must agree for multiplication");
			Matrix result;
			auto mulFunc = InstructionSet<T>::getMulFunc<Option>();
			mulFunc(result.m_data_, m_data_, other.m_data_, Rows);
			return result;
		}

		Matrix& operator*=(const Matrix& other)
		{
			//static_assert(Columns == other.Rows, "Matrix dimensions must agree for multiplication");
			Matrix temp = *this * other;
			*this = temp;
			return *this;
		}


		Matrix operator*(const T& scalar) const
		{
			Matrix result = *this;
			auto mulScalarFunc = InstructionSet<T>::getMulScalarFunc();
			mulScalarFunc(result.m_data_, scalar, Rows * Columns);
			return result;
		}

		Matrix& operator*=(const T& scalar)
		{
			auto mulScalarFunc = InstructionSet<T>::getMulScalarFunc();
			mulScalarFunc(m_data_, scalar, Rows * Columns);
			return *this;
		}

		Matrix operator/(const T& scalar) const
		{
			Matrix result = *this;
			auto divScalarFunc = InstructionSet<T>::getDivScalarFunc();
			divScalarFunc(result.m_data_, scalar, Rows * Columns);
			return result;
		}

		Matrix& operator/=(const T& scalar)
		{
			auto divScalarFunc = InstructionSet<T>::getDivScalarFunc();
			divScalarFunc(m_data_, scalar, Rows * Columns);
			return *this;
		}


	private:
		
		using DataType = typename std::conditional<UseHeap, T*, T[Rows * Columns]>::type;
		DataType m_data_;
	};
}
