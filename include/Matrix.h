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
#include <cmath>

constexpr unsigned int STACK_ALLOCATION_LIMIT = 16; // 4 by 4 matrix

namespace math
{

	template<typename U, typename... Args>
	concept AllSameAs = (std::same_as<Args, U> && ...);

	template<auto Count, typename... Args>
	concept ArgsSizeGreaterThanCount = (sizeof...(Args) > Count);

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


		template<typename... Args> requires
			AllSameAs<T, Args...> &&
			ArgsSizeGreaterThanCount<1, Args...>
			Matrix(Args... args)
		{
			static_assert(sizeof...(Args) == Rows * Columns, "Incorrect number of arguments for Matrix initialization");
			if constexpr (UseHeap) {
				m_data_ = new T[Rows * Columns];
			}
			T arr[] = { args... };
			std::copy(std::begin(arr), std::end(arr), m_data_);
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

		template <std::ranges::range Range>
		Matrix(const Range& range)
		{
			assert(std::ranges::size(range) <= Rows * Columns);
			if constexpr (UseHeap) {
				m_data_ = new T[Rows * Columns];
			}
			std::copy_n(range.begin(), Rows * Columns, m_data_);
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

		static constexpr unsigned int getRows()
		{
			return Rows;
		}

		static constexpr unsigned int getColumns()
		{
			return Columns;
		}

		static constexpr Options getOption()
		{
			return Option;
		}

		T* data()
		{
			return m_data_;
		}

		const T* data() const
		{
			return m_data_;
		}

		Matrix<T, Columns, Rows, Option> transpose() const
		{
			Matrix<T, Columns, Rows, Option> res;
			for (unsigned int i = 0; i < Rows; ++i) {
				for (unsigned int j = 0; j < Columns; ++j) {
					res(j, i) = (*this)(i, j);
				}
			}
			return res;
		}

		T determinant() const
		{
			static_assert(Rows == Columns, "Determinant is only defined for square matrices");
			assert(Rows == Columns);

			if constexpr (Rows == 1) {
				return m_data_[0];
			}
			else if constexpr (Rows == 2) {
				const T& a = operator()(0, 0);
				const T& b = operator()(0, 1);
				const T& c = operator()(1, 0);
				const T& d = operator()(1, 1);
				return a * d - b * c;
			}
			else {
				T det = 0;
				int sign = 1;
				for (unsigned int i = 0; i < Rows; ++i) {
					// Construct a sub-matrix
					Matrix<T, Rows - 1, Columns - 1, Option> submatrix;
					for (unsigned int j = 1; j < Rows; ++j) {
						unsigned int k = 0;
						for (unsigned int l = 0; l < Columns; ++l) {
							if (l != i) {
								submatrix(j - 1, k) = (*this)(j, l);
								++k;
							}
						}
					}
					// Recursive call
					det += sign * (*this)(0, i) * submatrix.determinant();
					sign = -sign;
				}
				return det;
			}
		}

		Matrix inverse() const
		{
			static_assert(Rows == Columns, "Inverse is only defined for square matrices");

			Matrix<T, Rows, 2 * Columns, Option> augmentedMatrix;
			// Fill augmentedMatrix
			for (unsigned int i = 0; i < Rows; ++i) {
				for (unsigned int j = 0; j < Columns; ++j) {
					augmentedMatrix(i, j) = (*this)(i, j);
				}
				for (unsigned int j = Columns; j < 2 * Columns; ++j) {
					if (i == j - Columns) {
						augmentedMatrix(i, j) = 1;
					}
					else {
						augmentedMatrix(i, j) = 0;
					}
				}
			}

			// Perform Gauss-Jordan elimination
			for (unsigned int i = 0; i < Rows; ++i) {
				// Search for maximum in this column
				T maxEl = std::abs(augmentedMatrix(i, i));
				unsigned int maxRow = i;
				for (unsigned int k = i + 1; k < Rows; ++k) {
					if (std::abs(augmentedMatrix(k, i)) > maxEl) {
						maxEl = augmentedMatrix(k, i);
						maxRow = k;
					}
				}

				// Swap maximum row with current row
				for (unsigned int k = i; k < 2 * Columns; ++k) {
					T tmp = augmentedMatrix(maxRow, k);
					augmentedMatrix(maxRow, k) = augmentedMatrix(i, k);
					augmentedMatrix(i, k) = tmp;
				}

				// Make all rows below this one 0 in current column
				for (unsigned int k = i + 1; k < Rows; ++k) {
					T c = -augmentedMatrix(k, i) / augmentedMatrix(i, i);
					for (unsigned int j = i; j < 2 * Columns; ++j) {
						if (i == j) {
							augmentedMatrix(k, j) = 0;
						}
						else {
							augmentedMatrix(k, j) += c * augmentedMatrix(i, j);
						}
					}
				}
			}

			// Make all rows above this one zero in current column
			for (int i = Rows - 1; i >= 0; i--) {
				for (int k = i - 1; k >= 0; k--) {
					T c = -augmentedMatrix(k, i) / augmentedMatrix(i, i);
					for (unsigned int j = i; j < 2 * Columns; ++j) {
						if (i == j) {
							augmentedMatrix(k, j) = 0;
						}
						else {
							augmentedMatrix(k, j) += c * augmentedMatrix(i, j);
						}
					}
				}
			}

			// Normalize diagonal
			for (unsigned int i = 0; i < Rows; ++i) {
				T c = 1.0 / augmentedMatrix(i, i);
				for (unsigned int j = i; j < 2 * Columns; ++j) {
					augmentedMatrix(i, j) *= c;
				}
			}

			// Copy the right half of the augmented matrix to the result
			Matrix<T, Rows, Columns, Option> inverseMatrix;
			for (unsigned int i = 0; i < Rows; ++i) {
				for (unsigned int j = 0; j < Columns; ++j) {
					inverseMatrix(i, j) = augmentedMatrix(i, j + Columns);
				}
			}

			return inverseMatrix;
		}

		int rank() const
		{
			// Create a copy of the matrix
			Matrix<T, Rows, Columns, Option> copy(*this);

			// Apply Gaussian elimination
			int rank = 0;
			for (int row = 0; row < Rows; ++row)
			{
				// Find the maximum element in this column
				T maxEl = std::abs(copy(row, rank));
				int maxRow = row;
				for (int i = row + 1; i < Rows; ++i)
				{
					if (std::abs(copy(i, rank)) > maxEl)
					{
						maxEl = std::abs(copy(i, rank));
						maxRow = i;
					}
				}

				// Swap maximum row with current row
				if (maxEl != 0)
				{
					for (int i = 0; i < Columns; ++i)
					{
						T tmp = copy(maxRow, i);
						copy(maxRow, i) = copy(row, i);
						copy(row, i) = tmp;
					}

					// Make all rows below this one 0 in current column
					for (int i = row + 1; i < Rows; ++i)
					{
						T c = -copy(i, rank) / copy(row, rank);
						for (int j = rank; j < Columns; ++j)
						{
							if (rank == j)
							{
								copy(i, j) = 0;
							}
							else
							{
								copy(i, j) += c * copy(row, j);
							}
						}
					}

					++rank;
				}

				// If rank is equal to Columns, no need to continue
				if (rank == Columns)
				{
					break;
				}
			}

			return rank;
		}
		}


		T trace() const
		{
			static_assert(Rows == Columns, "Trace is only defined for square matrices");
			T sum = 0;
			for (unsigned int i = 0; i < Rows; ++i) {
				sum += this->operator()(i, i);
			}
			return sum;
		}

		template<unsigned int NewRows, unsigned int NewColumns>
		Matrix<T, NewRows, NewColumns, Option> reshape() const
		{
			static_assert(Rows * Columns == NewRows * NewColumns, "New dimensions must have the same total size as the original matrix");
			Matrix<T, NewRows, NewColumns, Option> newMatrix;
			std::copy_n(m_data_, Rows * Columns, newMatrix.data());
			return newMatrix;
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

		template<unsigned int ResultColumns>
		Matrix<T, Rows, ResultColumns, Option> operator*(const Matrix<T, Columns, ResultColumns, Option>& other) const
		{
			Matrix<T, Rows, ResultColumns, Option> result;
			auto mulFunc = InstructionSet<T>::template getMulFunc<Option>();
			mulFunc(result.data(), m_data_, other.data(), Rows, ResultColumns, Columns);
			return result;
		}

		/**
		* @brief Matrix multiplication-assignment operator
		*
		* This operator multiplies the current matrix with the given one.
		* Note: This function only works when the matrices have the same dimensions and squared.
		*
		*/
		Matrix& operator*=(const Matrix& other)
		{
			static_assert(Columns == Rows, "For Matrix multiplication (*=), matrix dimensions must be squared");
			assert(this != &other && "Cannot perform operation on the same matrix instance");
			*this = *this * other;
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

		friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
		{
			for (unsigned int i = 0; i < Rows; ++i) {
				for (unsigned int j = 0; j < Columns; ++j) {
					os << matrix(i, j) << ' ';
				}
				os << '\n';
			}
			return os;
		}

		class MatrixInitializer
		{
			Matrix& m_matrix_;
			unsigned int m_row_;
			unsigned int m_column_;

		public:
			MatrixInitializer(Matrix& matrix, unsigned int row, unsigned int column)
				: m_matrix_(matrix), m_row_(row), m_column_(column)
			{}

			MatrixInitializer& operator,(const T& value)
			{
				if (m_column_ >= m_matrix_.getColumns()) {
					++m_row_;
					m_column_ = 0;
				}

				if (m_row_ < m_matrix_.getRows()) {
					m_matrix_(m_row_, m_column_) = value;
					++m_column_;
				}

				return *this;
			}
		};

		MatrixInitializer operator<<(const T& value)
		{
			this->operator()(0, 0) = value;
			return MatrixInitializer(*this, 0, 1);
		}


	private:

		using DataType = typename std::conditional<UseHeap, T*, T[Rows * Columns]>::type;
		DataType m_data_;
	};
}
