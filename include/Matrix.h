/**
 * @file Matrix.h
 */


#pragma once

#include <type_traits>
#include <cassert>
#include "Options.h"
#include "InstructionSet.h"

constexpr unsigned int STACK_ALLOCATION_LIMIT = 16;

namespace math
{
	template<typename T, unsigned int Rows, unsigned int Columns, Options Option = Options::ROW_MAJOR>
	class Matrix
	{
	public:

		static const bool UseHeap = sizeof(T) * Rows * Columns > STACK_ALLOCATION_LIMIT;

		Matrix()
		{
			if constexpr (UseHeap) 
			{
				m_data_ = new T[Rows * Columns];
			}
		}

		Matrix(const T& element)
		{
			if constexpr (UseHeap)
			{
				m_data_ = new T[Rows * Columns];
			}
			std::fill_n(m_data_, Rows * Columns, element);
		}

		Matrix(const Matrix& other)
		{
			if constexpr (UseHeap)
			{
				m_data_ = new T[Rows * Columns];
			}

			std::copy_n(other.m_data_, Rows * Columns, m_data_);
		}


		~Matrix()
		{
			if constexpr (UseHeap) 
			{
				delete[] m_data_;
			}
		}

		Matrix& operator=(const Matrix& other)
		{
			if (this != &other)  
			{
				if constexpr (UseHeap)
				{
					delete[] m_data_;
					m_data_ = new T[Rows * Columns];
				}
				std::copy_n(other.m_data_, Rows * Columns, m_data_);
			}
			return *this;
		}

		T& operator()(unsigned int row, unsigned int col)
		{
			if constexpr (Option == Options::ROW_MAJOR) 
			{
				return m_data_[row * Columns + col];
			}
			else 
			{
				return m_data_[col * Rows + row];
			}
		}

		const T& operator()(unsigned int row, unsigned int col) const
		{
			if constexpr (Option == Options::ROW_MAJOR) 
			{
				return m_data_[row * Columns + col];
			}
			else 
			{
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

		Matrix operator*(const Matrix& other) const
		{
			Matrix result;
			
			return result;
		}

		Matrix& operator*=(const Matrix& other)
		{
			static_assert(Columns == Rows, "Matrix dimensions must agree for multiplication");
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



	private:
		
		using DataType = typename std::conditional<UseHeap, T*, T[Rows * Columns]>::type;
		DataType m_data_;
	};
}
