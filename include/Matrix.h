#pragma once

#include <type_traits>
#include <cassert>
#include "Options.h"

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

		~Matrix()
		{
			if constexpr (UseHeap) 
			{
				delete[] m_data_;
			}
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

	private:
		static const bool UseHeap = sizeof(T) * Rows * Columns > STACK_ALLOCATION_LIMIT;
		using DataType = typename std::conditional<UseHeap, T*, T[Rows * Columns]>::type;
		DataType m_data_;
	};
}
