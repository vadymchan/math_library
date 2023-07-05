#pragma once

#include "Options.h"

namespace math
{
	template<typename T, unsigned int Rows, unsigned int Columns, Options Option>
	class Matrix
	{
	public:
		Matrix() : m_data_(new T[Rows * Columns]) {}
		
		~Matrix()
		{
			delete[] m_data_;
		}

		

	private:
		T* m_data_;
	};

}