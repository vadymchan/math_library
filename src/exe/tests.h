/**
 * @file tests.h
 */

#ifndef MATH_LIBRARY_TESTS_H
#define MATH_LIBRARY_TESTS_H

#include <math_library/all.h>
#include <gtest/gtest.h>

// TEST(MatrixTest, ConstructorDestructorFloat)
//  TEST(MatrixTest, CopyConstructorFloat)
// TEST(MatrixTest, MoveConstructorFloat)
// TEST(MatrixTest, TransposeFloat)
// TEST(MatrixTest, ReshapeFloat)
// TEST(MatrixTest, ElementAccessFloat)
// TEST(MatrixTest, ConstructorDestructorFailureFloat)
// TEST(MatrixTest, ElementAccessFailureFloat)
// TEST(MatrixTest, ElementModificationFloat)
// TEST(MatrixTest, ElementModificationFailureFloat)
// TEST(MatrixTest, OutOfBoundsAccessFloat)
// TEST(MatrixTest, OperatorAccessFloat)
// TEST(MatrixTest, OperatorAccessConstFloat)
// TEST(MatrixTest, CoeffAccessConstFloat)
// TEST(MatrixTest, StackAllocationFloat)
// TEST(MatrixTest, HeapAllocationFloat)
// TEST(MatrixTest, AdditionFloat)
// TEST(MatrixTest, AdditionFailureFloat)
// TEST(MatrixTest, ScalarAdditionFloat)
// TEST(MatrixTest, ScalarAdditionFailureFloat)
// TEST(MatrixTest, SubtractionFloat)
// TEST(MatrixTest, ScalarSubtractionFloat)
// TEST(MatrixTest, MultiplicationRowMajorFloat)
// TEST(MatrixTest, MultiplicationColumnMajorFloat)
// TEST(MatrixTest, MultiplicationRowMajorFloatInPlace)
// TEST(MatrixTest, MultiplicationColumnMajorFloatInPlace)
// TEST(MatrixTest, ScalarDivisionFloat)
// TEST(MatrixTest, ScalarDivisionFailureFloat)
// TEST(MatrixTest, DeterminantFloat)
// TEST(MatrixTest, DeterminantFailureFloat)
// TEST(MatrixTest, InverseFloat)
// TEST(MatrixTest, InverseFailureFloat)
// TEST(MatrixTest, DeterminantFloat_1)
// TEST(MatrixTest, InverseFloat_1)
// TEST(MatrixTest, RankFloat)
// TEST(MatrixTest, RankFailureFloat)
// TEST(MatrixTest, RankFloat_2)
// TEST(MatrixTest, RankFailureFloat_2)
// TEST(MatrixTest, RankFullFloat)
// TEST(MatrixTest, MagnitudeFloat)
// TEST(MatrixTest, MagnitudeFailureFloat)
// TEST(MatrixTest, NormalizeFloat)
// TEST(MatrixTest, NormalizeFailureFloat)
// TEST(MatrixTest, NormalizeZeroVector)
// TEST(MatrixTest, TraceFloat)
// TEST(MatrixTest, TraceFailureFloat)

// ============================== FLOAT ==================================

TEST(MatrixTest, ConstructorDestructorFloat) {
  math::Matrix<float, 2, 2> matrix;
  // If the constructor and destructor work correctly, this test will pass
}

TEST(MatrixTest, ConstructorDestructorFailureFloat) {
  math::Matrix<float, 2, 2> matrix;
  // This test will pass because the matrix is not null after construction
  EXPECT_NE(&matrix, nullptr);
}

// Method: Matrix(const T& element)

TEST(MatrixTest, ElementConstructorFloat) {
  const float                     kElementValue = 5.0f;
  const math::Matrix<float, 3, 3> kMatrix(kElementValue);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(kMatrix(i, j), kElementValue);
    }
  }
}

// Method: Matrix(const Matrix& other)

TEST(MatrixTest, CopyConstructorFloat) {
  math::Matrix<float, 2, 2> matrix1(1.0f, 2.0f, 3.0f, 4.0f);
  math::Matrix<float, 2, 2> matrix2(matrix1);

  ASSERT_EQ(matrix2(0, 0), 1.0f);
  ASSERT_EQ(matrix2(0, 1), 2.0f);
  ASSERT_EQ(matrix2(1, 0), 3.0f);
  ASSERT_EQ(matrix2(1, 1), 4.0f);
}

// Method: Matrix(Matrix&& other)

TEST(MatrixTest, MoveConstructorFloat) {
  math::Matrix<float, 2, 2> matrix1(1.0f, 2.0f, 3.0f, 4.0f);
  math::Matrix<float, 2, 2> matrix2(std::move(matrix1));

  ASSERT_EQ(matrix2(0, 0), 1.0f);
  ASSERT_EQ(matrix2(0, 1), 2.0f);
  ASSERT_EQ(matrix2(1, 0), 3.0f);
  ASSERT_EQ(matrix2(1, 1), 4.0f);
}

// Method: Matrix& operator=(const Matrix& other)

TEST(MatrixTest, AssignmentOperatorFloat) {
  const math::Matrix<float, 3, 3> kOriginal(5.0f);
  math::Matrix<float, 3, 3>       copy;
  copy = kOriginal;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(copy(i, j), kOriginal(i, j));
    }
  }
}

// Method: coeff(row, col)

TEST(MatrixTest, ElementAccessFloat) {
  math::Matrix<float, 2, 2> matrix;
  matrix.coeffRef(0, 0) = 1;
  matrix.coeffRef(0, 1) = 2;
  matrix.coeffRef(1, 0) = 3;
  matrix.coeffRef(1, 1) = 4;
  EXPECT_EQ(matrix.coeff(0, 0), 1);
  EXPECT_EQ(matrix.coeff(0, 1), 2);
  EXPECT_EQ(matrix.coeff(1, 0), 3);
  EXPECT_EQ(matrix.coeff(1, 1), 4);
}

// Method: coeff(row, col) - failure

TEST(MatrixTest, ElementAccessFailureFloat) {
  math::Matrix<float, 2, 2> matrix;
  matrix.coeffRef(0, 0) = 1;
  matrix.coeffRef(0, 1) = 2;
  matrix.coeffRef(1, 0) = 3;
  matrix.coeffRef(1, 1) = 4;
  // These tests will pass because the expected values match the actual values
  EXPECT_NE(matrix.coeff(0, 0), 2);
  EXPECT_NE(matrix.coeff(0, 1), 1);
  EXPECT_NE(matrix.coeff(1, 0), 4);
  EXPECT_NE(matrix.coeff(1, 1), 3);
}

// Method: modification operator(row, col)

TEST(MatrixTest, ElementModificationFloat) {
  math::Matrix<float, 2, 2> matrix;
  matrix.coeffRef(0, 0) = 1;
  EXPECT_EQ(matrix.coeff(0, 0), 1);
  matrix.coeffRef(0, 0) = 2;
  EXPECT_EQ(matrix.coeff(0, 0), 2);
}

// Method: modification operator(row, col) - failure

TEST(MatrixTest, ElementModificationFailureFloat) {
  math::Matrix<float, 2, 2> matrix;
  matrix.coeffRef(0, 0) = 1;
  EXPECT_NE(matrix.coeff(0, 0), 2);
  matrix.coeffRef(0, 0) = 2;
  EXPECT_NE(matrix.coeff(0, 0), 1);
}

#ifdef _DEBUG

// Method: coeff(row, col) - out of bounds

TEST(MatrixTest, OutOfBoundsAccessFloat) {
  math::Matrix<float, 2, 2> matrix;
  EXPECT_DEATH(matrix.coeff(2, 2), "Index out of bounds");
  EXPECT_DEATH(matrix.coeffRef(2, 2), "Index out of bounds");
}

#endif  // DEBUG

// Method: access operator(row, col)

TEST(MatrixTest, OperatorAccessFloat) {
  math::Matrix<float, 2, 2> matrix;
  matrix(0, 0) = 1;
  EXPECT_EQ(matrix(0, 0), 1);
  matrix(0, 0) = 2;
  EXPECT_EQ(matrix(0, 0), 2);
}

// Method: access operator(row, col) - failure

TEST(MatrixTest, OperatorAccessConstFloat) {
  math::Matrix<float, 2, 2> matrix;
  matrix(0, 0)             = 1;
  const auto& const_matrix = matrix;
  EXPECT_EQ(const_matrix(0, 0), 1);
  // Check that the type of the returned reference is const
  EXPECT_TRUE(
      std::is_const_v<std::remove_reference_t<decltype(const_matrix(0, 0))>>);
}

// Method: const operator(row, col)

TEST(MatrixTest, CoeffAccessConstFloat) {
  math::Matrix<float, 2, 2> matrix;
  matrix.coeffRef(0, 0)    = 1;
  const auto& const_matrix = matrix;
  EXPECT_EQ(const_matrix.coeff(0, 0), 1);
  // Check that the type of the returned reference is const
  EXPECT_TRUE(std::is_const_v<
              std::remove_reference_t<decltype(const_matrix.coeff(0, 0))>>);
}

// Method: stack allocation

TEST(MatrixTest, StackAllocationFloat) {
  // This matrix should be small enough to be allocated on the stack
  math::Matrix<float, 2, 2> matrix;
  EXPECT_FALSE(matrix.s_kUseHeap);
}

// Method: heap allocation

TEST(MatrixTest, HeapAllocationFloat) {
  // This matrix should be large enough to be allocated on the heap
  math::Matrix<float, 100, 100> matrix;
  EXPECT_TRUE(matrix.s_kUseHeap);
}

// Method: addition (matrix + matrix)

TEST(MatrixTest, AdditionFloat) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<float, kRows, kColumns> matrix1;
  math::Matrix<float, kRows, kColumns> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
      matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
    }
  }

  auto matrix3 = matrix1 + matrix2;

  // Check that each element of the result is the sum of the corresponding
  // elements of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      EXPECT_EQ(matrix3.coeff(i, j), matrix1.coeff(i, j) + matrix2.coeff(i, j));
    }
  }
}

// Method: addition (matrix + matrix) - failure

TEST(MatrixTest, AdditionFailureFloat) {
  math::Matrix<float, 4, 4, math::Options::ColumnMajor> matrix1;
  math::Matrix<float, 4, 4, math::Options::ColumnMajor> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
      matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
    }
  }

  auto matrix3 = matrix1 + matrix2;

  // Check that each element of the result is not the sum of the corresponding
  // elements of matrix1 and matrix2 plus 1
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NE(matrix3.coeff(i, j),
                matrix1.coeff(i, j) + matrix2.coeff(i, j) + 1);
    }
  }
}

// Method: addition (matrix + scalar)

TEST(MatrixTest, ScalarAdditionFloat) {
  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix2 = matrix1 + 10;

  // Check that each element of the result is the corresponding element of
  // matrix1 plus 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) + 10);
    }
  }
}

// Method: addition (matrix + scalar) - failure

TEST(MatrixTest, ScalarAdditionFailureFloat) {
  math::Matrix<float, 4, 4> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<float, 4, 4> matrix2 = matrix1 + 10;

  // Check that each element of the result is not the corresponding element of
  // matrix1 plus 11
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NE(matrix2.coeff(i, j), matrix1.coeff(i, j) + 11);
    }
  }
}

// Method: subtraction (matrix - matrix)

TEST(MatrixTest, SubtractionFloat) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<int, kRows, kColumns> matrix1;
  math::Matrix<int, kRows, kColumns> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
      matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
    }
  }

  auto matrix3 = matrix1 - matrix2;

  // Check that each element of the result is the difference of the
  // corresponding elements of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      EXPECT_EQ(matrix3.coeff(i, j), matrix1.coeff(i, j) - matrix2.coeff(i, j));
    }
  }
}

// Method: subtraction (matrix - scalar)

TEST(MatrixTest, ScalarSubtractionFloat) {
  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix2 = matrix1 - 10;

  // Check that each element of the result is the corresponding element of
  // matrix1 minus 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) - 10);
    }
  }
}

// Method: row major multiplication (matrix * matrix)

TEST(MatrixTest, MultiplicationRowMajorFloat) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<float, kRows, kColumns, math::Options::RowMajor> matrix1;
  math::Matrix<float, kRows, kColumns, math::Options::RowMajor> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
      matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
    }
  }

  auto matrix3 = matrix1 * matrix2;

  // Check that each element of the result is the correct multiplication of the
  // corresponding rows and columns of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      float expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: row major multiplication (matrix * matrix) - non square matrices

TEST(MatrixTest, MultiplicationRowMajorFloatNonSquare) {
  constexpr int kRowsA      = 3;
  constexpr int kColsARowsB = 4;
  constexpr int kColsB      = 2;

  math::Matrix<float, kRowsA, kColsARowsB, math::Options::RowMajor> matrix1;
  math::Matrix<float, kColsARowsB, kColsB, math::Options::RowMajor> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRowsA; ++i) {
    for (int j = 0; j < kColsARowsB; ++j) {
      matrix1.coeffRef(i, j) = static_cast<float>(rand())
                             / RAND_MAX;  // random values between 0 and 1
    }
  }

  for (int i = 0; i < kColsARowsB; ++i) {
    for (int j = 0; j < kColsB; ++j) {
      matrix2.coeffRef(i, j) = static_cast<float>(rand())
                             / RAND_MAX;  // random values between 0 and 1
    }
  }

  auto matrix3 = matrix1 * matrix2;

  for (int i = 0; i < kRowsA; ++i) {
    for (int j = 0; j < kColsB; ++j) {
      float expected_value = 0;
      for (int k = 0; k < kColsARowsB; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_NEAR(matrix3.coeff(i, j),
                  expected_value,
                  1e-5);  // Using EXPECT_NEAR due to potential precision issues
    }
  }
}

// Method: row major multiplication (matrix * matrix) - non square matrices with
// precise values

TEST(MatrixTest, MultiplicationRowMajorFloatNonSquare_2) {
  constexpr int kRowsA      = 3;
  constexpr int kColsARowsB = 4;
  constexpr int kColsB      = 2;

  math::Matrix<float, kRowsA, kColsARowsB, math::Options::RowMajor> matrix1;
  math::Matrix<float, kColsARowsB, kColsB, math::Options::RowMajor> matrix2;

  // Populate matrix1
  matrix1.coeffRef(0, 0) = 1;
  matrix1.coeffRef(0, 1) = 2;
  matrix1.coeffRef(0, 2) = 3;
  matrix1.coeffRef(0, 3) = 4;
  matrix1.coeffRef(1, 0) = 5;
  matrix1.coeffRef(1, 1) = 6;
  matrix1.coeffRef(1, 2) = 7;
  matrix1.coeffRef(1, 3) = 8;
  matrix1.coeffRef(2, 0) = 9;
  matrix1.coeffRef(2, 1) = 10;
  matrix1.coeffRef(2, 2) = 11;
  matrix1.coeffRef(2, 3) = 12;

  // Populate matrix2
  matrix2.coeffRef(0, 0) = 2;
  matrix2.coeffRef(0, 1) = 3;
  matrix2.coeffRef(1, 0) = 4;
  matrix2.coeffRef(1, 1) = 5;
  matrix2.coeffRef(2, 0) = 6;
  matrix2.coeffRef(2, 1) = 7;
  matrix2.coeffRef(3, 0) = 8;
  matrix2.coeffRef(3, 1) = 9;

  auto matrix3 = matrix1 * matrix2;

  // Expected values based on manual matrix multiplication
  EXPECT_EQ(matrix3.coeff(0, 0), 60);
  EXPECT_EQ(matrix3.coeff(0, 1), 70);
  EXPECT_EQ(matrix3.coeff(1, 0), 140);
  EXPECT_EQ(matrix3.coeff(1, 1), 166);
  EXPECT_EQ(matrix3.coeff(2, 0), 220);
  EXPECT_EQ(matrix3.coeff(2, 1), 262);
}

// Method: column major multiplication (matrix * matrix)

TEST(MatrixTest, MultiplicationColumnMajorFloat) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<float, kRows, kColumns, math::Options::ColumnMajor> matrix1;
  math::Matrix<float, kRows, kColumns, math::Options::ColumnMajor> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
      matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
    }
  }

  auto matrix3 = matrix1 * matrix2;

  // Check that each element of the result is the correct multiplication of the
  // corresponding rows and columns of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      float expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: row major multiplication (matrix *= matrix)

TEST(MatrixTest, MultiplicationRowMajorFloatInPlace) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<float, kRows, kColumns, math::Options::RowMajor> matrix1;
  math::Matrix<float, kRows, kColumns, math::Options::RowMajor> matrix2;

  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
      matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
    }
  }

  auto matrix3  = matrix1;
  matrix3      *= matrix2;

  // Check that each element of the result is the correct multiplication of the
  // corresponding rows and columns of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      float expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: column major multiplication (matrix *= matrix)

TEST(MatrixTest, MultiplicationColumnMajorFloatInPlace) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<float, kRows, kColumns, math::Options::ColumnMajor> matrix1;
  math::Matrix<float, kRows, kColumns, math::Options::ColumnMajor> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
      matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
    }
  }

  auto matrix3  = matrix1;  // copy of matrix1
  matrix3      *= matrix2;

  // Check that each element of the result is the correct multiplication of the
  // corresponding rows and columns of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      float expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: scalar multiplication

TEST(MatrixTest, ScalarMultiplicationFloat) {
  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix1;
  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }
  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix2 = matrix1 * 10;
  // Check that each element of the result is the corresponding element of
  // matrix1 multiplied by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) * 10);
    }
  }
}

// Method: scalar division

TEST(MatrixTest, ScalarDivisionFloat) {
  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix2 = matrix1 / 10;

  // Check that each element of the result is the corresponding element of
  // matrix1 divided by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) / 10);
    }
  }
}

// Method: scalar division - failure

TEST(MatrixTest, ScalarDivisionFailureFloat) {
  math::Matrix<float, 4, 4> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<float, 4, 4> matrix2 = matrix1 / 10;

  // Check that each element of the result is not the corresponding element of
  // matrix1 divided by 11
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NE(matrix2.coeff(i, j), matrix1.coeff(i, j) / 11);
    }
  }
}

// Method: scalar division in place

TEST(MatrixTest, ScalarDivisionInPlaceFloat) {
  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix;

  // Populate matrix with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix.coeffRef(i, j) = (i * 4 + j + 1);
    }
  }

  matrix /= 10.0f;

  // Check that each element of the matrix is the original value divided by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(matrix.coeff(i, j), (i * 4 + j + 1) / 10.0f);
    }
  }
}

// Method: scalar division in place - failure

TEST(MatrixTest, ScalarDivisionInPlaceFailureFloat) {
  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix;

  // Populate matrix with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix.coeffRef(i, j) = (i * 4 + j + 1);
    }
  }

  matrix /= 10.0f;

  // Check that each element of the matrix is the original value divided by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NE(matrix.coeff(i, j), (i * 4 + j + 1) / 11.0f);
    }
  }
}

// Method: matrix equality comparison

TEST(MatrixTest, MatrixEqualityTrueFloat) {
  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality comparison - failure

TEST(MatrixTest, MatrixEqualityFalseFloat) {
  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix2  = matrix1;
  matrix2.coeffRef(0, 0)                                     += 1;

  EXPECT_FALSE(matrix1 == matrix2);
}

// Method: matrix equality with very small numbers

TEST(MatrixTest, MatrixEqualityTrueFloatSmallNumbers) {
  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<float>::min();
  matrix1.coeffRef(0, 1) = std::numeric_limits<float>::min();

  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality with very large numbers

TEST(MatrixTest, MatrixEqualityTrueFloatLargeNumbers) {
  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<float>::max();
  matrix1.coeffRef(0, 1) = std::numeric_limits<float>::max();

  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality with numbers close to each other

TEST(MatrixTest, MatrixEqualityTrueFloatCloseNumbers) {
  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = 1.0f;
  matrix1.coeffRef(0, 1) = 1.0f;

  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix2 = matrix1;
  matrix2.coeffRef(0, 0) += std::numeric_limits<float>::epsilon();
  matrix2.coeffRef(0, 1) += std::numeric_limits<float>::epsilon();

  EXPECT_TRUE(matrix1 == matrix2);
}

TEST(MatrixTest, MatrixEqualityFailSmallDifferencesFloat) {
  math::Matrix<float, 1, 1, math::Options::RowMajor> matrix1;
  math::Matrix<float, 1, 1, math::Options::RowMajor> matrix2;

  matrix1.coeffRef(0, 0) = std::numeric_limits<float>::epsilon() / 2;
  matrix2.coeffRef(0, 0) = -std::numeric_limits<float>::epsilon() / 2;

  // Pay attention: this test case should fail since the difference between the
  // two numbers
  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality comparison

TEST(MatrixTest, MatrixEqualityLargeNumbersFloat) {
  math::Matrix<float, 1, 1, math::Options::RowMajor> matrix1;
  math::Matrix<float, 1, 1, math::Options::RowMajor> matrix2;

  matrix1.coeffRef(0, 0) = 1e10f;
  matrix2.coeffRef(0, 0) = 1e10f + 1e-5f;

  // Pay attention:
  // Even though the relative difference between the two numbers is negligible,
  // they are considered not equal because the absolute difference exceeds
  // epsilon
  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix inequality comparison

TEST(MatrixTest, MatrixInequalityTrueFloat) {
  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix2  = matrix1;
  matrix2.coeffRef(0, 0)                                     += 1;

  EXPECT_TRUE(matrix1 != matrix2);
}

// Method: matrix inequality with very small numbers

TEST(MatrixTest, MatrixInequalityFalseFloatSmallNumbers) {
  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<float>::min();
  matrix1.coeffRef(0, 1) = std::numeric_limits<float>::min();

  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: matrix inequality with very large numbers

TEST(MatrixTest, MatrixInequalityFalseFloatLargeNumbers) {
  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<float>::max();
  matrix1.coeffRef(0, 1) = std::numeric_limits<float>::max();

  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: matrix inequality with numbers close to each other

TEST(MatrixTest, MatrixInequalityFalseFloatCloseNumbers) {
  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = 1.0f;
  matrix1.coeffRef(0, 1) = 1.0f;

  math::Matrix<float, 1, 2, math::Options::RowMajor> matrix2 = matrix1;
  matrix2.coeffRef(0, 0) += std::numeric_limits<float>::epsilon();
  matrix2.coeffRef(0, 1) += std::numeric_limits<float>::epsilon();

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: matrix inequality comparison

TEST(MatrixTest, MatrixInequalityFalseFloat) {
  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<float, 4, 4, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: transpose

TEST(MatrixTest, TransposeFloat) {
  math::Matrix<float, 2, 2> matrix1(1.0f, 2.0f, 3.0f, 4.0f);
  auto                      matrix2 = matrix1.transpose();

  ASSERT_EQ(matrix2(0, 0), 1.0f);
  ASSERT_EQ(matrix2(0, 1), 3.0f);
  ASSERT_EQ(matrix2(1, 0), 2.0f);
  ASSERT_EQ(matrix2(1, 1), 4.0f);
}

// Method: reshape

TEST(MatrixTest, ReshapeFloat) {
  math::Matrix<float, 2, 2> matrix1(1.0f, 2.0f, 3.0f, 4.0f);
  auto                      matrix2 = matrix1.reshape<4, 1>();

  ASSERT_EQ(matrix2(0, 0), 1.0f);
  ASSERT_EQ(matrix2(1, 0), 2.0f);
  ASSERT_EQ(matrix2(2, 0), 3.0f);
  ASSERT_EQ(matrix2(3, 0), 4.0f);
}

// Method: determinant

TEST(MatrixTest, DeterminantFloat) {
  math::Matrix<float, 2, 2> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(1, 0) = 3;
  matrix(1, 1) = 4;

  // Expected determinant is 1*4 - 2*3 = -2
  EXPECT_FLOAT_EQ(matrix.determinant(), -2);
}

// Method: determinant - failure

TEST(MatrixTest, DeterminantFailureFloat) {
  math::Matrix<float, 2, 2> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(1, 0) = 3;
  matrix(1, 1) = 4;

  // Incorrect determinant is -3, the actual determinant is -2
  EXPECT_NE(matrix.determinant(), -3);
}

// Method: determinant - 4x4 matrix

TEST(MatrixTest, DeterminantFloat_1) {
  // Create a 4x4 matrix with arbitrary values
  math::Matrix<float, 4, 4> matrix;
  matrix << 5, 7, 6, 1, 2, 8, 4, 6, 3, 4, 2, 7, 7, 3, 5, 1;

  EXPECT_FLOAT_EQ(matrix.determinant(), -52);
}

// Method: inverse

TEST(MatrixTest, InverseFloat) {
  math::Matrix<float, 2, 2> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(1, 0) = 3;
  matrix(1, 1) = 4;

  math::Matrix<float, 2, 2> inverseMatrix = matrix.inverse();

  // The expected inverse matrix of [[1, 2], [3, 4]] is [[-2, 1], [1.5, -0.5]]
  EXPECT_FLOAT_EQ(inverseMatrix(0, 0), -2);
  EXPECT_FLOAT_EQ(inverseMatrix(0, 1), 1);
  EXPECT_FLOAT_EQ(inverseMatrix(1, 0), 1.5);
  EXPECT_FLOAT_EQ(inverseMatrix(1, 1), -0.5);
}

// Method: inverse - 4x4 matrix

TEST(MatrixTest, InverseFloat_1) {
  // Create a 4x4 matrix with arbitrary values that's invertible
  math::Matrix<float, 4, 4> matrix;
  matrix << 5, 7, 6, 1, 2, 8, 4, 6, 3, 4, 2, 7, 7, 3, 5, 1;

  // Confirm the matrix is invertible
  ASSERT_FALSE(matrix.determinant() == 0);

  math::Matrix<float, 4, 4> inverseMatrix = matrix.inverse();

  // The product of a matrix and its inverse should be the identity matrix
  math::Matrix<float, 4, 4> productMatrix = matrix * inverseMatrix;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      // If i==j, we are on the diagonal and the value should be close to 1.
      // Otherwise, it should be close to 0. We use EXPECT_NEAR because of
      // possible floating point inaccuracies.
      EXPECT_NEAR(productMatrix(i, j), (i == j) ? 1.0f : 0.0f, 1e-5);
    }
  }
}

// Method: inverse - failure

TEST(MatrixTest, InverseFailureFloat) {
  math::Matrix<float, 2, 2> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(1, 0) = 3;
  matrix(1, 1) = 4;

  math::Matrix<float, 2, 2> inverseMatrix = matrix.inverse();

  // The incorrect inverse matrix of [[1, 2], [3, 4]] is [[-2, 1], [2, -1]]
  EXPECT_NE(inverseMatrix(0, 0), -2);
  EXPECT_NE(inverseMatrix(0, 1), 1);
  EXPECT_NE(inverseMatrix(1, 0), 2);
  EXPECT_NE(inverseMatrix(1, 1), -1);
}

// Method: rank

TEST(MatrixTest, RankFloat) {
  math::Matrix<float, 3, 3> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(0, 2) = 3;
  matrix(1, 0) = 2;
  matrix(1, 1) = 4;
  matrix(1, 2) = 6;
  matrix(2, 0) = 3;
  matrix(2, 1) = 6;
  matrix(2, 2) = 9;

  // This matrix has rank 1 as all rows are linearly dependent
  EXPECT_EQ(matrix.rank(), 1);
}

// Method: rank - 2 test

TEST(MatrixTest, RankFloat_2) {
  math::Matrix<float, 3, 3> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(0, 2) = 3;
  matrix(1, 0) = 4;
  matrix(1, 1) = 5;
  matrix(1, 2) = 6;
  matrix(2, 0) = 7;
  matrix(2, 1) = 8;
  matrix(2, 2) = 9;

  // This matrix has full rank (rank = min(m, n) = 3) as no row is a linear
  // combination of others
  EXPECT_EQ(matrix.rank(), 2);
}

// Method: rank - full

TEST(MatrixTest, RankFullFloat) {
  math::Matrix<float, 3, 3> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 0;
  matrix(0, 2) = 0;
  matrix(1, 0) = 0;
  matrix(1, 1) = 1;
  matrix(1, 2) = 0;
  matrix(2, 0) = 0;
  matrix(2, 1) = 0;
  matrix(2, 2) = 1;

  // This matrix has full rank (rank = min(m, n) = 3) as no row is a linear
  // combination of others
  EXPECT_EQ(matrix.rank(), 3);
}

// Method: rank - failure

TEST(MatrixTest, RankFailureFloat) {
  math::Matrix<float, 3, 3> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(0, 2) = 3;
  matrix(1, 0) = 2;
  matrix(1, 1) = 4;
  matrix(1, 2) = 6;
  matrix(2, 0) = 3;
  matrix(2, 1) = 6;
  matrix(2, 2) = 9;

  // The actual rank is 1, not 2
  EXPECT_NE(matrix.rank(), 2);
}

// Method: rank - 2 test - failure

TEST(MatrixTest, RankFailureFloat_2) {
  math::Matrix<float, 3, 3> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(0, 2) = 3;
  matrix(1, 0) = 4;
  matrix(1, 1) = 5;
  matrix(1, 2) = 6;
  matrix(2, 0) = 7;
  matrix(2, 1) = 8;
  matrix(2, 2) = 9;

  // The actual rank is 3, not 2
  EXPECT_NE(matrix.rank(), 3);
}

// Method: magnitude

TEST(MatrixTest, MagnitudeFloat) {
  math::Matrix<float, 3, 1> vector(2.0f, 2.0f, 1.0f);  // 3D vector

  auto magnitude = vector.magnitude();
  // Expected magnitude is sqrt(2^2 + 2^2 + 1^2) = sqrt(9) = 3
  EXPECT_FLOAT_EQ(vector.magnitude(), 3);
}

// Method: magnitude - failure

TEST(MatrixTest, MagnitudeFailureFloat) {
  math::Matrix<float, 3, 1> vector(2.0f, 2.0f, 1.0f);  // 3D vector

  // Incorrect magnitude is 2, the actual magnitude is 3
  EXPECT_NE(vector.magnitude(), 2);
}

// Method: normalize

TEST(MatrixTest, NormalizeFloat) {
  math::Matrix<float, 3, 1> vector(2.0f, 2.0f, 1.0f);  // 3D vector

#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  vector.normalize();
  auto& normalizedVector = vector;
#else
  auto normalizedVector = vector.normalize();
#endif

  // The expected normalized vector is (2/3, 2/3, 1/3)
  EXPECT_NEAR(normalizedVector(0, 0), 2.0f / 3, 1e-5);
  EXPECT_NEAR(normalizedVector(1, 0), 2.0f / 3, 1e-5);
  EXPECT_NEAR(normalizedVector(2, 0), 1.0f / 3, 1e-5);
}

// Method: normalize - failure

TEST(MatrixTest, NormalizeFailureFloat) {
  math::Matrix<float, 3, 1> vector(2.0f, 2.0f, 1.0f);  // 3D vector

#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  vector.normalize();
  auto& normalizedVector = vector;
#else
  auto normalizedVector = vector.normalize();
#endif

  vector *= 2;

  // The incorrect normalized vector is (1, 1, 0.5), the actual normalized
  // vector is (2/3, 2/3, 1/3)
  EXPECT_NE(normalizedVector(0, 0), 1);
  EXPECT_NE(normalizedVector(1, 0), 1);
  EXPECT_NE(normalizedVector(2, 0), 0.5);
}

#ifdef _DEBUG

// Method: normalize - zero vector

TEST(MatrixTest, NormalizeZeroVector) {
  math::Matrix<float, 3, 1> vector(0.0f, 0.0f, 0.0f);  // Zero vector

  // Trying to normalize a zero vector should throw an assertion
  EXPECT_DEATH(
      vector.normalize(),
      "Normalization error: magnitude is zero, implying a zero matrix/vector");
}

#endif  // _DEBUG

// Method: trace

TEST(MatrixTest, TraceFloat) {
  math::Matrix<float, 3, 3> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(0, 2) = 3;
  matrix(1, 0) = 4;
  matrix(1, 1) = 5;
  matrix(1, 2) = 6;
  matrix(2, 0) = 7;
  matrix(2, 1) = 8;
  matrix(2, 2) = 9;

  // The trace of the matrix is the sum of the elements on the main diagonal
  // For this matrix, the trace is 1 + 5 + 9 = 15
  EXPECT_FLOAT_EQ(matrix.trace(), 15);
}

// Method: trace - failure

TEST(MatrixTest, TraceFailureFloat) {
  math::Matrix<float, 3, 3> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(0, 2) = 3;
  matrix(1, 0) = 4;
  matrix(1, 1) = 5;
  matrix(1, 2) = 6;
  matrix(2, 0) = 7;
  matrix(2, 1) = 8;
  matrix(2, 2) = 9;

  // The incorrect trace is 14, the actual trace is 15
  EXPECT_NE(matrix.trace(), 14);
}

// Method: determinant

// 1. Perpendicular vectors
TEST(MatrixTest, DotProductFloatPerpendicular) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = 1;
  vector1.coeffRef(1, 0) = 0;
  vector1.coeffRef(2, 0) = 0;

  vector2.coeffRef(0, 0) = 0;
  vector2.coeffRef(1, 0) = 1;
  vector2.coeffRef(2, 0) = 0;

  float result = vector1.dot(vector2);
  EXPECT_EQ(result, 0.0f);
}

// 2. Parallel vectors
TEST(MatrixTest, DotProductFloatParallel) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = 2;
  vector1.coeffRef(1, 0) = 2;
  vector1.coeffRef(2, 0) = 2;

  vector2.coeffRef(0, 0) = 2;
  vector2.coeffRef(1, 0) = 2;
  vector2.coeffRef(2, 0) = 2;

  float result = vector1.dot(vector2);
  EXPECT_EQ(result, 12.0f);
}

// 3. Zero vector
TEST(MatrixTest, DotProductFloatZeroVector) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<float, 3, 1, math::Options::RowMajor> zeroVector(0);

  vector1.coeffRef(0, 0) = 5;
  vector1.coeffRef(1, 0) = 3;
  vector1.coeffRef(2, 0) = 2;

  float result = vector1.dot(zeroVector);
  EXPECT_EQ(result, 0.0f);
}

// 4. Unit vectors
TEST(MatrixTest, DotProductFloatUnitVectors) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> unitVector1(1.0f
                                                                 / sqrt(3));
  math::Matrix<float, 3, 1, math::Options::RowMajor> unitVector2(1.0f
                                                                 / sqrt(3));

  float result = unitVector1.dot(unitVector2);
  EXPECT_NEAR(result, 1.0f, 1e-5);
}

// 5. Negative vectors
TEST(MatrixTest, DotProductFloatNegative) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = -2;
  vector1.coeffRef(1, 0) = 3;
  vector1.coeffRef(2, 0) = -4;

  vector2.coeffRef(0, 0) = 5;
  vector2.coeffRef(1, 0) = -6;
  vector2.coeffRef(2, 0) = 7;

  float result          = vector1.dot(vector2);
  float expected_result = (-2 * 5 + 3 * (-6) + (-4 * 7));
  EXPECT_EQ(result, expected_result);
}

// 6. Random vectors
TEST(MatrixTest, DotProductFloatRandom) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector2;

  for (int i = 0; i < 3; ++i) {
    vector1.coeffRef(i, 0) = static_cast<float>(rand()) / RAND_MAX;
    vector2.coeffRef(i, 0) = static_cast<float>(rand()) / RAND_MAX;
  }

  float result          = vector1.dot(vector2);
  float expected_result = 0;
  for (int i = 0; i < 3; ++i) {
    expected_result += vector1.coeff(i, 0) * vector2.coeff(i, 0);
  }
  EXPECT_NEAR(result, expected_result, 1e-5);
}

// Method: cross

// 1. Perpendicular vectors
TEST(MatrixTest, CrossProductFloatPerpendicular) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = 1;
  vector1.coeffRef(1, 0) = 0;
  vector1.coeffRef(2, 0) = 0;

  vector2.coeffRef(0, 0) = 0;
  vector2.coeffRef(1, 0) = 1;
  vector2.coeffRef(2, 0) = 0;

  auto result = vector1.cross(vector2);

  EXPECT_EQ(result.coeffRef(0, 0), 0);
  EXPECT_EQ(result.coeffRef(1, 0), 0);
  EXPECT_EQ(result.coeffRef(2, 0), 1);
}

// 2. Parallel vectors
TEST(MatrixTest, CrossProductFloatParallel) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = 2;
  vector1.coeffRef(1, 0) = 2;
  vector1.coeffRef(2, 0) = 2;

  vector2.coeffRef(0, 0) = 2;
  vector2.coeffRef(1, 0) = 2;
  vector2.coeffRef(2, 0) = 2;

  auto result = vector1.cross(vector2);

  EXPECT_EQ(result.coeffRef(0, 0), 0);
  EXPECT_EQ(result.coeffRef(1, 0), 0);
  EXPECT_EQ(result.coeffRef(2, 0), 0);
}

// 3. Zero vector
TEST(MatrixTest, CrossProductFloatZeroVector) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<float, 3, 1, math::Options::RowMajor> zeroVector(0);

  vector1.coeffRef(0, 0) = 5;
  vector1.coeffRef(1, 0) = 3;
  vector1.coeffRef(2, 0) = 2;

  auto result = vector1.cross(zeroVector);

  EXPECT_EQ(result.coeffRef(0, 0), 0);
  EXPECT_EQ(result.coeffRef(1, 0), 0);
  EXPECT_EQ(result.coeffRef(2, 0), 0);
}

// 4. Unit vectors
TEST(MatrixTest, CrossProductFloatUnitVectors) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> unitVector1(1.0f
                                                                 / sqrt(3));
  math::Matrix<float, 3, 1, math::Options::RowMajor> unitVector2(1.0f
                                                                 / sqrt(3));

  auto result = unitVector1.cross(unitVector2);

  EXPECT_NEAR(result.coeffRef(0, 0), 0.0f, 1e-5);
  EXPECT_NEAR(result.coeffRef(1, 0), 0.0f, 1e-5);
  EXPECT_NEAR(result.coeffRef(2, 0), 0.0f, 1e-5);
}

// 5. Negative vectors
TEST(MatrixTest, CrossProductFloatNegative) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = -2;
  vector1.coeffRef(1, 0) = 3;
  vector1.coeffRef(2, 0) = -4;

  vector2.coeffRef(0, 0) = 5;
  vector2.coeffRef(1, 0) = -6;
  vector2.coeffRef(2, 0) = 7;

  auto result = vector1.cross(vector2);

  EXPECT_EQ(result.coeffRef(0, 0), 3 * 7 - (-4) * (-6));
  EXPECT_EQ(result.coeffRef(1, 0), -4 * 5 - (-2) * 7);
  EXPECT_EQ(result.coeffRef(2, 0), -2 * (-6) - 3 * 5);
}

// 6. Random vectors
TEST(MatrixTest, CrossProductFloatRandom) {
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<float, 3, 1, math::Options::RowMajor> vector2;

  for (int i = 0; i < 3; ++i) {
    vector1.coeffRef(i, 0) = static_cast<float>(rand()) / RAND_MAX;
    vector2.coeffRef(i, 0) = static_cast<float>(rand()) / RAND_MAX;
  }

  auto result = vector1.cross(vector2);
  math::Matrix<float, 3, 1, math::Options::RowMajor> expected_result;
  expected_result.coeffRef(0, 0) = vector1.coeff(1, 0) * vector2.coeff(2, 0)
                                 - vector1.coeff(2, 0) * vector2.coeff(1, 0);
  expected_result.coeffRef(1, 0) = vector1.coeff(2, 0) * vector2.coeff(0, 0)
                                 - vector1.coeff(0, 0) * vector2.coeff(2, 0);
  expected_result.coeffRef(2, 0) = vector1.coeff(0, 0) * vector2.coeff(1, 0)
                                 - vector1.coeff(1, 0) * vector2.coeff(0, 0);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(result.coeff(i, 0), expected_result.coeff(i, 0), 1e-5);
  }
}

// ============================== DOUBLE ==================================

// Matrix equality with very small numbers
TEST(MatrixTest, MatrixEqualityTrueDoubleSmallNumbers) {
  math::Matrix<double, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<double>::min();
  matrix1.coeffRef(0, 1) = std::numeric_limits<double>::min();

  math::Matrix<double, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  // Check that each element of matrix1 is equal to the corresponding element of
  // matrix2
  EXPECT_TRUE(matrix1 == matrix2);
}

// Matrix equality with very large numbers
TEST(MatrixTest, MatrixEqualityTrueDoubleLargeNumbers) {
  math::Matrix<double, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<double>::max();
  matrix1.coeffRef(0, 1) = std::numeric_limits<double>::max();

  math::Matrix<double, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  // Check that each element of matrix1 is equal to the corresponding element of
  // matrix2
  EXPECT_TRUE(matrix1 == matrix2);
}

// Matrix equality with numbers close to each other
TEST(MatrixTest, MatrixEqualityTrueDoubleCloseNumbers) {
  math::Matrix<double, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = 1.0;
  matrix1.coeffRef(0, 1) = 1.0;

  math::Matrix<double, 1, 2, math::Options::RowMajor> matrix2 = matrix1;
  // Add a small difference within the range of epsilon
  matrix2.coeffRef(0, 0) += std::numeric_limits<double>::epsilon();
  matrix2.coeffRef(0, 1) += std::numeric_limits<double>::epsilon();

  // Check that each element of matrix1 is equal to the corresponding element of
  // matrix2
  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: row major multiplication (matrix * matrix)

TEST(MatrixTest, MultiplicationRowMajorDouble) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<double, kRows, kColumns, math::Options::RowMajor> matrix1;
  math::Matrix<double, kRows, kColumns, math::Options::RowMajor> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
      matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
    }
  }

  auto matrix3 = matrix1 * matrix2;

  // Check that each element of the result is the correct multiplication of the
  // corresponding rows and columns of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      double expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: row major multiplication (matrix * matrix) - non square matrices

TEST(MatrixTest, MultiplicationRowMajorDoubleNonSquare) {
  constexpr int kRowsA      = 3;
  constexpr int kColsARowsB = 4;
  constexpr int kColsB      = 2;

  math::Matrix<double, kRowsA, kColsARowsB, math::Options::RowMajor> matrix1;
  math::Matrix<double, kColsARowsB, kColsB, math::Options::RowMajor> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRowsA; ++i) {
    for (int j = 0; j < kColsARowsB; ++j) {
      matrix1.coeffRef(i, j) = static_cast<double>(rand())
                             / RAND_MAX;  // random values between 0 and 1
    }
  }

  for (int i = 0; i < kColsARowsB; ++i) {
    for (int j = 0; j < kColsB; ++j) {
      matrix2.coeffRef(i, j) = static_cast<double>(rand())
                             / RAND_MAX;  // random values between 0 and 1
    }
  }

  auto matrix3 = matrix1 * matrix2;

  for (int i = 0; i < kRowsA; ++i) {
    for (int j = 0; j < kColsB; ++j) {
      double expected_value = 0.0;
      for (int k = 0; k < kColsARowsB; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_NEAR(
          matrix3.coeff(i, j),
          expected_value,
          1e-9);  // Using EXPECT_NEAR with adjusted precision for double
    }
  }
}

// Method: row major multiplication (matrix * matrix) - non square matrices with
// precise values

TEST(MatrixTest, MultiplicationRowMajorDoubleNonSquare_2) {
  constexpr int kRowsA      = 3;
  constexpr int kColsARowsB = 4;
  constexpr int kColsB      = 2;

  math::Matrix<double, kRowsA, kColsARowsB, math::Options::RowMajor> matrix1;
  math::Matrix<double, kColsARowsB, kColsB, math::Options::RowMajor> matrix2;

  // Populate matrix1
  matrix1.coeffRef(0, 0) = 1.0;
  matrix1.coeffRef(0, 1) = 2.0;
  matrix1.coeffRef(0, 2) = 3.0;
  matrix1.coeffRef(0, 3) = 4.0;
  matrix1.coeffRef(1, 0) = 5.0;
  matrix1.coeffRef(1, 1) = 6.0;
  matrix1.coeffRef(1, 2) = 7.0;
  matrix1.coeffRef(1, 3) = 8.0;
  matrix1.coeffRef(2, 0) = 9.0;
  matrix1.coeffRef(2, 1) = 10.0;
  matrix1.coeffRef(2, 2) = 11.0;
  matrix1.coeffRef(2, 3) = 12.0;

  // Populate matrix2
  matrix2.coeffRef(0, 0) = 2.0;
  matrix2.coeffRef(0, 1) = 3.0;
  matrix2.coeffRef(1, 0) = 4.0;
  matrix2.coeffRef(1, 1) = 5.0;
  matrix2.coeffRef(2, 0) = 6.0;
  matrix2.coeffRef(2, 1) = 7.0;
  matrix2.coeffRef(3, 0) = 8.0;
  matrix2.coeffRef(3, 1) = 9.0;

  auto matrix3 = matrix1 * matrix2;

  // Expected values based on manual matrix multiplication
  EXPECT_EQ(matrix3.coeff(0, 0), 60.0);
  EXPECT_EQ(matrix3.coeff(0, 1), 70.0);
  EXPECT_EQ(matrix3.coeff(1, 0), 140.0);
  EXPECT_EQ(matrix3.coeff(1, 1), 166.0);
  EXPECT_EQ(matrix3.coeff(2, 0), 220.0);
  EXPECT_EQ(matrix3.coeff(2, 1), 262.0);
}

// Method: column major multiplication (matrix * matrix)

TEST(MatrixTest, MultiplicationColumnMajorDouble) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<double, kRows, kColumns, math::Options::ColumnMajor> matrix1;
  math::Matrix<double, kRows, kColumns, math::Options::ColumnMajor> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
      matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
    }
  }

  auto matrix3 = matrix1 * matrix2;

  // Check that each element of the result is the correct multiplication of the
  // corresponding rows and columns of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      double expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: row major multiplication (matrix *= matrix)

TEST(MatrixTest, MultiplicationRowMajorDoubleInPlace) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<double, kRows, kColumns, math::Options::RowMajor> matrix1;
  math::Matrix<double, kRows, kColumns, math::Options::RowMajor> matrix2;

  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
      matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
    }
  }

  auto matrix3  = matrix1;
  matrix3      *= matrix2;

  // Check that each element of the result is the correct multiplication of the
  // corresponding rows and columns of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      double expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_DOUBLE_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: column major multiplication (matrix *= matrix)

TEST(MatrixTest, MultiplicationColumnMajorDoubleInPlace) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<double, kRows, kColumns, math::Options::ColumnMajor> matrix1;
  math::Matrix<double, kRows, kColumns, math::Options::ColumnMajor> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
      matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
    }
  }

  auto matrix3  = matrix1;
  matrix3      *= matrix2;

  // Check that each element of the result is the correct multiplication of the
  // corresponding rows and columns of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      double expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_DOUBLE_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// ============================== INT ==================================

// Matrix equality with very small numbers
TEST(MatrixTest, MatrixEqualityTrueIntSmallNumbers) {
  math::Matrix<int, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<int>::min();
  matrix1.coeffRef(0, 1) = std::numeric_limits<int>::min();

  math::Matrix<int, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  // Check that each element of matrix1 is equal to the corresponding element of
  // matrix2
  EXPECT_TRUE(matrix1 == matrix2);
}

TEST(MatrixTest, MultiplicationRowMajorIntNonSquare_2) {
  constexpr int kRowsA      = 3;
  constexpr int kColsARowsB = 4;
  constexpr int kColsB      = 2;

  math::Matrix<int, kRowsA, kColsARowsB, math::Options::RowMajor> matrix1;
  math::Matrix<int, kColsARowsB, kColsB, math::Options::RowMajor> matrix2;

  // Populate matrix1
  matrix1.coeffRef(0, 0) = 1.0;
  matrix1.coeffRef(0, 1) = 2.0;
  matrix1.coeffRef(0, 2) = 3.0;
  matrix1.coeffRef(0, 3) = 4.0;
  matrix1.coeffRef(1, 0) = 5.0;
  matrix1.coeffRef(1, 1) = 6.0;
  matrix1.coeffRef(1, 2) = 7.0;
  matrix1.coeffRef(1, 3) = 8.0;
  matrix1.coeffRef(2, 0) = 9.0;
  matrix1.coeffRef(2, 1) = 10.0;
  matrix1.coeffRef(2, 2) = 11.0;
  matrix1.coeffRef(2, 3) = 12.0;

  // Populate matrix2
  matrix2.coeffRef(0, 0) = 2.0;
  matrix2.coeffRef(0, 1) = 3.0;
  matrix2.coeffRef(1, 0) = 4.0;
  matrix2.coeffRef(1, 1) = 5.0;
  matrix2.coeffRef(2, 0) = 6.0;
  matrix2.coeffRef(2, 1) = 7.0;
  matrix2.coeffRef(3, 0) = 8.0;
  matrix2.coeffRef(3, 1) = 9.0;

  auto matrix3 = matrix1 * matrix2;

  // Expected values based on manual matrix multiplication
  EXPECT_EQ(matrix3.coeff(0, 0), 60.0);
  EXPECT_EQ(matrix3.coeff(0, 1), 70.0);
  EXPECT_EQ(matrix3.coeff(1, 0), 140.0);
  EXPECT_EQ(matrix3.coeff(1, 1), 166.0);
  EXPECT_EQ(matrix3.coeff(2, 0), 220.0);
  EXPECT_EQ(matrix3.coeff(2, 1), 262.0);
}

#endif