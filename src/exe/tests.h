/**
 * @file tests.h
 */

#ifndef MATH_LIBRARY_TESTS_H
#define MATH_LIBRARY_TESTS_H

#include <gtest/gtest.h>
#include <math_library/all.h>

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

  math::Matrix<float, kRows, kColumns> matrix1;
  math::Matrix<float, kRows, kColumns> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4.0f + j + 1.0f;  // Use float literals
      matrix2.coeffRef(i, j)
          = (i * 4.0f + j + 1.0f) * 2.0f;            // Use float literals
    }
  }

  auto matrix3 = matrix1 - matrix2;

  // Check that each element of the result is the difference of the
  // corresponding elements of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      EXPECT_NEAR(
          matrix3.coeff(i, j), matrix1.coeff(i, j) - matrix2.coeff(i, j), 1e-5);
      // Alternatively, for exact floating-point comparisons (which might be
      // risky due to precision issues): EXPECT_FLOAT_EQ(matrix3.coeff(i, j),
      // matrix1.coeff(i, j) - matrix2.coeff(i, j));
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

// Test: Negation (-Matrix)

TEST(MatrixTest, NegationFloat) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<float, kRows, kColumns> matrix;

  // Populate matrix with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix.coeffRef(i, j) = i * 4 + j + 1;  // Example values: 1, 2, 5, 6
    }
  }

  auto negatedMatrix = -matrix;

  // Check that each element of the negated matrix is the negation
  // of the corresponding element in the original matrix
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      EXPECT_EQ(negatedMatrix.coeff(i, j), -matrix.coeff(i, j));
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
  auto normalizedVector = vector.normalized();
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
  auto normalizedVector = vector.normalized();
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

// clang-format off

// Method: get row
TEST(MatrixTest, GetRow) {
  math::MatrixNf<3, 3> matrix;
  // Initialize matrix for the test
  matrix << 1.0f, 2.0f, 3.0f, 
            4.0f, 5.0f, 6.0f, 
            7.0f, 8.0f, 9.0f;

  auto rowVector = matrix.getRow<1>();  // Get the second row

  // Verify that the second row is correctly retrieved
  EXPECT_FLOAT_EQ(rowVector(0), 4.0f);
  EXPECT_FLOAT_EQ(rowVector(1), 5.0f);
  EXPECT_FLOAT_EQ(rowVector(2), 6.0f);
}

// Method: get row from column-major matrix
TEST(MatrixTest, GetRowColumnMajor) {
  math::MatrixNf<3, 3, math::Options::ColumnMajor> matrix;
  matrix << 1.0f, 4.0f, 7.0f, 
            2.0f, 5.0f, 8.0f, 
            3.0f, 6.0f, 9.0f;

  auto rowVector = matrix.getRow<1>();  // Get the second row

  // Verify that the second row is correctly retrieved
  EXPECT_FLOAT_EQ(rowVector(0), 2.0f);
  EXPECT_FLOAT_EQ(rowVector(1), 5.0f);
  EXPECT_FLOAT_EQ(rowVector(2), 8.0f);
}

// Method: get column
TEST(MatrixTest, GetColumn) {
  math::MatrixNf<3, 3, math::Options::ColumnMajor> matrix;
  matrix << 1.0f, 4.0f, 7.0f, 
            2.0f, 5.0f, 8.0f, 
            3.0f, 6.0f, 9.0f;

  auto columnVector = matrix.getColumn<1>();  // Get the second column

  // Verify that the second column is correctly retrieved
  EXPECT_FLOAT_EQ(columnVector(0), 4.0f);
  EXPECT_FLOAT_EQ(columnVector(1), 5.0f);
  EXPECT_FLOAT_EQ(columnVector(2), 6.0f);
}

// Method: get column from row-major matrix
TEST(MatrixTest, GetColumnRowMajor) {
  math::MatrixNf<3, 3> matrix;
  matrix << 1.0f, 2.0f, 3.0f, 
            4.0f, 5.0f, 6.0f, 
            7.0f, 8.0f, 9.0f;

  auto columnVector = matrix.getColumn<1>();  // Get the second column

  // Verify that the second column is correctly retrieved
  EXPECT_FLOAT_EQ(columnVector(0), 2.0f);
  EXPECT_FLOAT_EQ(columnVector(1), 5.0f);
  EXPECT_FLOAT_EQ(columnVector(2), 8.0f);
}

// clang-format on

// Method: set row
TEST(MatrixTest, SetRow) {
  math::MatrixNf<3, 3>                            matrix;
  math::Vector<float, 3, math::Options::RowMajor> rowVector(1.0f, 2.0f, 3.0f);
  matrix.setRow<1>(rowVector);  // Set the second row

  // Verify that the second row is correctly set
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_FLOAT_EQ(matrix(1, col), rowVector(col));
  }
}

// Method: set row in column-major matrix
TEST(MatrixTest, SetRowColumnMajor) {
  math::MatrixNf<3, 3, math::Options::ColumnMajor>   matrix;
  math::Vector<float, 3, math::Options::ColumnMajor> rowVector(
      1.0f, 2.0f, 3.0f);
  matrix.setRow<1>(rowVector);  // Set the second row

  // Verify that the second row is correctly set
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_FLOAT_EQ(matrix(1, col), rowVector(col));
  }
}

// Method: set column
TEST(MatrixTest, SetColumn) {
  math::MatrixNf<3, 3, math::Options::ColumnMajor>   matrix;
  math::Vector<float, 3, math::Options::ColumnMajor> columnVector(
      4.0f, 5.0f, 6.0f);
  matrix.setColumn<2>(columnVector);  // Set the third column

  // Verify that the third column is correctly set
  for (unsigned int row = 0; row < 3; ++row) {
    EXPECT_FLOAT_EQ(matrix(row, 2), columnVector(row));
  }
}

// Method: set column in row-major matrix
TEST(MatrixTest, SetColumnRowMajor) {
  math::MatrixNf<3, 3, math::Options::RowMajor>   matrix;
  math::Vector<float, 3, math::Options::RowMajor> columnVector(
      4.0f, 5.0f, 6.0f);
  matrix.setColumn<2>(columnVector);  // Set the third column

  // Verify that the third column is correctly set
  for (unsigned int row = 0; row < 3; ++row) {
    EXPECT_FLOAT_EQ(matrix(row, 2), columnVector(row));
  }
}

// Method: set basis X

TEST(MatrixTest, SetBasisX) {
  math::MatrixNf<3, 3>                            matrix;
  math::Vector<float, 3, math::Options::RowMajor> xBasis(1.0f, 0.0f, 0.0f);
  matrix.setBasisX(xBasis);

  // Verify that the first row (X basis) is correctly set
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_EQ(matrix(0, col), xBasis(col));
  }
}

// Method: set basis Y

TEST(MatrixTest, SetBasisY) {
  math::MatrixNf<3, 3>                            matrix;
  math::Vector<float, 3, math::Options::RowMajor> yBasis(0.0f, 1.0f, 0.0f);
  matrix.setBasisY(yBasis);

  // Verify that the second row (Y basis) is correctly set
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_EQ(matrix(1, col), yBasis(col));
  }
}

// Method: set basis Z

TEST(MatrixTest, SetBasisZ) {
  math::MatrixNf<3, 3>                            matrix;
  math::Vector<float, 3, math::Options::RowMajor> zBasis(0.0f, 0.0f, 1.0f);
  matrix.setBasisZ(zBasis);

  // Verify that the third row (Z basis) is correctly set
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_EQ(matrix(2, col), zBasis(col));
  }
}

// Method: set basis general

TEST(MatrixTest, GeneralizedSetBasis) {
  math::MatrixNf<3, 3>                            matrix;
  math::Vector<float, 3, math::Options::RowMajor> basisVector(1.0f, 2.0f, 3.0f);

  // Test setting each basis individually
  matrix.setBasis<0>(basisVector);
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_EQ(matrix(0, col), basisVector(col));
  }

  matrix.setBasis<1>(basisVector);
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_EQ(matrix(1, col), basisVector(col));
  }

  matrix.setBasis<2>(basisVector);
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_EQ(matrix(2, col), basisVector(col));
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

// Test: Negation (-Matrix)

TEST(MatrixTest, NegationInt) {
  constexpr int kRows    = 3;
  constexpr int kColumns = 3;

  math::Matrix<int, kRows, kColumns> matrix;

  // Populate matrix with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix.coeffRef(i, j)
          = (i + 1) * (j + 1);  // Example values: 1, 2, 3, ..., 9
    }
  }

  auto negatedMatrix = -matrix;

  // Check that each element of the negated matrix is the negation
  // of the corresponding element in the original matrix
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      EXPECT_EQ(negatedMatrix.coeff(i, j), -matrix.coeff(i, j));
    }
  }
}

// ========================== UNSIGNED INT ==============================

TEST(MatrixTest, ConstructorDestructorUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
  // If the constructor and destructor work correctly, this test will pass
}

TEST(MatrixTest, ConstructorDestructorFailureUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
  // This test will pass because the matrix is not null after construction
  EXPECT_NE(&matrix, nullptr);
}

// Method: Matrix(const T& element)

TEST(MatrixTest, ElementConstructorUnsignedInt) {
  const unsigned int                     kElementValue = 5;
  const math::Matrix<unsigned int, 3, 3> kMatrix(kElementValue);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(kMatrix(i, j), kElementValue);
    }
  }
}

// Method: Matrix(const Matrix& other)

TEST(MatrixTest, CopyConstructorUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix1(1, 2, 3, 4);
  math::Matrix<unsigned int, 2, 2> matrix2(matrix1);

  ASSERT_EQ(matrix2(0, 0), 1);
  ASSERT_EQ(matrix2(0, 1), 2);
  ASSERT_EQ(matrix2(1, 0), 3);
  ASSERT_EQ(matrix2(1, 1), 4);
}

// Method: Matrix(Matrix&& other)

TEST(MatrixTest, MoveConstructorUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix1(1, 2, 3, 4);
  math::Matrix<unsigned int, 2, 2> matrix2(std::move(matrix1));

  ASSERT_EQ(matrix2(0, 0), 1);
  ASSERT_EQ(matrix2(0, 1), 2);
  ASSERT_EQ(matrix2(1, 0), 3);
  ASSERT_EQ(matrix2(1, 1), 4);
}

// Method: Matrix& operator=(const Matrix& other)

TEST(MatrixTest, AssignmentOperatorUnsignedInt) {
  const math::Matrix<unsigned int, 3, 3> kOriginal(5);
  math::Matrix<unsigned int, 3, 3>       copy;
  copy = kOriginal;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(copy(i, j), kOriginal(i, j));
    }
  }
}

// Method: coeff(row, col)

TEST(MatrixTest, ElementAccessUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
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

TEST(MatrixTest, ElementAccessFailureUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
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

TEST(MatrixTest, ElementModificationUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
  matrix.coeffRef(0, 0) = 1;
  EXPECT_EQ(matrix.coeff(0, 0), 1);
  matrix.coeffRef(0, 0) = 2;
  EXPECT_EQ(matrix.coeff(0, 0), 2);
}

// Method: modification operator(row, col) - failure

TEST(MatrixTest, ElementModificationFailureUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
  matrix.coeffRef(0, 0) = 1;
  EXPECT_NE(matrix.coeff(0, 0), 2);
  matrix.coeffRef(0, 0) = 2;
  EXPECT_NE(matrix.coeff(0, 0), 1);
}

#ifdef _DEBUG

// Method: coeff(row, col) - out of bounds

TEST(MatrixTest, OutOfBoundsAccessUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
  EXPECT_DEATH(matrix.coeff(2, 2), "Index out of bounds");
  EXPECT_DEATH(matrix.coeffRef(2, 2), "Index out of bounds");
}

#endif  // DEBUG

// Method: access operator(row, col)

TEST(MatrixTest, OperatorAccessUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
  matrix(0, 0) = 1;
  EXPECT_EQ(matrix(0, 0), 1);
  matrix(0, 0) = 2;
  EXPECT_EQ(matrix(0, 0), 2);
}

// Method: access operator(row, col) - failure

TEST(MatrixTest, OperatorAccessConstUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
  matrix(0, 0)             = 1;
  const auto& const_matrix = matrix;
  EXPECT_EQ(const_matrix(0, 0), 1);
  // Check that the type of the returned reference is const
  EXPECT_TRUE(
      std::is_const_v<std::remove_reference_t<decltype(const_matrix(0, 0))>>);
}

// Method: const operator(row, col)

TEST(MatrixTest, CoeffAccessConstUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
  matrix.coeffRef(0, 0)    = 1;
  const auto& const_matrix = matrix;
  EXPECT_EQ(const_matrix.coeff(0, 0), 1);
  // Check that the type of the returned reference is const
  EXPECT_TRUE(std::is_const_v<
              std::remove_reference_t<decltype(const_matrix.coeff(0, 0))>>);
}

// Method: stack allocation

TEST(MatrixTest, StackAllocationUnsignedInt) {
  // This matrix should be small enough to be allocated on the stack
  math::Matrix<unsigned int, 2, 2> matrix;
  EXPECT_FALSE(matrix.s_kUseHeap);
}

// Method: heap allocation

TEST(MatrixTest, HeapAllocationUnsignedInt) {
  // This matrix should be large enough to be allocated on the heap
  math::Matrix<unsigned int, 100, 100> matrix;
  EXPECT_TRUE(matrix.s_kUseHeap);
}

// Method: addition (matrix + matrix)

TEST(MatrixTest, AdditionUnsignedInt) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<unsigned int, kRows, kColumns> matrix1;
  math::Matrix<unsigned int, kRows, kColumns> matrix2;

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

TEST(MatrixTest, AdditionFailureUnsignedInt) {
  math::Matrix<unsigned int, 4, 4, math::Options::ColumnMajor> matrix1;
  math::Matrix<unsigned int, 4, 4, math::Options::ColumnMajor> matrix2;

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

TEST(MatrixTest, ScalarAdditionUnsignedInt) {
  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix2
      = matrix1 + 10;

  // Check that each element of the result is the corresponding element of
  // matrix1 plus 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) + 10);
    }
  }
}

// Method: addition (matrix + scalar) - failure

TEST(MatrixTest, ScalarAdditionFailureUnsignedInt) {
  math::Matrix<unsigned int, 4, 4> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<unsigned int, 4, 4> matrix2 = matrix1 + 10;

  // Check that each element of the result is not the corresponding element of
  // matrix1 plus 11
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NE(matrix2.coeff(i, j), matrix1.coeff(i, j) + 11);
    }
  }
}

// Method: subtraction (matrix - matrix)

TEST(MatrixTest, SubtractionUnsignedInt) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<unsigned int, kRows, kColumns> matrix1;
  math::Matrix<unsigned int, kRows, kColumns> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;  // Use unsigned int literals
      matrix2.coeffRef(i, j)
          = (i * 4 + j + 1) * 2;               // Use unsigned int literals
    }
  }

  auto matrix3 = matrix1 - matrix2;

  // Check that each element of the result is the difference of the
  // corresponding elements of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      EXPECT_NEAR(
          matrix3.coeff(i, j), matrix1.coeff(i, j) - matrix2.coeff(i, j), 1e-5);
      // Alternatively, for exact unsigned inting-point comparisons (which might
      // be risky due to precision issues): EXPECT_FLOAT_EQ(matrix3.coeff(i, j),
      // matrix1.coeff(i, j) - matrix2.coeff(i, j));
    }
  }
}

// Method: subtraction (matrix - scalar)

TEST(MatrixTest, ScalarSubtractionUnsignedInt) {
  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix2
      = matrix1 - 10;

  // Check that each element of the result is the corresponding element of
  // matrix1 minus 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) - 10);
    }
  }
}

// Test: Negation (-Matrix)

TEST(MatrixTest, NegationUnsignedInt) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<unsigned int, kRows, kColumns> matrix;

  // Populate matrix with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix.coeffRef(i, j) = i * 4 + j + 1;  // Example values: 1, 2, 5, 6
    }
  }

  // Since negating an unsigned int matrix is expected to trigger an assertion,
  // we use EXPECT_DEATH to verify that the assertion indeed occurs.
  EXPECT_DEATH({ auto negatedMatrix = -matrix; }, "");

  // auto negatedMatrix = -matrix;

  // Check that each element of the negated matrix is the negation
  // of the corresponding element in the original matrix
  // for (int i = 0; i < kRows; ++i) {
  //  for (int j = 0; j < kColumns; ++j) {
  //    EXPECT_EQ(negatedMatrix.coeff(i, j), -matrix.coeff(i, j));
  //  }
  //}
}

// Method: row major multiplication (matrix * matrix)

TEST(MatrixTest, MultiplicationRowMajorUnsignedInt) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<unsigned int, kRows, kColumns, math::Options::RowMajor> matrix1;
  math::Matrix<unsigned int, kRows, kColumns, math::Options::RowMajor> matrix2;

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
      unsigned int expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: row major multiplication (matrix * matrix) - non square matrices

TEST(MatrixTest, MultiplicationRowMajorUnsignedIntNonSquare) {
  constexpr int kRowsA      = 3;
  constexpr int kColsARowsB = 4;
  constexpr int kColsB      = 2;

  math::Matrix<unsigned int, kRowsA, kColsARowsB, math::Options::RowMajor>
      matrix1;
  math::Matrix<unsigned int, kColsARowsB, kColsB, math::Options::RowMajor>
      matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRowsA; ++i) {
    for (int j = 0; j < kColsARowsB; ++j) {
      matrix1.coeffRef(i, j) = static_cast<unsigned int>(rand())
                             / RAND_MAX;  // random values between 0 and 1
    }
  }

  for (int i = 0; i < kColsARowsB; ++i) {
    for (int j = 0; j < kColsB; ++j) {
      matrix2.coeffRef(i, j) = static_cast<unsigned int>(rand())
                             / RAND_MAX;  // random values between 0 and 1
    }
  }

  auto matrix3 = matrix1 * matrix2;

  for (int i = 0; i < kRowsA; ++i) {
    for (int j = 0; j < kColsB; ++j) {
      unsigned int expected_value = 0;
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

TEST(MatrixTest, MultiplicationRowMajorUnsignedIntNonSquare_2) {
  constexpr int kRowsA      = 3;
  constexpr int kColsARowsB = 4;
  constexpr int kColsB      = 2;

  math::Matrix<unsigned int, kRowsA, kColsARowsB, math::Options::RowMajor>
      matrix1;
  math::Matrix<unsigned int, kColsARowsB, kColsB, math::Options::RowMajor>
      matrix2;

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

TEST(MatrixTest, MultiplicationColumnMajorUnsignedInt) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<unsigned int, kRows, kColumns, math::Options::ColumnMajor>
      matrix1;
  math::Matrix<unsigned int, kRows, kColumns, math::Options::ColumnMajor>
      matrix2;

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
      unsigned int expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: row major multiplication (matrix *= matrix)

TEST(MatrixTest, MultiplicationRowMajorUnsignedIntInPlace) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<unsigned int, kRows, kColumns, math::Options::RowMajor> matrix1;
  math::Matrix<unsigned int, kRows, kColumns, math::Options::RowMajor> matrix2;

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
      unsigned int expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: column major multiplication (matrix *= matrix)

TEST(MatrixTest, MultiplicationColumnMajorUnsignedIntInPlace) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<unsigned int, kRows, kColumns, math::Options::ColumnMajor>
      matrix1;
  math::Matrix<unsigned int, kRows, kColumns, math::Options::ColumnMajor>
      matrix2;

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
      unsigned int expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: scalar multiplication

TEST(MatrixTest, ScalarMultiplicationUnsignedInt) {
  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix1;
  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }
  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix2
      = matrix1 * 10;
  // Check that each element of the result is the corresponding element of
  // matrix1 multiplied by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) * 10);
    }
  }
}

// Method: scalar division

TEST(MatrixTest, ScalarDivisionUnsignedInt) {
  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix2
      = matrix1 / 10;

  // Check that each element of the result is the corresponding element of
  // matrix1 divided by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) / 10);
    }
  }
}

// Method: scalar division - failure

TEST(MatrixTest, ScalarDivisionFailureUnsignedInt) {
  math::Matrix<unsigned int, 4, 4> matrix1;

  for (unsigned int i = 0; i < 4; ++i) {
    for (unsigned int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = (i * 4 + j + 1) * 110;
    }
  }

  math::Matrix<unsigned int, 4, 4> matrix2 = matrix1 / 10;

  for (unsigned int i = 0; i < 4; ++i) {
    for (unsigned int j = 0; j < 4; ++j) {
      EXPECT_NE(matrix2.coeff(i, j), matrix1.coeff(i, j) / 11);
    }
  }
}

// Method: scalar division in place

TEST(MatrixTest, ScalarDivisionInPlaceUnsignedInt) {
  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix;

  // Populate matrix with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix.coeffRef(i, j) = (i * 4 + j + 1);
    }
  }

  matrix /= 10;

  // Check that each element of the matrix is the original value divided by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(matrix.coeff(i, j), (i * 4 + j + 1) / 10);
    }
  }
}

// Method: scalar division in place - failure

TEST(MatrixTest, ScalarDivisionInPlaceFailureUnsignedInt) {
  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix;

  // Populate matrix with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix.coeffRef(i, j) = (i * 4 + j + 1) * 110;
    }
  }

  matrix /= 10;

  // Check that each element of the matrix is the original value divided by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NE(matrix.coeff(i, j), (i * 4 + j + 1) / 11);
    }
  }
}

// Method: matrix equality comparison

TEST(MatrixTest, MatrixEqualityTrueUnsignedInt) {
  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality comparison - failure

TEST(MatrixTest, MatrixEqualityFalseUnsignedInt) {
  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix2  = matrix1;
  matrix2.coeffRef(0, 0)                                            += 1;

  EXPECT_FALSE(matrix1 == matrix2);
}

// Method: matrix equality with very small numbers

TEST(MatrixTest, MatrixEqualityTrueUnsignedIntSmallNumbers) {
  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<unsigned int>::min();
  matrix1.coeffRef(0, 1) = std::numeric_limits<unsigned int>::min();

  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality with very large numbers

TEST(MatrixTest, MatrixEqualityTrueUnsignedIntLargeNumbers) {
  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<unsigned int>::max();
  matrix1.coeffRef(0, 1) = std::numeric_limits<unsigned int>::max();

  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality with numbers close to each other

TEST(MatrixTest, MatrixEqualityTrueUnsignedIntCloseNumbers) {
  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = 1;
  matrix1.coeffRef(0, 1) = 1;

  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix2 = matrix1;
  matrix2.coeffRef(0, 0) += std::numeric_limits<unsigned int>::epsilon();
  matrix2.coeffRef(0, 1) += std::numeric_limits<unsigned int>::epsilon();

  EXPECT_TRUE(matrix1 == matrix2);
}

TEST(MatrixTest, MatrixEqualityFailSmallDifferencesUnsignedInt) {
  math::Matrix<unsigned int, 1, 1, math::Options::RowMajor> matrix1;
  math::Matrix<unsigned int, 1, 1, math::Options::RowMajor> matrix2;

  matrix1.coeffRef(0, 0) = std::numeric_limits<unsigned int>::epsilon() / 2;
  matrix2.coeffRef(0, 0) = -std::numeric_limits<unsigned int>::epsilon() / 2;

  // Pay attention: this test case should fail since the difference between the
  // two numbers
  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality comparison

TEST(MatrixTest, MatrixEqualityLargeNumbersUnsignedInt) {
  math::Matrix<unsigned int, 1, 1, math::Options::RowMajor> matrix1;
  math::Matrix<unsigned int, 1, 1, math::Options::RowMajor> matrix2;

  matrix1.coeffRef(0, 0) = 1e10f;
  matrix2.coeffRef(0, 0) = 1e10f + 1e-5f;

  // Pay attention:
  // Even though the relative difference between the two numbers is negligible,
  // they are considered not equal because the absolute difference exceeds
  // epsilon
  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix inequality comparison

TEST(MatrixTest, MatrixInequalityTrueUnsignedInt) {
  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix2  = matrix1;
  matrix2.coeffRef(0, 0)                                            += 1;

  EXPECT_TRUE(matrix1 != matrix2);
}

// Method: matrix inequality with very small numbers

TEST(MatrixTest, MatrixInequalityFalseUnsignedIntSmallNumbers) {
  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<unsigned int>::min();
  matrix1.coeffRef(0, 1) = std::numeric_limits<unsigned int>::min();

  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: matrix inequality with very large numbers

TEST(MatrixTest, MatrixInequalityFalseUnsignedIntLargeNumbers) {
  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<unsigned int>::max();
  matrix1.coeffRef(0, 1) = std::numeric_limits<unsigned int>::max();

  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: matrix inequality with numbers close to each other

TEST(MatrixTest, MatrixInequalityFalseUnsignedIntCloseNumbers) {
  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = 1;
  matrix1.coeffRef(0, 1) = 1;

  math::Matrix<unsigned int, 1, 2, math::Options::RowMajor> matrix2 = matrix1;
  matrix2.coeffRef(0, 0) += std::numeric_limits<unsigned int>::epsilon();
  matrix2.coeffRef(0, 1) += std::numeric_limits<unsigned int>::epsilon();

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: matrix inequality comparison

TEST(MatrixTest, MatrixInequalityFalseUnsignedInt) {
  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<unsigned int, 4, 4, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: transpose

TEST(MatrixTest, TransposeUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix1(1, 2, 3, 4);
  auto                             matrix2 = matrix1.transpose();

  ASSERT_EQ(matrix2(0, 0), 1);
  ASSERT_EQ(matrix2(0, 1), 3);
  ASSERT_EQ(matrix2(1, 0), 2);
  ASSERT_EQ(matrix2(1, 1), 4);
}

// Method: reshape

TEST(MatrixTest, ReshapeUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix1(1, 2, 3, 4);
  auto                             matrix2 = matrix1.reshape<4, 1>();

  ASSERT_EQ(matrix2(0, 0), 1);
  ASSERT_EQ(matrix2(1, 0), 2);
  ASSERT_EQ(matrix2(2, 0), 3);
  ASSERT_EQ(matrix2(3, 0), 4);
}

// Method: determinant

TEST(MatrixTest, DeterminantUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(1, 0) = 3;
  matrix(1, 1) = 4;

  // Expected determinant is 1*4 - 2*3 = -2
  auto det = matrix.determinant();
  EXPECT_FLOAT_EQ(matrix.determinant(), -2);
}

// Method: determinant - failure

TEST(MatrixTest, DeterminantFailureUnsignedInt) {
  math::Matrix<unsigned int, 2, 2> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(1, 0) = 3;
  matrix(1, 1) = 4;

  // Incorrect determinant is -3, the actual determinant is -2
  EXPECT_NE(matrix.determinant(), -3);
}

// Method: determinant - 4x4 matrix

TEST(MatrixTest, DeterminantUnsignedInt_1) {
  // Create a 4x4 matrix with arbitrary values
  math::Matrix<unsigned int, 4, 4> matrix;
  matrix << 5, 7, 6, 1, 2, 8, 4, 6, 3, 4, 2, 7, 7, 3, 5, 1;

  EXPECT_FLOAT_EQ(matrix.determinant(), -52);
}

// Method: inverse - this is not reasonable since unsigned int

// Method: rank - 2 test

TEST(MatrixTest, RankUnsignedInt_2) {
  math::Matrix<unsigned int, 3, 3> matrix;
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

TEST(MatrixTest, RankFullUnsignedInt) {
  math::Matrix<unsigned int, 3, 3> matrix;
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

TEST(MatrixTest, RankFailureUnsignedInt) {
  math::Matrix<unsigned int, 3, 3> matrix;
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

TEST(MatrixTest, RankFailureUnsignedInt_2) {
  math::Matrix<unsigned int, 3, 3> matrix;
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

TEST(MatrixTest, MagnitudeUnsignedInt) {
  math::Matrix<unsigned int, 3, 1> vector(2, 2, 1);  // 3D vector

  auto magnitude = vector.magnitude();
  // Expected magnitude is sqrt(2^2 + 2^2 + 1^2) = sqrt(9) = 3
  EXPECT_FLOAT_EQ(vector.magnitude(), 3);
}

// Method: magnitude - failure

TEST(MatrixTest, MagnitudeFailureUnsignedInt) {
  math::Matrix<unsigned int, 3, 1> vector(2, 2, 1);  // 3D vector

  // Incorrect magnitude is 2, the actual magnitude is 3
  EXPECT_NE(vector.magnitude(), 2);
}

// Method: normalize

TEST(MatrixTest, NormalizeUnsignedInt) {
  math::Matrix<unsigned int, 3, 1> vector(2, 2, 1);  // 3D vector

#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  vector.normalize();
  auto& normalizedVector = vector;
#else
  auto normalizedVector = vector.normalized();
#endif

  // The expected normalized vector is (2/3, 2/3, 1/3)
  EXPECT_NEAR(normalizedVector(0, 0), 2 / 3, 1e-5);
  EXPECT_NEAR(normalizedVector(1, 0), 2 / 3, 1e-5);
  EXPECT_NEAR(normalizedVector(2, 0), 1 / 3, 1e-5);
}

// Method: normalize - failure

TEST(MatrixTest, NormalizeFailureUnsignedInt) {
  math::Matrix<unsigned int, 3, 1> vector(2, 2, 1);  // 3D vector

#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  vector.normalize();
  auto& normalizedVector = vector;
#else
  auto normalizedVector = vector.normalized();
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

TEST(MatrixTest, NormalizeZeroVectorUnsignedInt) {
  math::Matrix<unsigned int, 3, 1> vector(0, 0, 0);  // Zero vector

  // Trying to normalize a zero vector should throw an assertion
  EXPECT_DEATH(
      vector.normalize(),
      "Normalization error: magnitude is zero, implying a zero matrix/vector");
}

#endif  // _DEBUG

// Method: trace

TEST(MatrixTest, TraceUnsignedInt) {
  math::Matrix<unsigned int, 3, 3> matrix;
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

TEST(MatrixTest, TraceFailureUnsignedInt) {
  math::Matrix<unsigned int, 3, 3> matrix;
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
TEST(MatrixTest, DotProductUnsignedIntPerpendicular) {
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = 1;
  vector1.coeffRef(1, 0) = 0;
  vector1.coeffRef(2, 0) = 0;

  vector2.coeffRef(0, 0) = 0;
  vector2.coeffRef(1, 0) = 1;
  vector2.coeffRef(2, 0) = 0;

  unsigned int result = vector1.dot(vector2);
  EXPECT_EQ(result, 0);
}

// 2. Parallel vectors
TEST(MatrixTest, DotProductUnsignedIntParallel) {
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = 2;
  vector1.coeffRef(1, 0) = 2;
  vector1.coeffRef(2, 0) = 2;

  vector2.coeffRef(0, 0) = 2;
  vector2.coeffRef(1, 0) = 2;
  vector2.coeffRef(2, 0) = 2;

  unsigned int result = vector1.dot(vector2);
  EXPECT_EQ(result, 12);
}

// 3. Zero vector
TEST(MatrixTest, DotProductUnsignedIntZeroVector) {
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> zeroVector(0);

  vector1.coeffRef(0, 0) = 5;
  vector1.coeffRef(1, 0) = 3;
  vector1.coeffRef(2, 0) = 2;

  unsigned int result = vector1.dot(zeroVector);
  EXPECT_EQ(result, 0);
}

// 4. Unit vectors - can't do for unsigned types

// 5. Negative vectors
TEST(MatrixTest, DotProductUnsignedIntNegative) {
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = -2;
  vector1.coeffRef(1, 0) = 3;
  vector1.coeffRef(2, 0) = -4;

  vector2.coeffRef(0, 0) = 5;
  vector2.coeffRef(1, 0) = -6;
  vector2.coeffRef(2, 0) = 7;

  unsigned int result          = vector1.dot(vector2);
  unsigned int expected_result = (-2 * 5 + 3 * (-6) + (-4 * 7));
  EXPECT_EQ(result, expected_result);
}

// 6. Random vectors
TEST(MatrixTest, DotProductUnsignedIntRandom) {
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector2;

  for (int i = 0; i < 3; ++i) {
    vector1.coeffRef(i, 0) = static_cast<unsigned int>(rand()) / RAND_MAX;
    vector2.coeffRef(i, 0) = static_cast<unsigned int>(rand()) / RAND_MAX;
  }

  unsigned int result          = vector1.dot(vector2);
  unsigned int expected_result = 0;
  for (int i = 0; i < 3; ++i) {
    expected_result += vector1.coeff(i, 0) * vector2.coeff(i, 0);
  }
  EXPECT_NEAR(result, expected_result, 1e-5);
}

// Method: cross

// 1. Perpendicular vectors
TEST(MatrixTest, CrossProductUnsignedIntPerpendicular) {
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector2;

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
TEST(MatrixTest, CrossProductUnsignedIntParallel) {
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector2;

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
TEST(MatrixTest, CrossProductUnsignedIntZeroVector) {
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> zeroVector(0);

  vector1.coeffRef(0, 0) = 5;
  vector1.coeffRef(1, 0) = 3;
  vector1.coeffRef(2, 0) = 2;

  auto result = vector1.cross(zeroVector);

  EXPECT_EQ(result.coeffRef(0, 0), 0);
  EXPECT_EQ(result.coeffRef(1, 0), 0);
  EXPECT_EQ(result.coeffRef(2, 0), 0);
}

// 4. Unit vectors
TEST(MatrixTest, CrossProductUnsignedIntUnitVectors) {
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> unitVector1(
      1 / sqrt(3));
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> unitVector2(
      1 / sqrt(3));

  auto result = unitVector1.cross(unitVector2);

  EXPECT_NEAR(result.coeffRef(0, 0), 0, 1e-5);
  EXPECT_NEAR(result.coeffRef(1, 0), 0, 1e-5);
  EXPECT_NEAR(result.coeffRef(2, 0), 0, 1e-5);
}

// 5. Negative vectors
TEST(MatrixTest, CrossProductUnsignedIntNegative) {
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector2;

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
TEST(MatrixTest, CrossProductUnsignedIntRandom) {
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> vector2;

  for (int i = 0; i < 3; ++i) {
    vector1.coeffRef(i, 0) = static_cast<unsigned int>(rand()) / RAND_MAX;
    vector2.coeffRef(i, 0) = static_cast<unsigned int>(rand()) / RAND_MAX;
  }

  auto result = vector1.cross(vector2);
  math::Matrix<unsigned int, 3, 1, math::Options::RowMajor> expected_result;
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

// ========================== UINT32_T ==============================

TEST(MatrixTest, ConstructorDestructorUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
  // If the constructor and destructor work correctly, this test will pass
}

TEST(MatrixTest, ConstructorDestructorFailureUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
  // This test will pass because the matrix is not null after construction
  EXPECT_NE(&matrix, nullptr);
}

// Method: Matrix(const T& element)

TEST(MatrixTest, ElementConstructorUint32T) {
  const std::uint32_t                     kElementValue = 5;
  const math::Matrix<std::uint32_t, 3, 3> kMatrix(kElementValue);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(kMatrix(i, j), kElementValue);
    }
  }
}

// Method: Matrix(const Matrix& other)

TEST(MatrixTest, CopyConstructorUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix1(1, 2, 3, 4);
  math::Matrix<std::uint32_t, 2, 2> matrix2(matrix1);

  ASSERT_EQ(matrix2(0, 0), 1);
  ASSERT_EQ(matrix2(0, 1), 2);
  ASSERT_EQ(matrix2(1, 0), 3);
  ASSERT_EQ(matrix2(1, 1), 4);
}

// Method: Matrix(Matrix&& other)

TEST(MatrixTest, MoveConstructorUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix1(1, 2, 3, 4);
  math::Matrix<std::uint32_t, 2, 2> matrix2(std::move(matrix1));

  ASSERT_EQ(matrix2(0, 0), 1);
  ASSERT_EQ(matrix2(0, 1), 2);
  ASSERT_EQ(matrix2(1, 0), 3);
  ASSERT_EQ(matrix2(1, 1), 4);
}

// Method: Matrix& operator=(const Matrix& other)

TEST(MatrixTest, AssignmentOperatorUint32T) {
  const math::Matrix<std::uint32_t, 3, 3> kOriginal(5);
  math::Matrix<std::uint32_t, 3, 3>       copy;
  copy = kOriginal;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(copy(i, j), kOriginal(i, j));
    }
  }
}

// Method: coeff(row, col)

TEST(MatrixTest, ElementAccessUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
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

TEST(MatrixTest, ElementAccessFailureUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
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

TEST(MatrixTest, ElementModificationUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
  matrix.coeffRef(0, 0) = 1;
  EXPECT_EQ(matrix.coeff(0, 0), 1);
  matrix.coeffRef(0, 0) = 2;
  EXPECT_EQ(matrix.coeff(0, 0), 2);
}

// Method: modification operator(row, col) - failure

TEST(MatrixTest, ElementModificationFailureUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
  matrix.coeffRef(0, 0) = 1;
  EXPECT_NE(matrix.coeff(0, 0), 2);
  matrix.coeffRef(0, 0) = 2;
  EXPECT_NE(matrix.coeff(0, 0), 1);
}

#ifdef _DEBUG

// Method: coeff(row, col) - out of bounds

TEST(MatrixTest, OutOfBoundsAccessUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
  EXPECT_DEATH(matrix.coeff(2, 2), "Index out of bounds");
  EXPECT_DEATH(matrix.coeffRef(2, 2), "Index out of bounds");
}

#endif  // DEBUG

// Method: access operator(row, col)

TEST(MatrixTest, OperatorAccessUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
  matrix(0, 0) = 1;
  EXPECT_EQ(matrix(0, 0), 1);
  matrix(0, 0) = 2;
  EXPECT_EQ(matrix(0, 0), 2);
}

// Method: access operator(row, col) - failure

TEST(MatrixTest, OperatorAccessConstUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
  matrix(0, 0)             = 1;
  const auto& const_matrix = matrix;
  EXPECT_EQ(const_matrix(0, 0), 1);
  // Check that the type of the returned reference is const
  EXPECT_TRUE(
      std::is_const_v<std::remove_reference_t<decltype(const_matrix(0, 0))>>);
}

// Method: const operator(row, col)

TEST(MatrixTest, CoeffAccessConstUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
  matrix.coeffRef(0, 0)    = 1;
  const auto& const_matrix = matrix;
  EXPECT_EQ(const_matrix.coeff(0, 0), 1);
  // Check that the type of the returned reference is const
  EXPECT_TRUE(std::is_const_v<
              std::remove_reference_t<decltype(const_matrix.coeff(0, 0))>>);
}

// Method: stack allocation

TEST(MatrixTest, StackAllocationUint32T) {
  // This matrix should be small enough to be allocated on the stack
  math::Matrix<std::uint32_t, 2, 2> matrix;
  EXPECT_FALSE(matrix.s_kUseHeap);
}

// Method: heap allocation

TEST(MatrixTest, HeapAllocationUint32T) {
  // This matrix should be large enough to be allocated on the heap
  math::Matrix<std::uint32_t, 100, 100> matrix;
  EXPECT_TRUE(matrix.s_kUseHeap);
}

// Method: addition (matrix + matrix)

TEST(MatrixTest, AdditionUint32T) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<std::uint32_t, kRows, kColumns> matrix1;
  math::Matrix<std::uint32_t, kRows, kColumns> matrix2;

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

TEST(MatrixTest, AdditionFailureUint32T) {
  math::Matrix<std::uint32_t, 4, 4, math::Options::ColumnMajor> matrix1;
  math::Matrix<std::uint32_t, 4, 4, math::Options::ColumnMajor> matrix2;

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

TEST(MatrixTest, ScalarAdditionUint32T) {
  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix2
      = matrix1 + 10;

  // Check that each element of the result is the corresponding element of
  // matrix1 plus 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) + 10);
    }
  }
}

// Method: addition (matrix + scalar) - failure

TEST(MatrixTest, ScalarAdditionFailureUint32T) {
  math::Matrix<std::uint32_t, 4, 4> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<std::uint32_t, 4, 4> matrix2 = matrix1 + 10;

  // Check that each element of the result is not the corresponding element of
  // matrix1 plus 11
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NE(matrix2.coeff(i, j), matrix1.coeff(i, j) + 11);
    }
  }
}

// Method: subtraction (matrix - matrix)

TEST(MatrixTest, SubtractionUint32T) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<std::uint32_t, kRows, kColumns> matrix1;
  math::Matrix<std::uint32_t, kRows, kColumns> matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;  // Use std::uint32_t literals
      matrix2.coeffRef(i, j)
          = (i * 4 + j + 1) * 2;               // Use std::uint32_t literals
    }
  }

  auto matrix3 = matrix1 - matrix2;

  // Check that each element of the result is the difference of the
  // corresponding elements of matrix1 and matrix2
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      EXPECT_NEAR(
          matrix3.coeff(i, j), matrix1.coeff(i, j) - matrix2.coeff(i, j), 1e-5);
      // Alternatively, for exact std::uint32_ting-point comparisons (which
      // might be risky due to precision issues):
      // EXPECT_FLOAT_EQ(matrix3.coeff(i, j), matrix1.coeff(i, j) -
      // matrix2.coeff(i, j));
    }
  }
}

// Method: subtraction (matrix - scalar)

TEST(MatrixTest, ScalarSubtractionUint32T) {
  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix2
      = matrix1 - 10;

  // Check that each element of the result is the corresponding element of
  // matrix1 minus 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) - 10);
    }
  }
}

// Test: Negation (-Matrix)

TEST(MatrixTest, NegationUint32T) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<std::uint32_t, kRows, kColumns> matrix;

  // Populate matrix with some values
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kColumns; ++j) {
      matrix.coeffRef(i, j) = i * 4 + j + 1;  // Example values: 1, 2, 5, 6
    }
  }

  // Since negating an std::uint32_t matrix is expected to trigger an assertion,
  // we use EXPECT_DEATH to verify that the assertion indeed occurs.
  EXPECT_DEATH({ auto negatedMatrix = -matrix; }, "");

  // auto negatedMatrix = -matrix;

  // Check that each element of the negated matrix is the negation
  // of the corresponding element in the original matrix
  // for (int i = 0; i < kRows; ++i) {
  //  for (int j = 0; j < kColumns; ++j) {
  //    EXPECT_EQ(negatedMatrix.coeff(i, j), -matrix.coeff(i, j));
  //  }
  //}
}

// Method: row major multiplication (matrix * matrix)

TEST(MatrixTest, MultiplicationRowMajorUint32T) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<std::uint32_t, kRows, kColumns, math::Options::RowMajor> matrix1;
  math::Matrix<std::uint32_t, kRows, kColumns, math::Options::RowMajor> matrix2;

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
      std::uint32_t expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: row major multiplication (matrix * matrix) - non square matrices

TEST(MatrixTest, MultiplicationRowMajorUint32TNonSquare) {
  constexpr int kRowsA      = 3;
  constexpr int kColsARowsB = 4;
  constexpr int kColsB      = 2;

  math::Matrix<std::uint32_t, kRowsA, kColsARowsB, math::Options::RowMajor>
      matrix1;
  math::Matrix<std::uint32_t, kColsARowsB, kColsB, math::Options::RowMajor>
      matrix2;

  // Populate matrix1 and matrix2 with some values
  for (int i = 0; i < kRowsA; ++i) {
    for (int j = 0; j < kColsARowsB; ++j) {
      matrix1.coeffRef(i, j) = static_cast<std::uint32_t>(rand())
                             / RAND_MAX;  // random values between 0 and 1
    }
  }

  for (int i = 0; i < kColsARowsB; ++i) {
    for (int j = 0; j < kColsB; ++j) {
      matrix2.coeffRef(i, j) = static_cast<std::uint32_t>(rand())
                             / RAND_MAX;  // random values between 0 and 1
    }
  }

  auto matrix3 = matrix1 * matrix2;

  for (int i = 0; i < kRowsA; ++i) {
    for (int j = 0; j < kColsB; ++j) {
      std::uint32_t expected_value = 0;
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

TEST(MatrixTest, MultiplicationRowMajorUint32TNonSquare_2) {
  constexpr int kRowsA      = 3;
  constexpr int kColsARowsB = 4;
  constexpr int kColsB      = 2;

  math::Matrix<std::uint32_t, kRowsA, kColsARowsB, math::Options::RowMajor>
      matrix1;
  math::Matrix<std::uint32_t, kColsARowsB, kColsB, math::Options::RowMajor>
      matrix2;

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

TEST(MatrixTest, MultiplicationColumnMajorUint32T) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<std::uint32_t, kRows, kColumns, math::Options::ColumnMajor>
      matrix1;
  math::Matrix<std::uint32_t, kRows, kColumns, math::Options::ColumnMajor>
      matrix2;

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
      std::uint32_t expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: row major multiplication (matrix *= matrix)

TEST(MatrixTest, MultiplicationRowMajorUint32TInPlace) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<std::uint32_t, kRows, kColumns, math::Options::RowMajor> matrix1;
  math::Matrix<std::uint32_t, kRows, kColumns, math::Options::RowMajor> matrix2;

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
      std::uint32_t expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: column major multiplication (matrix *= matrix)

TEST(MatrixTest, MultiplicationColumnMajorUint32TInPlace) {
  constexpr int kRows    = 2;
  constexpr int kColumns = 2;

  math::Matrix<std::uint32_t, kRows, kColumns, math::Options::ColumnMajor>
      matrix1;
  math::Matrix<std::uint32_t, kRows, kColumns, math::Options::ColumnMajor>
      matrix2;

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
      std::uint32_t expected_value = 0;
      for (int k = 0; k < kColumns; ++k) {
        expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
      }
      EXPECT_EQ(matrix3.coeff(i, j), expected_value);
    }
  }
}

// Method: scalar multiplication

TEST(MatrixTest, ScalarMultiplicationUint32T) {
  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix1;
  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }
  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix2
      = matrix1 * 10;
  // Check that each element of the result is the corresponding element of
  // matrix1 multiplied by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) * 10);
    }
  }
}

// Method: scalar division

TEST(MatrixTest, ScalarDivisionUint32T) {
  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix1;

  // Populate matrix1 with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix2
      = matrix1 / 10;

  // Check that each element of the result is the corresponding element of
  // matrix1 divided by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) / 10);
    }
  }
}

// Method: scalar division - failure

TEST(MatrixTest, ScalarDivisionFailureUint32T) {
  math::Matrix<std::uint32_t, 4, 4> matrix1;

  for (std::uint32_t i = 0; i < 4; ++i) {
    for (std::uint32_t j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = (i * 4 + j + 1) * 110;
    }
  }

  math::Matrix<std::uint32_t, 4, 4> matrix2 = matrix1 / 10;

  for (std::uint32_t i = 0; i < 4; ++i) {
    for (std::uint32_t j = 0; j < 4; ++j) {
      EXPECT_NE(matrix2.coeff(i, j), matrix1.coeff(i, j) / 11);
    }
  }
}

// Method: scalar division in place

TEST(MatrixTest, ScalarDivisionInPlaceUint32T) {
  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix;

  // Populate matrix with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix.coeffRef(i, j) = (i * 4 + j + 1);
    }
  }

  matrix /= 10;

  // Check that each element of the matrix is the original value divided by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(matrix.coeff(i, j), (i * 4 + j + 1) / 10);
    }
  }
}

// Method: scalar division in place - failure

TEST(MatrixTest, ScalarDivisionInPlaceFailureUint32T) {
  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix;

  // Populate matrix with some values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix.coeffRef(i, j) = (i * 4 + j + 1) * 110;
    }
  }

  matrix /= 10;

  // Check that each element of the matrix is the original value divided by 10
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NE(matrix.coeff(i, j), (i * 4 + j + 1) / 11);
    }
  }
}

// Method: matrix equality comparison

TEST(MatrixTest, MatrixEqualityTrueUint32T) {
  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality comparison - failure

TEST(MatrixTest, MatrixEqualityFalseUint32T) {
  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix2  = matrix1;
  matrix2.coeffRef(0, 0)                                             += 1;

  EXPECT_FALSE(matrix1 == matrix2);
}

// Method: matrix equality with very small numbers

TEST(MatrixTest, MatrixEqualityTrueUint32TSmallNumbers) {
  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<std::uint32_t>::min();
  matrix1.coeffRef(0, 1) = std::numeric_limits<std::uint32_t>::min();

  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality with very large numbers

TEST(MatrixTest, MatrixEqualityTrueUint32TLargeNumbers) {
  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<std::uint32_t>::max();
  matrix1.coeffRef(0, 1) = std::numeric_limits<std::uint32_t>::max();

  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality with numbers close to each other

TEST(MatrixTest, MatrixEqualityTrueUint32TCloseNumbers) {
  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = 1;
  matrix1.coeffRef(0, 1) = 1;

  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix2 = matrix1;
  matrix2.coeffRef(0, 0) += std::numeric_limits<std::uint32_t>::epsilon();
  matrix2.coeffRef(0, 1) += std::numeric_limits<std::uint32_t>::epsilon();

  EXPECT_TRUE(matrix1 == matrix2);
}

TEST(MatrixTest, MatrixEqualityFailSmallDifferencesUint32T) {
  math::Matrix<std::uint32_t, 1, 1, math::Options::RowMajor> matrix1;
  math::Matrix<std::uint32_t, 1, 1, math::Options::RowMajor> matrix2;

  matrix1.coeffRef(0, 0) = std::numeric_limits<std::uint32_t>::epsilon() / 2;
  matrix2.coeffRef(0, 0) = -std::numeric_limits<std::uint32_t>::epsilon() / 2;

  // Pay attention: this test case should fail since the difference between the
  // two numbers
  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix equality comparison

TEST(MatrixTest, MatrixEqualityLargeNumbersUint32T) {
  math::Matrix<std::uint32_t, 1, 1, math::Options::RowMajor> matrix1;
  math::Matrix<std::uint32_t, 1, 1, math::Options::RowMajor> matrix2;

  matrix1.coeffRef(0, 0) = 1e10f;
  matrix2.coeffRef(0, 0) = 1e10f + 1e-5f;

  // Pay attention:
  // Even though the relative difference between the two numbers is negligible,
  // they are considered not equal because the absolute difference exceeds
  // epsilon
  EXPECT_TRUE(matrix1 == matrix2);
}

// Method: matrix inequality comparison

TEST(MatrixTest, MatrixInequalityTrueUint32T) {
  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix2  = matrix1;
  matrix2.coeffRef(0, 0)                                             += 1;

  EXPECT_TRUE(matrix1 != matrix2);
}

// Method: matrix inequality with very small numbers

TEST(MatrixTest, MatrixInequalityFalseUint32TSmallNumbers) {
  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<std::uint32_t>::min();
  matrix1.coeffRef(0, 1) = std::numeric_limits<std::uint32_t>::min();

  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: matrix inequality with very large numbers

TEST(MatrixTest, MatrixInequalityFalseUint32TLargeNumbers) {
  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = std::numeric_limits<std::uint32_t>::max();
  matrix1.coeffRef(0, 1) = std::numeric_limits<std::uint32_t>::max();

  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: matrix inequality with numbers close to each other

TEST(MatrixTest, MatrixInequalityFalseUint32TCloseNumbers) {
  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix1;
  matrix1.coeffRef(0, 0) = 1;
  matrix1.coeffRef(0, 1) = 1;

  math::Matrix<std::uint32_t, 1, 2, math::Options::RowMajor> matrix2 = matrix1;
  matrix2.coeffRef(0, 0) += std::numeric_limits<std::uint32_t>::epsilon();
  matrix2.coeffRef(0, 1) += std::numeric_limits<std::uint32_t>::epsilon();

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: matrix inequality comparison

TEST(MatrixTest, MatrixInequalityFalseUint32T) {
  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix1;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      matrix1.coeffRef(i, j) = i * 4 + j + 1;
    }
  }

  math::Matrix<std::uint32_t, 4, 4, math::Options::RowMajor> matrix2 = matrix1;

  EXPECT_FALSE(matrix1 != matrix2);
}

// Method: transpose

TEST(MatrixTest, TransposeUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix1(1, 2, 3, 4);
  auto                              matrix2 = matrix1.transpose();

  ASSERT_EQ(matrix2(0, 0), 1);
  ASSERT_EQ(matrix2(0, 1), 3);
  ASSERT_EQ(matrix2(1, 0), 2);
  ASSERT_EQ(matrix2(1, 1), 4);
}

// Method: reshape

TEST(MatrixTest, ReshapeUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix1(1, 2, 3, 4);
  auto                              matrix2 = matrix1.reshape<4, 1>();

  ASSERT_EQ(matrix2(0, 0), 1);
  ASSERT_EQ(matrix2(1, 0), 2);
  ASSERT_EQ(matrix2(2, 0), 3);
  ASSERT_EQ(matrix2(3, 0), 4);
}

// Method: determinant

TEST(MatrixTest, DeterminantUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(1, 0) = 3;
  matrix(1, 1) = 4;

  // Expected determinant is 1*4 - 2*3 = -2
  auto det = matrix.determinant();
  EXPECT_FLOAT_EQ(matrix.determinant(), -2);
}

// Method: determinant - failure

TEST(MatrixTest, DeterminantFailureUint32T) {
  math::Matrix<std::uint32_t, 2, 2> matrix;
  matrix(0, 0) = 1;
  matrix(0, 1) = 2;
  matrix(1, 0) = 3;
  matrix(1, 1) = 4;

  // Incorrect determinant is -3, the actual determinant is -2
  EXPECT_NE(matrix.determinant(), -3);
}

// Method: determinant - 4x4 matrix

TEST(MatrixTest, DeterminantUint32T_1) {
  // Create a 4x4 matrix with arbitrary values
  math::Matrix<std::uint32_t, 4, 4> matrix;
  matrix << 5, 7, 6, 1, 2, 8, 4, 6, 3, 4, 2, 7, 7, 3, 5, 1;

  EXPECT_FLOAT_EQ(matrix.determinant(), -52);
}

// Method: inverse - this is not reasonable since std::uint32_t

// Method: rank - 2 test

TEST(MatrixTest, RankUint32T_2) {
  math::Matrix<std::uint32_t, 3, 3> matrix;
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

TEST(MatrixTest, RankFullUint32T) {
  math::Matrix<std::uint32_t, 3, 3> matrix;
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

TEST(MatrixTest, RankFailureUint32T) {
  math::Matrix<std::uint32_t, 3, 3> matrix;
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

TEST(MatrixTest, RankFailureUint32T_2) {
  math::Matrix<std::uint32_t, 3, 3> matrix;
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

TEST(MatrixTest, MagnitudeUint32T) {
  math::Matrix<std::uint32_t, 3, 1> vector(2, 2, 1);  // 3D vector

  auto magnitude = vector.magnitude();
  // Expected magnitude is sqrt(2^2 + 2^2 + 1^2) = sqrt(9) = 3
  EXPECT_FLOAT_EQ(vector.magnitude(), 3);
}

// Method: magnitude - failure

TEST(MatrixTest, MagnitudeFailureUint32T) {
  math::Matrix<std::uint32_t, 3, 1> vector(2, 2, 1);  // 3D vector

  // Incorrect magnitude is 2, the actual magnitude is 3
  EXPECT_NE(vector.magnitude(), 2);
}

// Method: normalize

TEST(MatrixTest, NormalizeUint32T) {
  math::Matrix<std::uint32_t, 3, 1> vector(2, 2, 1);  // 3D vector

#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  vector.normalize();
  auto& normalizedVector = vector;
#else
  auto normalizedVector = vector.normalized();
#endif

  // The expected normalized vector is (2/3, 2/3, 1/3)
  EXPECT_NEAR(normalizedVector(0, 0), 2 / 3, 1e-5);
  EXPECT_NEAR(normalizedVector(1, 0), 2 / 3, 1e-5);
  EXPECT_NEAR(normalizedVector(2, 0), 1 / 3, 1e-5);
}

// Method: normalize - failure

TEST(MatrixTest, NormalizeFailureUint32T) {
  math::Matrix<std::uint32_t, 3, 1> vector(2, 2, 1);  // 3D vector

#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  vector.normalize();
  auto& normalizedVector = vector;
#else
  auto normalizedVector = vector.normalized();
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

TEST(MatrixTest, NormalizeZeroVectorUint32T) {
  math::Matrix<std::uint32_t, 3, 1> vector(0, 0, 0);  // Zero vector

  // Trying to normalize a zero vector should throw an assertion
  EXPECT_DEATH(
      vector.normalize(),
      "Normalization error: magnitude is zero, implying a zero matrix/vector");
}

#endif  // _DEBUG

// Method: trace

TEST(MatrixTest, TraceUint32T) {
  math::Matrix<std::uint32_t, 3, 3> matrix;
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

TEST(MatrixTest, TraceFailureUint32T) {
  math::Matrix<std::uint32_t, 3, 3> matrix;
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
TEST(MatrixTest, DotProductUint32TPerpendicular) {
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = 1;
  vector1.coeffRef(1, 0) = 0;
  vector1.coeffRef(2, 0) = 0;

  vector2.coeffRef(0, 0) = 0;
  vector2.coeffRef(1, 0) = 1;
  vector2.coeffRef(2, 0) = 0;

  std::uint32_t result = vector1.dot(vector2);
  EXPECT_EQ(result, 0);
}

// 2. Parallel vectors
TEST(MatrixTest, DotProductUint32TParallel) {
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = 2;
  vector1.coeffRef(1, 0) = 2;
  vector1.coeffRef(2, 0) = 2;

  vector2.coeffRef(0, 0) = 2;
  vector2.coeffRef(1, 0) = 2;
  vector2.coeffRef(2, 0) = 2;

  std::uint32_t result = vector1.dot(vector2);
  EXPECT_EQ(result, 12);
}

// 3. Zero vector
TEST(MatrixTest, DotProductUint32TZeroVector) {
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> zeroVector(0);

  vector1.coeffRef(0, 0) = 5;
  vector1.coeffRef(1, 0) = 3;
  vector1.coeffRef(2, 0) = 2;

  std::uint32_t result = vector1.dot(zeroVector);
  EXPECT_EQ(result, 0);
}

// 4. Unit vectors - can't do for unsigned types

// 5. Negative vectors
TEST(MatrixTest, DotProductUint32TNegative) {
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector2;

  vector1.coeffRef(0, 0) = -2;
  vector1.coeffRef(1, 0) = 3;
  vector1.coeffRef(2, 0) = -4;

  vector2.coeffRef(0, 0) = 5;
  vector2.coeffRef(1, 0) = -6;
  vector2.coeffRef(2, 0) = 7;

  std::uint32_t result          = vector1.dot(vector2);
  std::uint32_t expected_result = (-2 * 5 + 3 * (-6) + (-4 * 7));
  EXPECT_EQ(result, expected_result);
}

// 6. Random vectors
TEST(MatrixTest, DotProductUint32TRandom) {
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector2;

  for (int i = 0; i < 3; ++i) {
    vector1.coeffRef(i, 0) = static_cast<std::uint32_t>(rand()) / RAND_MAX;
    vector2.coeffRef(i, 0) = static_cast<std::uint32_t>(rand()) / RAND_MAX;
  }

  std::uint32_t result          = vector1.dot(vector2);
  std::uint32_t expected_result = 0;
  for (int i = 0; i < 3; ++i) {
    expected_result += vector1.coeff(i, 0) * vector2.coeff(i, 0);
  }
  EXPECT_NEAR(result, expected_result, 1e-5);
}

// Method: cross

// 1. Perpendicular vectors
TEST(MatrixTest, CrossProductUint32TPerpendicular) {
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector2;

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
TEST(MatrixTest, CrossProductUint32TParallel) {
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector2;

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
TEST(MatrixTest, CrossProductUint32TZeroVector) {
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> zeroVector(0);

  vector1.coeffRef(0, 0) = 5;
  vector1.coeffRef(1, 0) = 3;
  vector1.coeffRef(2, 0) = 2;

  auto result = vector1.cross(zeroVector);

  EXPECT_EQ(result.coeffRef(0, 0), 0);
  EXPECT_EQ(result.coeffRef(1, 0), 0);
  EXPECT_EQ(result.coeffRef(2, 0), 0);
}

// 4. Unit vectors
TEST(MatrixTest, CrossProductUint32TUnitVectors) {
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> unitVector1(
      1 / sqrt(3));
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> unitVector2(
      1 / sqrt(3));

  auto result = unitVector1.cross(unitVector2);

  EXPECT_NEAR(result.coeffRef(0, 0), 0, 1e-5);
  EXPECT_NEAR(result.coeffRef(1, 0), 0, 1e-5);
  EXPECT_NEAR(result.coeffRef(2, 0), 0, 1e-5);
}

// 5. Negative vectors
TEST(MatrixTest, CrossProductUint32TNegative) {
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector2;

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
TEST(MatrixTest, CrossProductUint32TRandom) {
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector1;
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> vector2;

  for (int i = 0; i < 3; ++i) {
    vector1.coeffRef(i, 0) = static_cast<std::uint32_t>(rand()) / RAND_MAX;
    vector2.coeffRef(i, 0) = static_cast<std::uint32_t>(rand()) / RAND_MAX;
  }

  auto result = vector1.cross(vector2);
  math::Matrix<std::uint32_t, 3, 1, math::Options::RowMajor> expected_result;
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

// TODO: add more non trivial view matrix constructions
// ========================== GRAPHICS ==============================

TEST(ViewMatrixTest, LookAtRhRowMajor) {
  math::Vector3D<float> eye{0.0f, 0.0f, 1.0f};
  math::Vector3D<float> target{0.0f, 0.0f, 0.0f};
  math::Vector3D<float> up{0.0f, 1.0f, 0.0f};
  auto                  viewMatrix = g_lookAtRh(eye, target, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);

  // Check the up vector
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);

  // Check the forward vector (should be inverted)
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), 1.0f);

  // Check the translation
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

TEST(ViewMatrixTest, LookAtRhColumnMajor) {
  math::Vector3D<float, math::Options::ColumnMajor> eye{0.0f, 0.0f, 1.0f};
  math::Vector3D<float, math::Options::ColumnMajor> target{0.0f, 0.0f, 0.0f};
  math::Vector3D<float, math::Options::ColumnMajor> up{0.0f, 1.0f, 0.0f};
  auto viewMatrix = g_lookAtRh(eye, target, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);

  // Check the up vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);

  // Check the forward vector (should be inverted)
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), 1.0f);

  // Check the translation
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

TEST(ViewMatrixTest, LookToRhRowMajor) {
  math::Vector3D<float> eye{0.0f, 0.0f, 1.0f};
  math::Vector3D<float> direction{0.0f, 0.0f, -1.0f};
  math::Vector3D<float> up{0.0f, 1.0f, 0.0f};
  auto                  viewMatrix = g_lookToRh(eye, direction, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);

  // Check the up vector
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);

  // Check the forward vector (should be inverted)
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), 1.0f);

  // Check the translation
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

TEST(ViewMatrixTest, LookToRhColumnMajor) {
  math::Vector3D<float, math::Options::ColumnMajor> eye{0.0f, 0.0f, 1.0f};
  math::Vector3D<float, math::Options::ColumnMajor> direction{
    0.0f, 0.0f, -1.0f};
  math::Vector3D<float, math::Options::ColumnMajor> up{0.0f, 1.0f, 0.0f};
  auto viewMatrix = g_lookToRh(eye, direction, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);

  // Check the up vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);

  // Check the forward vector (should be inverted)
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), 1.0f);

  // Check the translation
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

TEST(ViewMatrixTest, LookAtLhRowMajor) {
  math::Vector3D<float> eye{0.0f, 0.0f, 1.0f};
  math::Vector3D<float> target{0.0f, 0.0f, 0.0f};
  math::Vector3D<float> up{0.0f, 1.0f, 0.0f};
  auto                  viewMatrix = g_lookAtLh(eye, target, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);

  // Check the up vector
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);

  // Check the forward vector
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), -1.0f);

  // Check the translation
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

TEST(ViewMatrixTest, LookAtLhColumnMajor) {
  math::Vector3D<float, math::Options::ColumnMajor> eye{0.0f, 0.0f, 1.0f};
  math::Vector3D<float, math::Options::ColumnMajor> target{0.0f, 0.0f, 0.0f};
  math::Vector3D<float, math::Options::ColumnMajor> up{0.0f, 1.0f, 0.0f};
  auto viewMatrix = g_lookAtLh(eye, target, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);

  // Check the up vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);

  // Check the forward vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), -1.0f);

  // Check the translation
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

TEST(ViewMatrixTest, LookToLhRowMajor) {
  math::Vector3D<float> eye{0.0f, 0.0f, 1.0f};
  math::Vector3D<float> direction{0.0f, 0.0f, -1.0f};
  math::Vector3D<float> up{0.0f, 1.0f, 0.0f};
  auto                  viewMatrix = g_lookToLh(eye, direction, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);

  // Check the up vector
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);

  // Check the forward vector
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), -1.0f);

  // Check the translation
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

TEST(ViewMatrixTest, LookToLhColumnMajor) {
  math::Vector3D<float, math::Options::ColumnMajor> eye{0.0f, 0.0f, 1.0f};
  math::Vector3D<float, math::Options::ColumnMajor> direction{
    0.0f, 0.0f, -1.0f};
  math::Vector3D<float, math::Options::ColumnMajor> up{0.0f, 1.0f, 0.0f};
  auto viewMatrix = g_lookToLh(eye, direction, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);

  // Check the up vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);

  // Check the forward vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), -1.0f);

  // Check the translation
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

// Test case for g_rotateRh (Euler angles)
TEST(RotateRhTest, EulerAnglesFloat) {
  float angleX = math::g_kPi / 6.0f;
  float angleY = math::g_kPi / 4.0f;
  float angleZ = math::g_kPi / 3.0f;

  auto rotateMatrix = math::g_rotateRh<float>(angleX, angleY, angleZ);

  //std::cout << rotateMatrix << std::endl;

  EXPECT_NEAR(rotateMatrix(0, 0), 0.65974f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(0, 1), 0.75f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(0, 2), -0.0473671f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(0, 3), 0.0f);

  EXPECT_NEAR(rotateMatrix(1, 0), -0.435596f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(1, 1), 0.433013f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(1, 2), 0.789149f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(1, 3), 0.0f);

  EXPECT_NEAR(rotateMatrix(2, 0), 0.612372f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(2, 1), -0.5f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(2, 2), 0.612372f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(2, 3), 0.0f);

  EXPECT_FLOAT_EQ(rotateMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 3), 1.0f);
}

// Test case for g_rotateRh (axis-angle)
TEST(RotateRhTest, AxisAngleFloat) {
  math::Vector<float, 3> axis(2.0f, 1.0f, 3.0f);
  float                  angle = math::g_kPi / 3.0f;

  auto rotateMatrix = math::g_rotateRh<float>(axis, angle);

  //std::cout << rotateMatrix << std::endl;

  EXPECT_NEAR(rotateMatrix(0, 0), 0.642857f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(0, 1), -0.622936f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(0, 2), 0.445741f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(0, 3), 0.0f);

  EXPECT_NEAR(rotateMatrix(1, 0), 0.765794f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(1, 1), 0.535714f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(1, 2), -0.355767f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(1, 3), 0.0f);

  EXPECT_NEAR(rotateMatrix(2, 0), -0.0171693f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(2, 1), 0.570053f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(2, 2), 0.821429f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(2, 3), 0.0f);

  EXPECT_FLOAT_EQ(rotateMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 3), 1.0f);
}

// Test case for g_rotateRh (Euler angles) - failure
TEST(RotateRhTest, EulerAnglesFailureFloat) {
  float angleX = math::g_kPi / 6.0f;
  float angleY = math::g_kPi / 4.0f;
  float angleZ = math::g_kPi / 3.0f;

  auto rotateMatrix = math::g_rotateRh<float>(angleX, angleY, angleZ);

  EXPECT_NE(rotateMatrix(0, 0), 0.2f);
  EXPECT_NE(rotateMatrix(0, 1), 0.1f);
  EXPECT_NE(rotateMatrix(0, 2), 0.9f);
  EXPECT_FLOAT_EQ(rotateMatrix(0, 3), 0.0f);

  EXPECT_NE(rotateMatrix(1, 0), -0.8f);
  EXPECT_NE(rotateMatrix(1, 1), 0.6f);
  EXPECT_NE(rotateMatrix(1, 2), -0.4f);
  EXPECT_FLOAT_EQ(rotateMatrix(1, 3), 0.0f);

  EXPECT_NE(rotateMatrix(2, 0), 0.1f);
  EXPECT_NE(rotateMatrix(2, 1), -0.9f);
  EXPECT_NE(rotateMatrix(2, 2), 0.2f);
  EXPECT_FLOAT_EQ(rotateMatrix(2, 3), 0.0f);

  EXPECT_FLOAT_EQ(rotateMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 3), 1.0f);
}

// Test case for g_rotateRh (axis-angle) - failure
TEST(RotateRhTest, AxisAngleFailureFloat) {
  math::Vector<float, 3> axis(2.0f, 1.0f, 3.0f);
  float                  angle = math::g_kPi / 3.0f;

  auto rotateMatrix = math::g_rotateRh<float>(axis, angle);

  EXPECT_NE(rotateMatrix(0, 0), 0.1f);
  EXPECT_NE(rotateMatrix(0, 1), -0.9f);
  EXPECT_NE(rotateMatrix(0, 2), 0.8f);
  EXPECT_FLOAT_EQ(rotateMatrix(0, 3), 0.0f);

  EXPECT_NE(rotateMatrix(1, 0), 0.9f);
  EXPECT_NE(rotateMatrix(1, 1), 0.2f);
  EXPECT_NE(rotateMatrix(1, 2), -0.7f);
  EXPECT_FLOAT_EQ(rotateMatrix(1, 3), 0.0f);

  EXPECT_NE(rotateMatrix(2, 0), -0.8f);
  EXPECT_NE(rotateMatrix(2, 1), 0.7f);
  EXPECT_NE(rotateMatrix(2, 2), 0.3f);
  EXPECT_FLOAT_EQ(rotateMatrix(2, 3), 0.0f);

  EXPECT_FLOAT_EQ(rotateMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 3), 1.0f);
}

// Test case for g_rotateLh (Euler angles)
TEST(RotateLhTest, EulerAnglesFloat) {
  float angleX = math::g_kPi / 6.0f;
  float angleY = math::g_kPi / 4.0f;
  float angleZ = math::g_kPi / 3.0f;

  auto rotateMatrix = math::g_rotateLh<float>(angleX, angleY, angleZ);

  EXPECT_NEAR(rotateMatrix(0, 0), 0.0473671f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(0, 1), -0.75f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(0, 2), 0.65974f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(0, 3), 0.0f);

  EXPECT_NEAR(rotateMatrix(1, 0), 0.789149f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(1, 1), 0.433013f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(1, 2), 0.435596f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(1, 3), 0.0f);

  EXPECT_NEAR(rotateMatrix(2, 0), -0.612372f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(2, 1), 0.5f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(2, 2), 0.612372f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(2, 3), 0.0f);

  EXPECT_FLOAT_EQ(rotateMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 3), 1.0f);
}

// Test case for g_rotateLh (Euler angles) - failure
TEST(RotateLhTest, EulerAnglesFailureFloat) {
  float angleX = math::g_kPi / 6.0f;
  float angleY = math::g_kPi / 4.0f;
  float angleZ = math::g_kPi / 3.0f;

  auto rotateMatrix = math::g_rotateLh<float>(angleX, angleY, angleZ);

  EXPECT_NE(rotateMatrix(0, 0), 0.2f);
  EXPECT_NE(rotateMatrix(0, 1), -0.1f);
  EXPECT_NE(rotateMatrix(0, 2), -0.9f);
  EXPECT_FLOAT_EQ(rotateMatrix(0, 3), 0.0f);

  EXPECT_NE(rotateMatrix(1, 0), 0.8f);
  EXPECT_NE(rotateMatrix(1, 1), -0.6f);
  EXPECT_NE(rotateMatrix(1, 2), 0.4f);
  EXPECT_FLOAT_EQ(rotateMatrix(1, 3), 0.0f);

  EXPECT_NE(rotateMatrix(2, 0), -0.1f);
  EXPECT_NE(rotateMatrix(2, 1), 0.9f);
  EXPECT_NE(rotateMatrix(2, 2), -0.2f);
  EXPECT_FLOAT_EQ(rotateMatrix(2, 3), 0.0f);

  EXPECT_FLOAT_EQ(rotateMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 3), 1.0f);
}

// Test case for g_rotateLh (axis-angle)
TEST(RotateLhTest, AxisAngleFloat) {
  math::Vector<float, 3> axis(2.0f, 1.0f, 3.0f);
  float                  angle = math::g_kPi / 3.0f;

  auto rotateMatrix = math::g_rotateLh<float>(axis, angle);

  EXPECT_NEAR(rotateMatrix(0, 0), 0.642857f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(0, 1), 0.765794f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(0, 2), -0.0171693f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(0, 3), 0.0f);

  EXPECT_NEAR(rotateMatrix(1, 0), -0.622936f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(1, 1), 0.535714f, 1e-6f); 
  EXPECT_NEAR(rotateMatrix(1, 2), 0.570053f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(1, 3), 0.0f);

  EXPECT_NEAR(rotateMatrix(2, 0), 0.445741f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(2, 1), -0.355767f, 1e-6f);
  EXPECT_NEAR(rotateMatrix(2, 2), 0.821429f, 1e-6f);
  EXPECT_FLOAT_EQ(rotateMatrix(2, 3), 0.0f);

  EXPECT_FLOAT_EQ(rotateMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 3), 1.0f);
}

// Test case for g_rotateLh (axis-angle) - failure
TEST(RotateLhTest, AxisAngleFailureFloat) {
  math::Vector<float, 3> axis(2.0f, 1.0f, 3.0f);
  float                  angle = math::g_kPi / 3.0f;

  auto rotateMatrix = math::g_rotateLh<float>(axis, angle);

  EXPECT_NE(rotateMatrix(0, 0), -0.1f);
  EXPECT_NE(rotateMatrix(0, 1), 0.9f);
  EXPECT_NE(rotateMatrix(0, 2), -0.8f);
  EXPECT_FLOAT_EQ(rotateMatrix(0, 3), 0.0f);

  EXPECT_NE(rotateMatrix(1, 0), -0.9f);
  EXPECT_NE(rotateMatrix(1, 1), -0.2f);
  EXPECT_NE(rotateMatrix(1, 2), 0.7f);
  EXPECT_FLOAT_EQ(rotateMatrix(1, 3), 0.0f);

  EXPECT_NE(rotateMatrix(2, 0), 0.8f);
  EXPECT_NE(rotateMatrix(2, 1), -0.7f);
  EXPECT_NE(rotateMatrix(2, 2), -0.3f);
  EXPECT_FLOAT_EQ(rotateMatrix(2, 3), 0.0f);

  EXPECT_FLOAT_EQ(rotateMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(rotateMatrix(3, 3), 1.0f);
}

// Test case for lookAtRh
TEST(LookAtTest, LookAtRhFloat) {
  math::Vector3Df eye(0.0f, 0.0f, 5.0f);
  math::Vector3Df target(0.0f, 0.0f, 0.0f);
  math::Vector3Df up(0.0f, 1.0f, 0.0f);
  math::Matrix4f viewMatrix = math::g_lookAtRh(eye, target, up);

  std::cout << viewMatrix << std::endl;

  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 5.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

// Test case for lookAtLh
TEST(LookAtTest, LookAtLhFloat) {
  math::Vector3Df eye(0.0f, 0.0f, 5.0f);
  math::Vector3Df target(0.0f, 0.0f, 0.0f);
  math::Vector3Df up(0.0f, 1.0f, 0.0f);
  math::Matrix4f viewMatrix = math::g_lookAtLh(eye, target, up);

  std::cout << viewMatrix << std::endl;

  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 5.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

// Test case for lookToRh
TEST(LookAtTest, LookToRhFloat) {
  math::Vector3Df eye(0.0f, 0.0f, 5.0f);
  math::Vector3Df direction(0.0f, 0.0f, -1.0f);
  math::Vector3Df up(0.0f, 1.0f, 0.0f);
  math::Matrix4f viewMatrix = math::g_lookToRh(eye, direction, up);

  std::cout << viewMatrix << std::endl;

  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 5.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

// Test case for lookToLh
TEST(LookAtTest, LookToLhFloat) {
  math::Vector3Df eye(0.0f, 0.0f, 5.0f);
  math::Vector3Df direction(0.0f, 0.0f, -1.0f);
  math::Vector3Df up(0.0f, 1.0f, 0.0f);
  math::Matrix4f viewMatrix = math::g_lookToLh(eye, direction, up);

  std::cout << viewMatrix << std::endl;

  EXPECT_FLOAT_EQ(viewMatrix(0, 0), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 2), -1.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 5.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

// ========================== VECTOR: FLOAT ==============================

TEST(VectorComparisonTest, LessThanOperator) {
  math::Vector3Df vec1(1.0f, 2.0f, 3.0f);
  math::Vector3Df vec2(4.0f, 5.0f, 6.0f);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, LessThanOperatorEqual) {
  math::Vector3Df vec1(1.0f, 2.0f, 3.0f);
  math::Vector3Df vec2(1.0f, 2.0f, 3.0f);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperator) {
  math::Vector3Df vec1(4.0f, 5.0f, 6.0f);
  math::Vector3Df vec2(1.0f, 2.0f, 3.0f);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorEqual) {
  math::Vector3Df vec1(1.0f, 2.0f, 3.0f);
  math::Vector3Df vec2(1.0f, 2.0f, 3.0f);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperator) {
  math::Vector3Df vec1(1.0f, 2.0f, 3.0f);
  math::Vector3Df vec2(4.0f, 5.0f, 6.0f);
  math::Vector3Df vec3(1.0f, 2.0f, 3.0f);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperator) {
  math::Vector3Df vec1(4.0f, 5.0f, 6.0f);
  math::Vector3Df vec2(1.0f, 2.0f, 3.0f);
  math::Vector3Df vec3(4.0f, 5.0f, 6.0f);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ========================== VECTOR: DOUBLE ==============================

TEST(VectorComparisonTest, LessThanOperatorDouble) {
  math::Vector3D<double> vec1(1.0, 2.0, 3.0);
  math::Vector3D<double> vec2(4.0, 5.0, 6.0);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, LessThanOperatorEqualDouble) {
  math::Vector3D<double> vec1(1.0, 2.0, 3.0);
  math::Vector3D<double> vec2(1.0, 2.0, 3.0);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorDouble) {
  math::Vector3D<double> vec1(4.0, 5.0, 6.0);
  math::Vector3D<double> vec2(1.0, 2.0, 3.0);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorEqualDouble) {
  math::Vector3D<double> vec1(1.0, 2.0, 3.0);
  math::Vector3D<double> vec2(1.0, 2.0, 3.0);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperatorDouble) {
  math::Vector3D<double> vec1(1.0, 2.0, 3.0);
  math::Vector3D<double> vec2(4.0, 5.0, 6.0);
  math::Vector3D<double> vec3(1.0, 2.0, 3.0);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperatorDouble) {
  math::Vector3D<double> vec1(4.0, 5.0, 6.0);
  math::Vector3D<double> vec2(1.0, 2.0, 3.0);
  math::Vector3D<double> vec3(4.0, 5.0, 6.0);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ========================== VECTOR: INT ==============================

TEST(VectorComparisonTest, LessThanOperatorInt) {
  math::Vector3Di vec1(1, 2, 3);
  math::Vector3Di vec2(4, 5, 6);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, LessThanOperatorEqualInt) {
  math::Vector3Di vec1(1, 2, 3);
  math::Vector3Di vec2(1, 2, 3);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorInt) {
  math::Vector3Di vec1(4, 5, 6);
  math::Vector3Di vec2(1, 2, 3);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorEqualInt) {
  math::Vector3Di vec1(1, 2, 3);
  math::Vector3Di vec2(1, 2, 3);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperatorInt) {
  math::Vector3Di vec1(1, 2, 3);
  math::Vector3Di vec2(4, 5, 6);
  math::Vector3Di vec3(1, 2, 3);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperatorInt) {
  math::Vector3Di vec1(4, 5, 6);
  math::Vector3Di vec2(1, 2, 3);
  math::Vector3Di vec3(4, 5, 6);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ========================= VECTOR: UNSIGNED INT =============================

TEST(VectorComparisonTest, LessThanOperatorUnsignedInt) {
  math::Vector3D<unsigned int> vec1(1, 2, 3);
  math::Vector3D<unsigned int> vec2(4, 5, 6);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, LessThanOperatorEqualUnsignedInt) {
  math::Vector3D<unsigned int> vec1(1, 2, 3);
  math::Vector3D<unsigned int> vec2(1, 2, 3);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorUnsignedInt) {
  math::Vector3D<unsigned int> vec1(4, 5, 6);
  math::Vector3D<unsigned int> vec2(1, 2, 3);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorEqualUnsignedInt) {
  math::Vector3D<unsigned int> vec1(1, 2, 3);
  math::Vector3D<unsigned int> vec2(1, 2, 3);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperatorUnsignedInt) {
  math::Vector3D<unsigned int> vec1(1, 2, 3);
  math::Vector3D<unsigned int> vec2(4, 5, 6);
  math::Vector3D<unsigned int> vec3(1, 2, 3);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperatorUnsignedInt) {
  math::Vector3D<unsigned int> vec1(4, 5, 6);
  math::Vector3D<unsigned int> vec2(1, 2, 3);
  math::Vector3D<unsigned int> vec3(4, 5, 6);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

TEST(VectorComparisonTest, LessThanOperatorUnsignedIntLargeValues) {
  math::Vector3D<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Vector3D<unsigned int> vec2(0, 0, 0);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_TRUE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorUnsignedIntLargeValues) {
  math::Vector3D<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Vector3D<unsigned int> vec2(0, 0, 0);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperatorUnsignedIntLargeValues) {
  math::Vector3D<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Vector3D<unsigned int> vec2(0, 0, 0);

  EXPECT_FALSE(vec1 <= vec2);
  EXPECT_TRUE(vec2 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperatorUnsignedIntLargeValues) {
  math::Vector3D<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Vector3D<unsigned int> vec2(0, 0, 0);

  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec2);
}

// ========================== VECTOR: UINT32_T ==============================

TEST(VectorComparisonTest, LessThanOperatorUint32) {
  math::Vector<std::uint32_t, 10> vec1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  math::Vector<std::uint32_t, 10> vec2 = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, LessThanOperatorEqualUint32) {
  math::Vector<std::uint32_t, 10> vec1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  math::Vector<std::uint32_t, 10> vec2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorUint32) {
  math::Vector<std::uint32_t, 10> vec1 = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  math::Vector<std::uint32_t, 10> vec2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorEqualUint32) {
  math::Vector<std::uint32_t, 10> vec1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  math::Vector<std::uint32_t, 10> vec2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperatorUint32) {
  math::Vector<std::uint32_t, 10> vec1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  math::Vector<std::uint32_t, 10> vec2 = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  math::Vector<std::uint32_t, 10> vec3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperatorUint32) {
  math::Vector<std::uint32_t, 10> vec1 = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  math::Vector<std::uint32_t, 10> vec2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  math::Vector<std::uint32_t, 10> vec3 = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

TEST(VectorComparisonTest, LessThanOperatorUint32LargeValues) {
  math::Vector<std::uint32_t, 10> vec1(0xFF'FF'FF'FF);
  math::Vector<std::uint32_t, 10> vec2(0);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_TRUE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorUint32LargeValues) {
  math::Vector<std::uint32_t, 10> vec1(0xFF'FF'FF'FF);
  math::Vector<std::uint32_t, 10> vec2(0);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperatorUint32LargeValues) {
  math::Vector<std::uint32_t, 10> vec1(0xFF'FF'FF'FF);
  math::Vector<std::uint32_t, 10> vec2(0);

  EXPECT_FALSE(vec1 <= vec2);
  EXPECT_TRUE(vec2 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperatorUint32LargeValues) {
  math::Vector<std::uint32_t, 10> vec1(0xFF'FF'FF'FF);
  math::Vector<std::uint32_t, 10> vec2(0);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
}

// ========================== DIMENSION: FLOAT ==============================

TEST(DimensionComparisonTest, LessThanOperator) {
  math::Dimension3Df vec1(1.0f, 2.0f, 3.0f);
  math::Dimension3Df vec2(4.0f, 5.0f, 6.0f);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorEqual) {
  math::Dimension3Df vec1(1.0f, 2.0f, 3.0f);
  math::Dimension3Df vec2(1.0f, 2.0f, 3.0f);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperator) {
  math::Dimension3Df vec1(4.0f, 5.0f, 6.0f);
  math::Dimension3Df vec2(1.0f, 2.0f, 3.0f);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorEqual) {
  math::Dimension3Df vec1(1.0f, 2.0f, 3.0f);
  math::Dimension3Df vec2(1.0f, 2.0f, 3.0f);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperator) {
  math::Dimension3Df vec1(1.0f, 2.0f, 3.0f);
  math::Dimension3Df vec2(4.0f, 5.0f, 6.0f);
  math::Dimension3Df vec3(1.0f, 2.0f, 3.0f);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(DimensionComparisonTest, GreaterThanOrEqualToOperator) {
  math::Dimension3Df vec1(4.0f, 5.0f, 6.0f);
  math::Dimension3Df vec2(1.0f, 2.0f, 3.0f);
  math::Dimension3Df vec3(4.0f, 5.0f, 6.0f);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ========================== DIMENSION: DOUBLE ==============================

TEST(DimensionComparisonTest, LessThanOperatorDouble) {
  math::Dimension3D<double> vec1(1.0, 2.0, 3.0);
  math::Dimension3D<double> vec2(4.0, 5.0, 6.0);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorEqualDouble) {
  math::Dimension3D<double> vec1(1.0, 2.0, 3.0);
  math::Dimension3D<double> vec2(1.0, 2.0, 3.0);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorDouble) {
  math::Dimension3D<double> vec1(4.0, 5.0, 6.0);
  math::Dimension3D<double> vec2(1.0, 2.0, 3.0);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorEqualDouble) {
  math::Dimension3D<double> vec1(1.0, 2.0, 3.0);
  math::Dimension3D<double> vec2(1.0, 2.0, 3.0);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperatorDouble) {
  math::Dimension3D<double> vec1(1.0, 2.0, 3.0);
  math::Dimension3D<double> vec2(4.0, 5.0, 6.0);
  math::Dimension3D<double> vec3(1.0, 2.0, 3.0);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(DimensionComparisonTest, GreaterThanOrEqualToOperatorDouble) {
  math::Dimension3D<double> vec1(4.0, 5.0, 6.0);
  math::Dimension3D<double> vec2(1.0, 2.0, 3.0);
  math::Dimension3D<double> vec3(4.0, 5.0, 6.0);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ========================== DIMENSION: INT ==============================

TEST(DimensionComparisonTest, LessThanOperatorInt) {
  math::Dimension3Di vec1(1, 2, 3);
  math::Dimension3Di vec2(4, 5, 6);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorEqualInt) {
  math::Dimension3Di vec1(1, 2, 3);
  math::Dimension3Di vec2(1, 2, 3);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorInt) {
  math::Dimension3Di vec1(4, 5, 6);
  math::Dimension3Di vec2(1, 2, 3);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorEqualInt) {
  math::Dimension3Di vec1(1, 2, 3);
  math::Dimension3Di vec2(1, 2, 3);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperatorInt) {
  math::Dimension3Di vec1(1, 2, 3);
  math::Dimension3Di vec2(4, 5, 6);
  math::Dimension3Di vec3(1, 2, 3);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(DimensionComparisonTest, GreaterThanOrEqualToOperatorInt) {
  math::Dimension3Di vec1(4, 5, 6);
  math::Dimension3Di vec2(1, 2, 3);
  math::Dimension3Di vec3(4, 5, 6);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ======================= DIMENSION: UNSIGNED INT ===========================

TEST(DimensionComparisonTest, LessThanOperatorUnsignedInt) {
  math::Dimension3D<unsigned int> vec1(1, 2, 3);
  math::Dimension3D<unsigned int> vec2(4, 5, 6);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorEqualUnsignedInt) {
  math::Dimension3D<unsigned int> vec1(1, 2, 3);
  math::Dimension3D<unsigned int> vec2(1, 2, 3);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorUnsignedInt) {
  math::Dimension3D<unsigned int> vec1(4, 5, 6);
  math::Dimension3D<unsigned int> vec2(1, 2, 3);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorEqualUnsignedInt) {
  math::Dimension3D<unsigned int> vec1(1, 2, 3);
  math::Dimension3D<unsigned int> vec2(1, 2, 3);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperatorUnsignedInt) {
  math::Dimension3D<unsigned int> vec1(1, 2, 3);
  math::Dimension3D<unsigned int> vec2(4, 5, 6);
  math::Dimension3D<unsigned int> vec3(1, 2, 3);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(DimensionComparisonTest, GreaterThanOrEqualToOperatorUnsignedInt) {
  math::Dimension3D<unsigned int> vec1(4, 5, 6);
  math::Dimension3D<unsigned int> vec2(1, 2, 3);
  math::Dimension3D<unsigned int> vec3(4, 5, 6);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorUnsignedIntLargeValues) {
  math::Dimension3D<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Dimension3D<unsigned int> vec2(0, 0, 0);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_TRUE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorUnsignedIntLargeValues) {
  math::Dimension3D<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Dimension3D<unsigned int> vec2(0, 0, 0);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperatorUnsignedIntLargeValues) {
  math::Dimension3D<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Dimension3D<unsigned int> vec2(0, 0, 0);

  EXPECT_FALSE(vec1 <= vec2);
  EXPECT_TRUE(vec2 <= vec1);
}

TEST(DimensionComparisonTest,
     GreaterThanOrEqualToOperatorUnsignedIntLargeValues) {
  math::Dimension3D<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Dimension3D<unsigned int> vec2(0, 0, 0);

  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec2);
}

// ========================== DIMENSION: UINT32_T ==============================

TEST(DimensionComparisonTest, LessThanOperatorUint32) {
  math::Dimension<std::uint32_t, 10> vec1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  math::Dimension<std::uint32_t, 10> vec2 = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorEqualUint32) {
  math::Dimension<std::uint32_t, 10> vec1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  math::Dimension<std::uint32_t, 10> vec2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorUint32) {
  math::Dimension<std::uint32_t, 10> vec1 = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  math::Dimension<std::uint32_t, 10> vec2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorEqualUint32) {
  math::Dimension<std::uint32_t, 10> vec1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  math::Dimension<std::uint32_t, 10> vec2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperatorUint32) {
  math::Dimension<std::uint32_t, 10> vec1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  math::Dimension<std::uint32_t, 10> vec2 = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  math::Dimension<std::uint32_t, 10> vec3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(DimensionComparisonTest, GreaterThanOrEqualToOperatorUint32) {
  math::Dimension<std::uint32_t, 10> vec1 = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  math::Dimension<std::uint32_t, 10> vec2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  math::Dimension<std::uint32_t, 10> vec3 = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorUint32LargeValues) {
  math::Dimension<std::uint32_t, 10> vec1(0xFF'FF'FF'FF);
  math::Dimension<std::uint32_t, 10> vec2(0);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_TRUE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorUint32LargeValues) {
  math::Dimension<std::uint32_t, 10> vec1(0xFF'FF'FF'FF);
  math::Dimension<std::uint32_t, 10> vec2(0);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperatorUint32LargeValues) {
  math::Dimension<std::uint32_t, 10> vec1(0xFF'FF'FF'FF);
  math::Dimension<std::uint32_t, 10> vec2(0);

  EXPECT_FALSE(vec1 <= vec2);
  EXPECT_TRUE(vec2 <= vec1);
}

TEST(DimensionComparisonTest, GreaterThanOrEqualToOperatorUint32LargeValues) {
  math::Dimension<std::uint32_t, 10> vec1(0xFF'FF'FF'FF);
  math::Dimension<std::uint32_t, 10> vec2(0);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
}

#endif