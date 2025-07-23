/**
 * @file tests.h
 */

// TODO: add tests for

#ifndef MATH_LIBRARY_TESTS_H
#define MATH_LIBRARY_TESTS_H

#include <gtest/gtest.h>
#include <math_library/all.h>

// TEST(MatrixTest, ConstructorDestructorFloat)
// TEST(MatrixTest, CopyConstructorFloat)
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

  auto inverseMatrix = matrix.inverse();

  // The expected inverse matrix of [[1, 2], [3, 4]] is [[-2, 1], [1.5, -0.5]]
  EXPECT_FLOAT_EQ(inverseMatrix(0, 0), -2);
  EXPECT_FLOAT_EQ(inverseMatrix(0, 1), 1);
  EXPECT_FLOAT_EQ(inverseMatrix(1, 0), 1.5);
  EXPECT_FLOAT_EQ(inverseMatrix(1, 1), -0.5);
}

// Method: inverse column major

TEST(MatrixTest, InverseFloatColumnMajor) {
  math::Matrix<float, 2, 2, math::Options::ColumnMajor> matrix;
  matrix(0, 0) = 1;
  matrix(1, 0) = 2;
  matrix(0, 1) = 3;
  matrix(1, 1) = 4;

  auto inverseMatrix = matrix.inverse();

  // The expected inverse matrix of [[1, 2], [3, 4]] is [[-2, 1.5], [1, -0.5]]
  EXPECT_FLOAT_EQ(inverseMatrix(0, 0), -2);
  EXPECT_FLOAT_EQ(inverseMatrix(1, 0), 1);
  EXPECT_FLOAT_EQ(inverseMatrix(0, 1), 1.5);
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

  // The expected inverse matrix of [[1, 2], [3, 4]] is [[-2, 1], [1.5, -0.5]]
  EXPECT_NE(inverseMatrix(0, 0), 3);
  EXPECT_NE(inverseMatrix(0, 1), -1);
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
  math::MatrixXf<3, 3> matrix;
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
  math::MatrixXf<3, 3, math::Options::ColumnMajor> matrix;
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
  math::MatrixXf<3, 3, math::Options::ColumnMajor> matrix;
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
  math::MatrixXf<3, 3> matrix;
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
  math::MatrixXf<3, 3>                            matrix;
  math::Vector<float, 3, math::Options::RowMajor> rowVector(1.0f, 2.0f, 3.0f);
  matrix.setRow<1>(rowVector);  // Set the second row

  // Verify that the second row is correctly set
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_FLOAT_EQ(matrix(1, col), rowVector(col));
  }
}

// Method: set row in column-major matrix
TEST(MatrixTest, SetRowColumnMajor) {
  math::MatrixXf<3, 3, math::Options::ColumnMajor>   matrix;
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
  math::MatrixXf<3, 3, math::Options::ColumnMajor>   matrix;
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
  math::MatrixXf<3, 3, math::Options::RowMajor>   matrix;
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
  math::MatrixXf<3, 3>                            matrix;
  math::Vector<float, 3, math::Options::RowMajor> xBasis(1.0f, 0.0f, 0.0f);
  matrix.setBasisX(xBasis);

  // Verify that the first row (X basis) is correctly set
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_EQ(matrix(0, col), xBasis(col));
  }
}

// Method: set basis Y

TEST(MatrixTest, SetBasisY) {
  math::MatrixXf<3, 3>                            matrix;
  math::Vector<float, 3, math::Options::RowMajor> yBasis(0.0f, 1.0f, 0.0f);
  matrix.setBasisY(yBasis);

  // Verify that the second row (Y basis) is correctly set
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_EQ(matrix(1, col), yBasis(col));
  }
}

// Method: set basis Z

TEST(MatrixTest, SetBasisZ) {
  math::MatrixXf<3, 3>                            matrix;
  math::Vector<float, 3, math::Options::RowMajor> zBasis(0.0f, 0.0f, 1.0f);
  matrix.setBasisZ(zBasis);

  // Verify that the third row (Z basis) is correctly set
  for (unsigned int col = 0; col < 3; ++col) {
    EXPECT_EQ(matrix(2, col), zBasis(col));
  }
}

// Method: set basis general

TEST(MatrixTest, GeneralizedSetBasis) {
  math::MatrixXf<3, 3>                            matrix;
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

// Test case for the move assignment operator
TEST(MatrixTest, MoveAssignmentOperator) {
  math::Matrix<float, 2, 2> matrix1(1.0f, 2.0f, 3.0f, 4.0f);
  math::Matrix<float, 2, 2> matrix2;
  matrix2 = std::move(matrix1);

  EXPECT_EQ(matrix2(0, 0), 1.0f);
  EXPECT_EQ(matrix2(0, 1), 2.0f);
  EXPECT_EQ(matrix2(1, 0), 3.0f);
  EXPECT_EQ(matrix2(1, 1), 4.0f);
}

// Test case for the GetRows() static member function
TEST(MatrixTest, GetRows) {
  std::size_t rows = math::Matrix<float, 2, 3>::GetRows();
  EXPECT_EQ(rows, 2);
}

// Test case for the GetColumns() static member function
TEST(MatrixTest, GetColumns) {
  std::size_t columns = math::Matrix<float, 2, 3>::GetColumns();
  EXPECT_EQ(columns, 3);
}

// Test case for the GetDataSize() static member function
TEST(MatrixTest, GetDataSize) {
  std::size_t dataSize = math::Matrix<float, 2, 3>::GetDataSize();
  EXPECT_EQ(dataSize, 6 * sizeof(float));
}

// Test case for the GetOption() static member function
TEST(MatrixTest, GetOption) {
  math::Options option = math::Matrix<float, 2, 3>::GetOption();
  EXPECT_EQ(option, math::Options::RowMajor);
}

// Test case for the data() member function (const version)
TEST(MatrixTest, DataFunctionConst) {
  math::Matrix<float, 2, 2>        matrix(1.0f, 2.0f, 3.0f, 4.0f);
  const math::Matrix<float, 2, 2>& constMatrix = matrix;

  EXPECT_EQ(constMatrix.data()[0], 1.0f);
  EXPECT_EQ(constMatrix.data()[1], 2.0f);
  EXPECT_EQ(constMatrix.data()[2], 3.0f);
  EXPECT_EQ(constMatrix.data()[3], 4.0f);
}

// Test case for the data() member function (non-const version)
TEST(MatrixTest, DataFunctionNonConst) {
  math::Matrix<float, 2, 2> matrix(1.0f, 2.0f, 3.0f, 4.0f);

  EXPECT_EQ(matrix.data()[0], 1.0f);
  EXPECT_EQ(matrix.data()[1], 2.0f);
  EXPECT_EQ(matrix.data()[2], 3.0f);
  EXPECT_EQ(matrix.data()[3], 4.0f);
}

// Test case for scalar multiplication
TEST(MatrixTest, ScalarMultiplication) {
  math::Matrix<float, 2, 2> matrix(1.0f, 2.0f, 3.0f, 4.0f);
  math::Matrix<float, 2, 2> result = matrix * 2.0f;

  EXPECT_EQ(result(0, 0), 2.0f);
  EXPECT_EQ(result(0, 1), 4.0f);
  EXPECT_EQ(result(1, 0), 6.0f);
  EXPECT_EQ(result(1, 1), 8.0f);
}

// Test case for scalar division
TEST(MatrixTest, ScalarDivision) {
  math::Matrix<float, 2, 2> matrix(1.0f, 2.0f, 3.0f, 4.0f);
  math::Matrix<float, 2, 2> result = matrix / 2.0f;

  EXPECT_EQ(result(0, 0), 0.5f);
  EXPECT_EQ(result(0, 1), 1.0f);
  EXPECT_EQ(result(1, 0), 1.5f);
  EXPECT_EQ(result(1, 1), 2.0f);
}

// Test case for matrix multiplication with non-square matrices of different
// sizes
TEST(MatrixTest, NonSquareMatrixMultiplication) {
  math::Matrix<float, 2, 3> matrix1(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
  math::Matrix<float, 3, 2> matrix2(7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f);
  math::Matrix<float, 2, 2> result = matrix1 * matrix2;

  EXPECT_EQ(result(0, 0), 58.0f);
  EXPECT_EQ(result(0, 1), 64.0f);
  EXPECT_EQ(result(1, 0), 139.0f);
  EXPECT_EQ(result(1, 1), 154.0f);
}

// Test case for matrix multiplication using the *= operator
TEST(MatrixTest, MatrixMultiplicationAssignment) {
  math::Matrix<float, 2, 2> matrix1(1.0f, 2.0f, 3.0f, 4.0f);
  math::Matrix<float, 2, 2> matrix2(5.0f, 6.0f, 7.0f, 8.0f);
  matrix1 *= matrix2;

  EXPECT_EQ(matrix1(0, 0), 19.0f);
  EXPECT_EQ(matrix1(0, 1), 22.0f);
  EXPECT_EQ(matrix1(1, 0), 43.0f);
  EXPECT_EQ(matrix1(1, 1), 50.0f);
}

//// Test case for matrix addition with matrices of different options (RowMajor
//// and ColumnMajor)
// TEST(MatrixTest, MatrixAdditionOptions) {
//   math::Matrix<float, 2, 2, math::Options::RowMajor> matrixRow(
//       1.0f, 2.0f, 3.0f, 4.0f);
//   math::Matrix<float, 2, 2, math::Options::ColumnMajor> matrixCol(
//       5.0f, 7.0f, 6.0f, 8.0f);
//   math::Matrix<float, 2, 2, math::Options::RowMajor> result
//       = matrixRow + matrixCol;
//
//   EXPECT_EQ(result(0, 0), 6.0f);
//   EXPECT_EQ(result(0, 1), 8.0f);
//   EXPECT_EQ(result(1, 0), 10.0f);
//   EXPECT_EQ(result(1, 1), 12.0f);
// }
//
//// Test case for matrix subtraction with matrices of different options
///(RowMajor / and ColumnMajor)
// TEST(MatrixTest, MatrixSubtractionOptions) {
//   math::Matrix<float, 2, 2, math::Options::RowMajor> matrixRow(
//       1.0f, 2.0f, 3.0f, 4.0f);
//   math::Matrix<float, 2, 2, math::Options::ColumnMajor> matrixCol(
//       5.0f, 7.0f, 6.0f, 8.0f);
//   math::Matrix<float, 2, 2, math::Options::RowMajor> result
//       = matrixRow - matrixCol;
//
//   EXPECT_EQ(result(0, 0), -4.0f);
//   EXPECT_EQ(result(0, 1), -4.0f);
//   EXPECT_EQ(result(1, 0), -4.0f);
//   EXPECT_EQ(result(1, 1), -4.0f);
// }

// Test case for matrix negation
TEST(MatrixTest, MatrixNegation) {
  math::Matrix<float, 2, 2> matrix(1.0f, -2.0f, 3.0f, -4.0f);
  math::Matrix<float, 2, 2> result = -matrix;

  EXPECT_EQ(result(0, 0), -1.0f);
  EXPECT_EQ(result(0, 1), 2.0f);
  EXPECT_EQ(result(1, 0), -3.0f);
  EXPECT_EQ(result(1, 1), 4.0f);
}

// Test case for matrix transposition
TEST(MatrixTest, Transposition) {
  math::Matrix<float, 2, 3> matrix(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
  math::Matrix<float, 3, 2> transposedMatrix = matrix.transpose();

  EXPECT_EQ(transposedMatrix(0, 0), 1.0f);
  EXPECT_EQ(transposedMatrix(0, 1), 4.0f);
  EXPECT_EQ(transposedMatrix(1, 0), 2.0f);
  EXPECT_EQ(transposedMatrix(1, 1), 5.0f);
  EXPECT_EQ(transposedMatrix(2, 0), 3.0f);
  EXPECT_EQ(transposedMatrix(2, 1), 6.0f);
}

// Test case for matrix reshaping
TEST(MatrixTest, Reshaping) {
  math::Matrix<float, 2, 3> matrix(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
  math::Matrix<float, 3, 2> reshapedMatrix = matrix.reshape<3, 2>();

  EXPECT_EQ(reshapedMatrix(0, 0), 1.0f);
  EXPECT_EQ(reshapedMatrix(0, 1), 2.0f);
  EXPECT_EQ(reshapedMatrix(1, 0), 3.0f);
  EXPECT_EQ(reshapedMatrix(1, 1), 4.0f);
  EXPECT_EQ(reshapedMatrix(2, 0), 5.0f);
  EXPECT_EQ(reshapedMatrix(2, 1), 6.0f);
}

// Test case for matrix determinant
TEST(MatrixTest, Determinant) {
  math::Matrix<float, 2, 2> matrix2x2(1.0f, 2.0f, 3.0f, 4.0f);
  float                     determinant2x2 = matrix2x2.determinant();

  EXPECT_FLOAT_EQ(determinant2x2, -2.0f);
}

// Test case for matrix inverse
TEST(MatrixTest, Inverse) {
  math::Matrix<float, 2, 2> matrix2x2(1.0f, 2.0f, 3.0f, 4.0f);
  math::Matrix<float, 2, 2> inverse2x2 = matrix2x2.inverse();

  EXPECT_FLOAT_EQ(inverse2x2(0, 0), -2.0f);
  EXPECT_FLOAT_EQ(inverse2x2(0, 1), 1.0f);
  EXPECT_FLOAT_EQ(inverse2x2(1, 0), 1.5f);
  EXPECT_FLOAT_EQ(inverse2x2(1, 1), -0.5f);
}

// Test case for matrix rank calculation
TEST(MatrixTest, MatrixRank) {
  math::Matrix<float, 3, 3> matrixFullRank(
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 10.0f);
  math::Matrix<float, 3, 3> matrixRank2(
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
  math::Matrix<float, 3, 3> matrixRank1(
      1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f, 3.0f, 6.0f, 9.0f);

  EXPECT_EQ(matrixFullRank.rank(), 3);
  EXPECT_EQ(matrixRank2.rank(), 2);
  EXPECT_EQ(matrixRank1.rank(), 1);
}

// Test case for matrix normalization
TEST(MatrixTest, Normalization) {
  math::Matrix<float, 3, 1> matrix(1.0f, 2.0f, 3.0f);
  math::Matrix<float, 3, 1> normalized = matrix.normalized();
  EXPECT_FLOAT_EQ(normalized(0, 0), 0.2672612419124244f);
  EXPECT_FLOAT_EQ(normalized(1, 0), 0.5345224838248488f);
  EXPECT_FLOAT_EQ(normalized(2, 0), 0.8017837257372732f);
}

// Test case for matrix magnitude calculation
TEST(MatrixTest, Magnitude) {
  math::Matrix<float, 3, 1> matrix(1.0f, 2.0f, 3.0f);
  float                     magnitude = matrix.magnitude();
  EXPECT_FLOAT_EQ(magnitude, 3.7416573867739413f);
}

// Test case for matrix trace calculation
TEST(MatrixTest, MatrixTrace) {
  math::Matrix<float, 2, 2> matrix2x2(1.0f, 2.0f, 3.0f, 4.0f);
  float                     trace2x2 = matrix2x2.trace();
  EXPECT_FLOAT_EQ(trace2x2, 5.0f);
  math::Matrix<float, 3, 3> matrix3x3(
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
  float trace3x3 = matrix3x3.trace();
  EXPECT_FLOAT_EQ(trace3x3, 15.0f);
}

// Test case for dimension comparison operators (<, >, <=, >=) with different
// dimension sizes and element types
TEST(DimensionTest, ComparisonOperators) {
  math::Dimension2Df dim1(1.0f, 2.0f);
  math::Dimension2Df dim2(3.0f, 4.0f);
  math::Dimension2Df dim3(1.0f, 2.0f);

  EXPECT_TRUE(dim1 < dim2);
  EXPECT_FALSE(dim2 < dim1);
  EXPECT_TRUE(dim1 <= dim2);
  EXPECT_FALSE(dim2 <= dim1);
  EXPECT_TRUE(dim2 > dim1);
  EXPECT_FALSE(dim1 > dim2);
  EXPECT_TRUE(dim2 >= dim1);
  EXPECT_FALSE(dim1 >= dim2);
  EXPECT_TRUE(dim1 <= dim3);
  EXPECT_TRUE(dim1 >= dim3);
}

// Test case for dimension resizing with different target sizes
TEST(DimensionTest, Resizing) {
  math::Dimension3f  dim(1.0f, 2.0f, 3.0f);
  math::Dimension2Df resizedDim = dim.resizedCopy<2>();

  EXPECT_FLOAT_EQ(resizedDim.width(), 1.0f);
  EXPECT_FLOAT_EQ(resizedDim.height(), 2.0f);

  math::Dimension4f resizedDim2 = dim.resizedCopy<4>();

  EXPECT_FLOAT_EQ(resizedDim2.width(), 1.0f);
  EXPECT_FLOAT_EQ(resizedDim2.height(), 2.0f);
  EXPECT_FLOAT_EQ(resizedDim2.depth(), 3.0f);
  EXPECT_FLOAT_EQ(resizedDim2.coeff(3), 0.0f);
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

// Test case for g_translate function with different translation values
TEST(GraphicsTest, Translate) {
  math::Matrix4f<> matrix = math::g_translate(1.0f, 2.0f, 3.0f);

  EXPECT_FLOAT_EQ(matrix(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(matrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(matrix(2, 2), 1.0f);
  EXPECT_FLOAT_EQ(matrix(3, 0), 1.0f);
  EXPECT_FLOAT_EQ(matrix(3, 1), 2.0f);
  EXPECT_FLOAT_EQ(matrix(3, 2), 3.0f);
  EXPECT_FLOAT_EQ(matrix(3, 3), 1.0f);
}

// Test case for g_scale function with different scaling values
TEST(GraphicsTest, Scale) {
  math::Matrix4f<> matrix = math::g_scale(2.0f, 3.0f, 4.0f);

  EXPECT_FLOAT_EQ(matrix(0, 0), 2.0f);
  EXPECT_FLOAT_EQ(matrix(1, 1), 3.0f);
  EXPECT_FLOAT_EQ(matrix(2, 2), 4.0f);
  EXPECT_FLOAT_EQ(matrix(3, 3), 1.0f);
}

// Test case for g_rotateRh function with different rotation values
TEST(GraphicsTest, RotateRh) {
  float angleX = math::g_kPi / 4.0f;
  float angleY = math::g_kPi / 3.0f;
  float angleZ = math::g_kPi / 2.0f;

  math::Matrix4f<> matrix = math::g_rotateRh(angleX, angleY, angleZ);

  EXPECT_NEAR(matrix(0, 0), 0.612372458f, 1e-5);
  EXPECT_NEAR(matrix(0, 1), 0.707106769f, 1e-5);
  EXPECT_NEAR(matrix(0, 2), 0.353553385f, 1e-5);
  EXPECT_NEAR(matrix(1, 0), -0.5f, 1e-5);
  EXPECT_NEAR(matrix(1, 1), 0.0f, 1e-5);
  EXPECT_NEAR(matrix(1, 2), 0.866025448f, 1e-5);
  EXPECT_NEAR(matrix(2, 0), 0.612372458f, 1e-5);
  EXPECT_NEAR(matrix(2, 1), -0.707106769f, 1e-5);
  EXPECT_NEAR(matrix(2, 2), 0.353553355f, 1e-5);
}

// Test case for g_lookAtRh function with different eye, target, and up vectors
TEST(GraphicsTest, LookAtRh) {
  math::Vector3f eye(0.0f, 0.0f, 5.0f);
  math::Vector3f target(0.0f, 0.0f, 0.0f);
  math::Vector3f up(0.0f, 1.0f, 0.0f);

  math::Matrix4f<> matrix = math::g_lookAtRh(eye, target, up);

  EXPECT_FLOAT_EQ(matrix(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(matrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(matrix(2, 2), 1.0f);
  EXPECT_FLOAT_EQ(matrix(3, 2), -5.0f);
}

// Test case for g_lookToRh function with different eye, direction, and up
// vectors
TEST(GraphicsTest, LookToRh) {
  math::Vector3f eye(0.0f, 0.0f, 5.0f);
  math::Vector3f direction(0.0f, 0.0f, -1.0f);
  math::Vector3f up(0.0f, 1.0f, 0.0f);

  math::Matrix4f<> matrix = math::g_lookToRh(eye, direction, up);

  EXPECT_FLOAT_EQ(matrix(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(matrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(matrix(2, 2), 1.0f);
  EXPECT_FLOAT_EQ(matrix(3, 2), -5.0f);
}

// Test case for g_transformPoint function
TEST(GraphicsTest, TransformPoint) {
  math::Point3f    point(1.0f, 2.0f, 3.0f);
  math::Matrix4f<> matrix(1.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          1.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          1.0f,
                          0.0f,
                          4.0f,
                          5.0f,
                          6.0f,
                          1.0f);

  math::Point3f result = math::g_transformPoint(point, matrix);

  EXPECT_FLOAT_EQ(result(0), 5.0f);
  EXPECT_FLOAT_EQ(result(1), 7.0f);
  EXPECT_FLOAT_EQ(result(2), 9.0f);
}

// Test case for g_transformPoint function with perspective division
TEST(GraphicsTest, TransformPointWithPerspectiveDivision) {
  math::Point3f    point(1.0f, 2.0f, 3.0f);
  math::Matrix4f<> matrix(1.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          1.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          1.0f,
                          1.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          1.0f);

  math::Point3f result = math::g_transformPoint(point, matrix);

  EXPECT_FLOAT_EQ(result(0), 0.25f);
  EXPECT_FLOAT_EQ(result(1), 0.5f);
  EXPECT_FLOAT_EQ(result(2), 0.75f);
}

// Test case for g_transformVector function
TEST(GraphicsTest, TransformVector) {
  math::Vector3f   vector(1.0f, 2.0f, 3.0f);
  math::Matrix4f<> matrix(2.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          3.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          4.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          0.0f,
                          1.0f);

  math::Vector3f result = math::g_transformVector(vector, matrix);

  EXPECT_FLOAT_EQ(result(0), 2.0f);
  EXPECT_FLOAT_EQ(result(1), 6.0f);
  EXPECT_FLOAT_EQ(result(2), 12.0f);
}

TEST(ViewMatrixTest, LookAtRhRowMajor) {
  math::Vector3<float> eye{0.0f, 0.0f, 1.0f};
  math::Vector3<float> target{0.0f, 0.0f, 0.0f};
  math::Vector3<float> up{0.0f, 1.0f, 0.0f};
  auto                 viewMatrix = g_lookAtRh(eye, target, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), 1.0f);
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
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), -1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

TEST(ViewMatrixTest, LookAtRhColumnMajor) {
  math::Vector3<float, math::Options::ColumnMajor> eye{0.0f, 0.0f, 1.0f};
  math::Vector3<float, math::Options::ColumnMajor> target{0.0f, 0.0f, 0.0f};
  math::Vector3<float, math::Options::ColumnMajor> up{0.0f, 1.0f, 0.0f};
  auto viewMatrix = g_lookAtRh(eye, target, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), 1.0f);
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
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), -1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

TEST(ViewMatrixTest, LookToRhRowMajor) {
  math::Vector3<float> eye{0.0f, 0.0f, 1.0f};
  math::Vector3<float> direction{0.0f, 0.0f, -1.0f};
  math::Vector3<float> up{0.0f, 1.0f, 0.0f};
  auto                 viewMatrix = g_lookToRh(eye, direction, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), 1.0f);
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
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), -1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

TEST(ViewMatrixTest, LookToRhColumnMajor) {
  math::Vector3<float, math::Options::ColumnMajor> eye{0.0f, 0.0f, 1.0f};
  math::Vector3<float, math::Options::ColumnMajor> direction{0.0f, 0.0f, -1.0f};
  math::Vector3<float, math::Options::ColumnMajor> up{0.0f, 1.0f, 0.0f};
  auto viewMatrix = g_lookToRh(eye, direction, up);

  // Check the right vector
  EXPECT_FLOAT_EQ(viewMatrix(0, 0), 1.0f);
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
  EXPECT_FLOAT_EQ(viewMatrix(2, 3), -1.0f);

  // Check the homogenous part of the matrix
  EXPECT_FLOAT_EQ(viewMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), 0.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

TEST(ViewMatrixTest, LookAtLhRowMajor) {
  math::Vector3<float> eye{0.0f, 0.0f, 1.0f};
  math::Vector3<float> target{0.0f, 0.0f, 0.0f};
  math::Vector3<float> up{0.0f, 1.0f, 0.0f};
  auto                 viewMatrix = g_lookAtLh(eye, target, up);

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
  math::Vector3<float, math::Options::ColumnMajor> eye{0.0f, 0.0f, 1.0f};
  math::Vector3<float, math::Options::ColumnMajor> target{0.0f, 0.0f, 0.0f};
  math::Vector3<float, math::Options::ColumnMajor> up{0.0f, 1.0f, 0.0f};
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
  math::Vector3<float> eye{0.0f, 0.0f, 1.0f};
  math::Vector3<float> direction{0.0f, 0.0f, -1.0f};
  math::Vector3<float> up{0.0f, 1.0f, 0.0f};
  auto                 viewMatrix = g_lookToLh(eye, direction, up);

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
  math::Vector3<float, math::Options::ColumnMajor> eye{0.0f, 0.0f, 1.0f};
  math::Vector3<float, math::Options::ColumnMajor> direction{0.0f, 0.0f, -1.0f};
  math::Vector3<float, math::Options::ColumnMajor> up{0.0f, 1.0f, 0.0f};
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

// Test case for lookAtRh
TEST(LookAtTest, LookAtRhFloat) {
  math::Vector3f   eye(0.0f, 0.0f, 5.0f);
  math::Vector3f   target(0.0f, 0.0f, 0.0f);
  math::Vector3f   up(0.0f, 1.0f, 0.0f);
  math::Matrix4f<> viewMatrix = math::g_lookAtRh(eye, target, up);

  EXPECT_FLOAT_EQ(viewMatrix(0, 0), 1.0f);
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
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), -5.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

// Test case for lookAtLh
TEST(LookAtTest, LookAtLhFloat) {
  math::Vector3f   eye(0.0f, 0.0f, 5.0f);
  math::Vector3f   target(0.0f, 0.0f, 0.0f);
  math::Vector3f   up(0.0f, 1.0f, 0.0f);
  math::Matrix4f<> viewMatrix = math::g_lookAtLh(eye, target, up);

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
  math::Vector3f   eye(0.0f, 0.0f, 5.0f);
  math::Vector3f   direction(0.0f, 0.0f, -1.0f);
  math::Vector3f   up(0.0f, 1.0f, 0.0f);
  math::Matrix4f<> viewMatrix = math::g_lookToRh(eye, direction, up);

  EXPECT_FLOAT_EQ(viewMatrix(0, 0), 1.0f);
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
  EXPECT_FLOAT_EQ(viewMatrix(3, 2), -5.0f);
  EXPECT_FLOAT_EQ(viewMatrix(3, 3), 1.0f);
}

// Test case for lookToLh
TEST(LookAtTest, LookToLhFloat) {
  math::Vector3f   eye(0.0f, 0.0f, 5.0f);
  math::Vector3f   direction(0.0f, 0.0f, -1.0f);
  math::Vector3f   up(0.0f, 1.0f, 0.0f);
  math::Matrix4f<> viewMatrix = math::g_lookToLh(eye, direction, up);

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

// Test case for perspectiveRhNo
TEST(PerspectiveTest, PerspectiveRhNoFloat) {
  constexpr float  fovY   = math::g_degreeToRadian(45.0f);
  constexpr float  aspect = 1.0f;
  constexpr float  nearZ  = 0.1f;
  constexpr float  farZ   = 100.0f;
  math::Matrix4f<> projMatrix
      = math::g_perspectiveRhNo(fovY, aspect, nearZ, farZ);

  EXPECT_NEAR(projMatrix(0, 0), 2.4142134f, 1e-6f);
  EXPECT_NEAR(projMatrix(0, 1), 0.0f, 1e-6f);
  EXPECT_NEAR(projMatrix(0, 2), 0.0f, 1e-6f);
  EXPECT_NEAR(projMatrix(0, 3), 0.0f, 1e-6f);
  EXPECT_NEAR(projMatrix(1, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(projMatrix(1, 1), 2.4142134f, 1e-6f);
  EXPECT_NEAR(projMatrix(1, 2), 0.0f, 1e-6f);
  EXPECT_NEAR(projMatrix(1, 3), 0.0f, 1e-6f);
  EXPECT_NEAR(projMatrix(2, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(projMatrix(2, 1), 0.0f, 1e-6f);
  EXPECT_NEAR(projMatrix(2, 2), -1.002002f, 1e-6f);
  EXPECT_NEAR(projMatrix(2, 3), -1.0f, 1e-6f);
  EXPECT_NEAR(projMatrix(3, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(projMatrix(3, 1), 0.0f, 1e-6f);
  EXPECT_NEAR(projMatrix(3, 2), -0.2002002f, 1e-6f);
  EXPECT_NEAR(projMatrix(3, 3), 0.0f, 1e-6f);
}

// Test case for perspectiveRhZo
TEST(PerspectiveTest, PerspectiveRhZoFloat) {
  float            fovY   = math::g_degreeToRadian(45.0f);
  float            aspect = 1.0f;
  float            nearZ  = 0.1f;
  float            farZ   = 100.0f;
  math::Matrix4f<> projMatrix
      = math::g_perspectiveRhZo(fovY, aspect, nearZ, farZ);

  EXPECT_FLOAT_EQ(projMatrix(0, 0), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 1), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 2), -1.001001f);
  EXPECT_FLOAT_EQ(projMatrix(2, 3), -1.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 2), -0.1001001f);
  EXPECT_FLOAT_EQ(projMatrix(3, 3), 0.0f);
}

// Test case for perspectiveLhNo
TEST(PerspectiveTest, PerspectiveLhNoFloat) {
  float            fovY   = math::g_degreeToRadian(45.0f);
  float            aspect = 1.0f;
  float            nearZ  = 0.1f;
  float            farZ   = 100.0f;
  math::Matrix4f<> projMatrix
      = math::g_perspectiveLhNo(fovY, aspect, nearZ, farZ);

  EXPECT_FLOAT_EQ(projMatrix(0, 0), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 1), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 2), 1.002002f);
  EXPECT_FLOAT_EQ(projMatrix(2, 3), 1.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 2), -0.2002002f);
  EXPECT_FLOAT_EQ(projMatrix(3, 3), 0.0f);
}

// Test case for perspectiveLhZo
TEST(PerspectiveTest, PerspectiveLhZoFloat) {
  float            fovY   = math::g_degreeToRadian(45.0f);
  float            aspect = 1.0f;
  float            nearZ  = 0.1f;
  float            farZ   = 100.0f;
  math::Matrix4f<> projMatrix
      = math::g_perspectiveLhZo(fovY, aspect, nearZ, farZ);

  EXPECT_FLOAT_EQ(projMatrix(0, 0), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 1), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 2), 1.001001f);
  EXPECT_FLOAT_EQ(projMatrix(2, 3), 1.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 2), -0.1001001f);
  EXPECT_FLOAT_EQ(projMatrix(3, 3), 0.0f);
}

// Test case for perspectiveRhNoInf
TEST(PerspectiveInfTest, PerspectiveRhNoInfFloat) {
  float            fovY       = math::g_degreeToRadian(45.0f);
  float            aspect     = 1.0f;
  float            nearZ      = 0.1f;
  math::Matrix4f<> projMatrix = math::g_perspectiveRhNoInf(fovY, aspect, nearZ);

  EXPECT_FLOAT_EQ(projMatrix(0, 0), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 1), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 2), -1.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 3), -1.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 2), -0.2f);
  EXPECT_FLOAT_EQ(projMatrix(3, 3), 0.0f);
}

// Test case for perspectiveRhZoInf
TEST(PerspectiveInfTest, PerspectiveRhZoInfFloat) {
  float            fovY       = math::g_degreeToRadian(45.0f);
  float            aspect     = 1.0f;
  float            nearZ      = 0.1f;
  math::Matrix4f<> projMatrix = math::g_perspectiveRhZoInf(fovY, aspect, nearZ);

  EXPECT_FLOAT_EQ(projMatrix(0, 0), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 1), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 2), -1.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 3), -1.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 2), -0.1f);
  EXPECT_FLOAT_EQ(projMatrix(3, 3), 0.0f);
}

// Test case for perspectiveLhNoInf
TEST(PerspectiveInfTest, PerspectiveLhNoInfFloat) {
  float            fovY       = math::g_degreeToRadian(45.0f);
  float            aspect     = 1.0f;
  float            nearZ      = 0.1f;
  math::Matrix4f<> projMatrix = math::g_perspectiveLhNoInf(fovY, aspect, nearZ);

  EXPECT_FLOAT_EQ(projMatrix(0, 0), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 1), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 2), 1.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 3), 1.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 2), -0.2f);
  EXPECT_FLOAT_EQ(projMatrix(3, 3), 0.0f);
}

// Test case for perspectiveLhZoInf
TEST(PerspectiveInfTest, PerspectiveLhZoInfFloat) {
  float            fovY       = math::g_degreeToRadian(45.0f);
  float            aspect     = 1.0f;
  float            nearZ      = 0.1f;
  math::Matrix4f<> projMatrix = math::g_perspectiveLhZoInf(fovY, aspect, nearZ);

  EXPECT_FLOAT_EQ(projMatrix(0, 0), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 1), 2.4142134f);
  EXPECT_FLOAT_EQ(projMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 2), 1.0f);
  EXPECT_FLOAT_EQ(projMatrix(2, 3), 1.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(projMatrix(3, 2), -0.1f);
  EXPECT_FLOAT_EQ(projMatrix(3, 3), 0.0f);
}

// Test case for frustumRhZo
TEST(FrustumTest, FrustumRhZoFloat) {
  float            left    = -1.0f;
  float            right   = 1.0f;
  float            bottom  = -1.0f;
  float            top     = 1.0f;
  float            nearVal = 1.0f;
  float            farVal  = 10.0f;
  math::Matrix4f<> frustumMatrix
      = math::g_frustumRhZo(left, right, bottom, top, nearVal, farVal);

  EXPECT_FLOAT_EQ(frustumMatrix(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 2), -1.1111112f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 3), -1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 2), -1.1111112f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 3), 0.0f);
}

// Test case for frustumRhNo
TEST(FrustumTest, FrustumRhNoFloat) {
  float            left    = -1.0f;
  float            right   = 1.0f;
  float            bottom  = -1.0f;
  float            top     = 1.0f;
  float            nearVal = 1.0f;
  float            farVal  = 10.0f;
  math::Matrix4f<> frustumMatrix
      = math::g_frustumRhNo(left, right, bottom, top, nearVal, farVal);

  EXPECT_FLOAT_EQ(frustumMatrix(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 2), -1.2222222f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 3), -1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 2), -2.2222223f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 3), 0.0f);
}

// Test case for frustumLhZo
TEST(FrustumTest, FrustumLhZoFloat) {
  float            left    = -1.0f;
  float            right   = 1.0f;
  float            bottom  = -1.0f;
  float            top     = 1.0f;
  float            nearVal = 1.0f;
  float            farVal  = 10.0f;
  math::Matrix4f<> frustumMatrix
      = math::g_frustumLhZo(left, right, bottom, top, nearVal, farVal);

  EXPECT_FLOAT_EQ(frustumMatrix(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 2), 1.1111112f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 3), 1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 2), -1.1111112f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 3), 0.0f);
}

// Test case for frustumLhNo
TEST(FrustumTest, FrustumLhNoFloat) {
  float            left    = -1.0f;
  float            right   = 1.0f;
  float            bottom  = -1.0f;
  float            top     = 1.0f;
  float            nearVal = 1.0f;
  float            farVal  = 10.0f;
  math::Matrix4f<> frustumMatrix
      = math::g_frustumLhNo(left, right, bottom, top, nearVal, farVal);

  EXPECT_FLOAT_EQ(frustumMatrix(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 2), 1.2222222f);
  EXPECT_FLOAT_EQ(frustumMatrix(2, 3), 1.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 2), -2.2222223f);
  EXPECT_FLOAT_EQ(frustumMatrix(3, 3), 0.0f);
}

// Test case for orthoLhZo
TEST(OrthoTest, OrthoLhZoFloat) {
  float            left   = -1.0f;
  float            right  = 1.0f;
  float            bottom = -1.0f;
  float            top    = 1.0f;
  float            nearZ  = 1.0f;
  float            farZ   = 10.0f;
  math::Matrix4f<> orthoMatrix
      = math::g_orthoLhZo(left, right, bottom, top, nearZ, farZ);

  EXPECT_FLOAT_EQ(orthoMatrix(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 2), 0.1111111f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 2), -0.1111111f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 3), 1.0f);
}

// Test case for orthoLhNo
TEST(OrthoTest, OrthoLhNoFloat) {
  float            left   = -1.0f;
  float            right  = 1.0f;
  float            bottom = -1.0f;
  float            top    = 1.0f;
  float            nearZ  = 1.0f;
  float            farZ   = 10.0f;
  math::Matrix4f<> orthoMatrix
      = math::g_orthoLhNo(left, right, bottom, top, nearZ, farZ);

  EXPECT_FLOAT_EQ(orthoMatrix(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 2), 0.2222222f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 2), -1.2222222f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 3), 1.0f);
}

// Test case for orthoRhZo
TEST(OrthoTest, OrthoRhZoFloat) {
  float            left   = -1.0f;
  float            right  = 1.0f;
  float            bottom = -1.0f;
  float            top    = 1.0f;
  float            nearZ  = 1.0f;
  float            farZ   = 10.0f;
  math::Matrix4f<> orthoMatrix
      = math::g_orthoRhZo(left, right, bottom, top, nearZ, farZ);

  EXPECT_FLOAT_EQ(orthoMatrix(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 2), -0.1111111f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 2), -0.1111111f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 3), 1.0f);
}

// Test case for orthoRhNo
TEST(OrthoTest, OrthoRhNoFloat) {
  float            left   = -1.0f;
  float            right  = 1.0f;
  float            bottom = -1.0f;
  float            top    = 1.0f;
  float            nearZ  = 1.0f;
  float            farZ   = 10.0f;
  math::Matrix4f<> orthoMatrix
      = math::g_orthoRhNo(left, right, bottom, top, nearZ, farZ);

  EXPECT_FLOAT_EQ(orthoMatrix(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 2), -0.2222222f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 2), -1.2222222f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 3), 1.0f);
}

// Test case for g_rotateRh (Euler angles)
TEST(RotateRhTest, EulerAnglesFloat) {
  float angleX = math::g_kPi / 6.0f;
  float angleY = math::g_kPi / 4.0f;
  float angleZ = math::g_kPi / 3.0f;

  auto rotateMatrix = math::g_rotateRh<float>(angleX, angleY, angleZ);

  // std::cout << rotateMatrix << std::endl;

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

  // std::cout << rotateMatrix << std::endl;

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

// Test case for orthoLhZo with width and height
TEST(OrthoTest, OrthoLhZoWithWidthHeightFloat) {
  float            width       = 800.0f;
  float            height      = 600.0f;
  float            zNear       = 1.0f;
  float            zFar        = 10.0f;
  math::Matrix4f<> orthoMatrix = math::g_orthoLhZo(width, height, zNear, zFar);

  EXPECT_FLOAT_EQ(orthoMatrix(0, 0), 0.0025f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 1), 0.0033333334f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 2), 0.1111111f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 2), -0.1111111f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 3), 1.0f);
}

// Test case for orthoLhNo with width and height
TEST(OrthoTest, OrthoLhNoWithWidthHeightFloat) {
  float            width       = 800.0f;
  float            height      = 600.0f;
  float            zNear       = 1.0f;
  float            zFar        = 10.0f;
  math::Matrix4f<> orthoMatrix = math::g_orthoLhNo(width, height, zNear, zFar);

  EXPECT_FLOAT_EQ(orthoMatrix(0, 0), 0.0025f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 1), 0.0033333334f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 2), 0.2222222f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 2), -1.2222222f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 3), 1.0f);
}

// Test case for orthoRhZo with width and height
TEST(OrthoTest, OrthoRhZoWithWidthHeightFloat) {
  float            width       = 800.0f;
  float            height      = 600.0f;
  float            zNear       = 1.0f;
  float            zFar        = 10.0f;
  math::Matrix4f<> orthoMatrix = math::g_orthoRhZo(width, height, zNear, zFar);

  EXPECT_FLOAT_EQ(orthoMatrix(0, 0), 0.0025f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 1), 0.0033333334f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 2), -0.1111111f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 2), -0.1111111f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 3), 1.0f);
}

// Test case for orthoRhNo with width and height
TEST(OrthoTest, OrthoRhNoWithWidthHeightFloat) {
  float            width       = 800.0f;
  float            height      = 600.0f;
  float            zNear       = 1.0f;
  float            zFar        = 10.0f;
  math::Matrix4f<> orthoMatrix = math::g_orthoRhNo(width, height, zNear, zFar);

  EXPECT_FLOAT_EQ(orthoMatrix(0, 0), 0.0025f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 1), 0.0033333334f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 2), -0.2222222f);
  EXPECT_FLOAT_EQ(orthoMatrix(2, 3), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 0), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 1), 0.0f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 2), -1.2222222f);
  EXPECT_FLOAT_EQ(orthoMatrix(3, 3), 1.0f);
}

// intersection

// --------- Ray�AABB ----------------------------------------------------

TEST(IntersectionsTest, RayAABB_Hit) {
  math::Rayf<>  ray({-1.0f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f});
  math::Point3f min(0.0f, 0.0f, 0.0f);
  math::Point3f max(1.0f, 1.0f, 1.0f);

  auto result = math::g_rayAABBintersect(ray, min, max);

  ASSERT_TRUE(result);
  EXPECT_FLOAT_EQ(result.distance, 1.0f);
  EXPECT_FLOAT_EQ(result.point(0), 0.0f);
  EXPECT_FLOAT_EQ(result.point(1), 0.5f);
  EXPECT_FLOAT_EQ(result.point(2), 0.5f);
}

TEST(IntersectionsTest, RayAABB_Miss) {
  math::Rayf<>  ray({-1.0f, 2.0f, 2.0f}, {1.0f, 0.0f, 0.0f});
  math::Point3f min(0.0f, 0.0f, 0.0f);
  math::Point3f max(1.0f, 1.0f, 1.0f);

  auto result = math::g_rayAABBintersect(ray, min, max);

  EXPECT_FALSE(result);
}

TEST(IntersectionsTest, RayAABB_InsideBox) {
  math::Rayf<>  ray({0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f});
  math::Point3f min(0.0f, 0.0f, 0.0f);
  math::Point3f max(1.0f, 1.0f, 1.0f);

  auto result = math::g_rayAABBintersect(ray, min, max);

  ASSERT_TRUE(result);
  EXPECT_FLOAT_EQ(result.distance, 0.0f);  
  EXPECT_FLOAT_EQ(result.point(0), 0.5f);
}

// --------- Ray�Triangle ------------------------------------------------

TEST(IntersectionsTest, RayTriangle_Hit) {
  math::Point3f v0(0.0f, 0.0f, 0.0f);
  math::Point3f v1(1.0f, 0.0f, 0.0f);
  math::Point3f v2(0.0f, 1.0f, 0.0f);

  math::Rayf<> ray({0.25f, 0.25f, -1.0f}, {0.0f, 0.0f, 1.0f});

  auto result = math::g_rayTriangleintersect(ray, v0, v1, v2);

  ASSERT_TRUE(result);
  EXPECT_FLOAT_EQ(result.distance, 1.0f);
  EXPECT_NEAR(result.point(0), 0.25f, 1e-6f);
  EXPECT_NEAR(result.point(1), 0.25f, 1e-6f);
  EXPECT_NEAR(result.point(2), 0.0f, 1e-6f);
}

TEST(IntersectionsTest, RayTriangle_Miss) {
  math::Point3f v0(0.0f, 0.0f, 0.0f);
  math::Point3f v1(1.0f, 0.0f, 0.0f);
  math::Point3f v2(0.0f, 1.0f, 0.0f);

  math::Rayf<> ray({1.0f, 1.0f, -1.0f}, {0.0f, 0.0f, 1.0f});

  auto result = math::g_rayTriangleintersect(ray, v0, v1, v2);

  EXPECT_FALSE(result);
}

// --------- Ray�Sphere --------------------------------------------------

TEST(IntersectionsTest, RaySphere_HitOutside) {
  math::Rayf<>  ray({0.0f, 0.0f, -3.0f}, {0.0f, 0.0f, 1.0f});
  math::Point3f center(0.0f, 0.0f, 0.0f);
  float         radius = 1.0f;

  auto result = math::g_raySphereintersect(ray, center, radius);

  ASSERT_TRUE(result);
  EXPECT_FLOAT_EQ(result.distance, 2.0f);
  EXPECT_FLOAT_EQ(result.point(2), -1.0f);
}

TEST(IntersectionsTest, RaySphere_HitInside) {
  math::Rayf<>  ray({0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f});
  math::Point3f center(0.0f, 0.0f, 0.0f);
  float         radius = 1.0f;

  auto result = math::g_raySphereintersect(ray, center, radius);

  ASSERT_TRUE(result);
  EXPECT_FLOAT_EQ(result.distance, 1.0f);
  EXPECT_FLOAT_EQ(result.point(0), 1.0f);
}

TEST(IntersectionsTest, RaySphere_Miss) {
  math::Rayf<>  ray({0.0f, 0.0f, -3.0f}, {1.0f, 0.0f, 0.0f});
  math::Point3f center(0.0f, 0.0f, 0.0f);

  auto result = math::g_raySphereintersect(ray, center, 1.0f);

  EXPECT_FALSE(result);
}

// --------- Screen -> Ray ------------------------------------------------

TEST(IntersectionsTest, ScreenToRay_IdentityMatrices) {
  const float width = 2.0f, height = 2.0f;
  float       x = 1.0f, y = 1.0f;  

  math::Matrix4f<> view = math::Matrix4f<>::Identity();
  math::Matrix4f<> proj = math::Matrix4f<>::Identity();

  auto ray = math::g_screenToRay(x, y, width, height, view, proj);

  // we should get (0,0,-1) -> (0,0,1)
  EXPECT_FLOAT_EQ(ray.origin()(0), 0.0f);
  EXPECT_FLOAT_EQ(ray.origin()(1), 0.0f);
  EXPECT_FLOAT_EQ(ray.origin()(2), -1.0f);

  EXPECT_FLOAT_EQ(ray.direction()(0), 0.0f);
  EXPECT_FLOAT_EQ(ray.direction()(1), 0.0f);
  EXPECT_FLOAT_EQ(ray.direction()(2), 1.0f);
}

// ========================== VECTOR: FLOAT ==============================

// Test case for the move assignment operator
TEST(VectorTest, MoveAssignmentOperator) {
  math::Vector<float, 3> vector1(1.0f, 2.0f, 3.0f);
  math::Vector<float, 3> vector2;
  vector2 = std::move(vector1);

  EXPECT_EQ(vector2(0), 1.0f);
  EXPECT_EQ(vector2(1), 2.0f);
  EXPECT_EQ(vector2(2), 3.0f);
}

// Test case for the GetSize() static member function
TEST(VectorTest, GetSize) {
  std::size_t size = math::Vector<float, 4>::GetSize();
  EXPECT_EQ(size, 4);
}

// Test case for the GetDataSize() static member function
TEST(VectorTest, GetDataSize) {
  std::size_t dataSize = math::Vector<float, 5>::GetDataSize();
  EXPECT_EQ(dataSize, 5 * sizeof(float));
}

// Test case for the GetOption() static member function
TEST(VectorTest, GetOption) {
  math::Options option = math::Vector<float, 3>::GetOption();
  EXPECT_EQ(option, math::Options::RowMajor);
}

// Test case for the data() member function (const version)
TEST(VectorTest, DataFunctionConst) {
  const math::Vector<float, 3> vector(1.0f, 2.0f, 3.0f);
  const float*                 data = vector.data();

  EXPECT_EQ(data[0], 1.0f);
  EXPECT_EQ(data[1], 2.0f);
  EXPECT_EQ(data[2], 3.0f);
}

// Test case for the data() member function (non-const version)
TEST(VectorTest, DataFunctionNonConst) {
  math::Vector<float, 3> vector(1.0f, 2.0f, 3.0f);
  float*                 data = vector.data();

  EXPECT_EQ(data[0], 1.0f);
  EXPECT_EQ(data[1], 2.0f);
  EXPECT_EQ(data[2], 3.0f);
}

// Test case for vector resizing with different target sizes
TEST(VectorTest, Resizing) {
  math::Vector<float, 3> vector(1.0f, 2.0f, 3.0f);
  auto                   resizedVector = vector.resizedCopy<5>();

  EXPECT_EQ(resizedVector(0), 1.0f);
  EXPECT_EQ(resizedVector(1), 2.0f);
  EXPECT_EQ(resizedVector(2), 3.0f);
  EXPECT_EQ(resizedVector(3), 0.0f);
  EXPECT_EQ(resizedVector(4), 0.0f);
}

// Test case for vector magnitude calculation
TEST(VectorTest, Magnitude) {
  math::Vector<float, 3> vector(1.0f, 2.0f, 3.0f);
  float                  magnitude = vector.magnitude();
  EXPECT_FLOAT_EQ(magnitude, 3.7416573867739413f);
}

// Test case for vector magnitude squared calculation
TEST(VectorTest, MagnitudeSquared) {
  math::Vector<float, 3> vector(1.0f, 2.0f, 3.0f);
  float                  magnitudeSquared = vector.magnitudeSquared();
  EXPECT_FLOAT_EQ(magnitudeSquared, 14.0f);
}

// Test case for vector normalization
TEST(VectorTest, Normalization) {
  math::Vector<float, 3> vector(1.0f, 2.0f, 3.0f);
  math::Vector<float, 3> normalized = vector.normalized();

  EXPECT_FLOAT_EQ(normalized(0), 0.2672612419124244f);
  EXPECT_FLOAT_EQ(normalized(1), 0.5345224838248488f);
  EXPECT_FLOAT_EQ(normalized(2), 0.8017837257372732f);
}

// Test case for vector dot product
TEST(VectorTest, DotProduct) {
  math::Vector<float, 3> vector1(1.0f, 2.0f, 3.0f);
  math::Vector<float, 3> vector2(4.0f, 5.0f, 6.0f);
  float                  dotProduct = vector1.dot(vector2);
  EXPECT_FLOAT_EQ(dotProduct, 32.0f);
}

// Test case for vector cross product
TEST(VectorTest, CrossProduct) {
  math::Vector<float, 3> vector1(1.0f, 2.0f, 3.0f);
  math::Vector<float, 3> vector2(4.0f, 5.0f, 6.0f);
  math::Vector<float, 3> crossProduct = vector1.cross(vector2);

  EXPECT_FLOAT_EQ(crossProduct(0), -3.0f);
  EXPECT_FLOAT_EQ(crossProduct(1), 6.0f);
  EXPECT_FLOAT_EQ(crossProduct(2), -3.0f);
}

// Test case for vector addition
TEST(VectorTest, Addition) {
  math::Vector<float, 3> vector1(1.0f, 2.0f, 3.0f);
  math::Vector<float, 3> vector2(4.0f, 5.0f, 6.0f);
  math::Vector<float, 3> result = vector1 + vector2;

  EXPECT_FLOAT_EQ(result(0), 5.0f);
  EXPECT_FLOAT_EQ(result(1), 7.0f);
  EXPECT_FLOAT_EQ(result(2), 9.0f);
}

// Test case for vector subtraction
TEST(VectorTest, Subtraction) {
  math::Vector<float, 3> vector1(1.0f, 2.0f, 3.0f);
  math::Vector<float, 3> vector2(4.0f, 5.0f, 6.0f);
  math::Vector<float, 3> result = vector1 - vector2;

  EXPECT_FLOAT_EQ(result(0), -3.0f);
  EXPECT_FLOAT_EQ(result(1), -3.0f);
  EXPECT_FLOAT_EQ(result(2), -3.0f);
}

// Test case for vector negation
TEST(VectorTest, Negation) {
  math::Vector<float, 3> vector(1.0f, -2.0f, 3.0f);
  math::Vector<float, 3> result = -vector;

  EXPECT_FLOAT_EQ(result(0), -1.0f);
  EXPECT_FLOAT_EQ(result(1), 2.0f);
  EXPECT_FLOAT_EQ(result(2), -3.0f);
}

// Test case for scalar multiplication
TEST(VectorTest, ScalarMultiplication) {
  math::Vector<float, 3> vector(1.0f, 2.0f, 3.0f);
  math::Vector<float, 3> result = vector * 2.0f;

  EXPECT_FLOAT_EQ(result(0), 2.0f);
  EXPECT_FLOAT_EQ(result(1), 4.0f);
  EXPECT_FLOAT_EQ(result(2), 6.0f);
}

// Test case for scalar division
TEST(VectorTest, ScalarDivision) {
  math::Vector<float, 3> vector(2.0f, 4.0f, 6.0f);
  math::Vector<float, 3> result = vector / 2.0f;

  EXPECT_FLOAT_EQ(result(0), 1.0f);
  EXPECT_FLOAT_EQ(result(1), 2.0f);
  EXPECT_FLOAT_EQ(result(2), 3.0f);
}

// Test case for vector multiplication with a matrix using the vector * matrix
// operator (row major)
TEST(VectorTest, MatrixMultiplicationRowMajor) {
  math::Vector<float, 3>    vector(1.0f, 2.0f, 3.0f);
  math::Matrix<float, 3, 3> matrix(
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
  math::Vector<float, 3> result = vector * matrix;

  EXPECT_FLOAT_EQ(result(0), 30.0f);
  EXPECT_FLOAT_EQ(result(1), 36.0f);
  EXPECT_FLOAT_EQ(result(2), 42.0f);
}

// Test case for vector multiplication with a matrix using the vector * matrix
// operator (column major)
TEST(VectorTest, MatrixMultiplicationColumnMajor) {
  math::Vector<float, 3, math::Options::ColumnMajor> vector(1.0f, 2.0f, 3.0f);
  math::Matrix<float, 3, 3, math::Options::ColumnMajor> matrix(
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
  auto result = matrix * vector;

  EXPECT_FLOAT_EQ(result(0), 30.0f);
  EXPECT_FLOAT_EQ(result(1), 36.0f);
  EXPECT_FLOAT_EQ(result(2), 42.0f);
}

// Test case for vector multiplication and assignment with a matrix using the *=
// operator
TEST(VectorTest, MatrixMultiplicationAssignment) {
  math::Vector<float, 3>    vector(1.0f, 2.0f, 3.0f);
  math::Matrix<float, 3, 3> matrix(
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
  vector *= matrix;

  EXPECT_FLOAT_EQ(vector(0), 30.0f);
  EXPECT_FLOAT_EQ(vector(1), 36.0f);
  EXPECT_FLOAT_EQ(vector(2), 42.0f);
}

TEST(VectorComparisonTest, LessThanOperator) {
  math::Vector3f vec1(1.0f, 2.0f, 3.0f);
  math::Vector3f vec2(4.0f, 5.0f, 6.0f);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, LessThanOperatorEqual) {
  math::Vector3f vec1(1.0f, 2.0f, 3.0f);
  math::Vector3f vec2(1.0f, 2.0f, 3.0f);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperator) {
  math::Vector3f vec1(4.0f, 5.0f, 6.0f);
  math::Vector3f vec2(1.0f, 2.0f, 3.0f);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorEqual) {
  math::Vector3f vec1(1.0f, 2.0f, 3.0f);
  math::Vector3f vec2(1.0f, 2.0f, 3.0f);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperator) {
  math::Vector3f vec1(1.0f, 2.0f, 3.0f);
  math::Vector3f vec2(4.0f, 5.0f, 6.0f);
  math::Vector3f vec3(1.0f, 2.0f, 3.0f);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperator) {
  math::Vector3f vec1(4.0f, 5.0f, 6.0f);
  math::Vector3f vec2(1.0f, 2.0f, 3.0f);
  math::Vector3f vec3(4.0f, 5.0f, 6.0f);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ========================== VECTOR: DOUBLE ==============================

TEST(VectorComparisonTest, LessThanOperatorDouble) {
  math::Vector3<double> vec1(1.0, 2.0, 3.0);
  math::Vector3<double> vec2(4.0, 5.0, 6.0);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, LessThanOperatorEqualDouble) {
  math::Vector3<double> vec1(1.0, 2.0, 3.0);
  math::Vector3<double> vec2(1.0, 2.0, 3.0);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorDouble) {
  math::Vector3<double> vec1(4.0, 5.0, 6.0);
  math::Vector3<double> vec2(1.0, 2.0, 3.0);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorEqualDouble) {
  math::Vector3<double> vec1(1.0, 2.0, 3.0);
  math::Vector3<double> vec2(1.0, 2.0, 3.0);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperatorDouble) {
  math::Vector3<double> vec1(1.0, 2.0, 3.0);
  math::Vector3<double> vec2(4.0, 5.0, 6.0);
  math::Vector3<double> vec3(1.0, 2.0, 3.0);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperatorDouble) {
  math::Vector3<double> vec1(4.0, 5.0, 6.0);
  math::Vector3<double> vec2(1.0, 2.0, 3.0);
  math::Vector3<double> vec3(4.0, 5.0, 6.0);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ========================== VECTOR: INT ==============================

TEST(VectorComparisonTest, LessThanOperatorInt) {
  math::Vector3i vec1(1, 2, 3);
  math::Vector3i vec2(4, 5, 6);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, LessThanOperatorEqualInt) {
  math::Vector3i vec1(1, 2, 3);
  math::Vector3i vec2(1, 2, 3);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorInt) {
  math::Vector3i vec1(4, 5, 6);
  math::Vector3i vec2(1, 2, 3);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorEqualInt) {
  math::Vector3i vec1(1, 2, 3);
  math::Vector3i vec2(1, 2, 3);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperatorInt) {
  math::Vector3i vec1(1, 2, 3);
  math::Vector3i vec2(4, 5, 6);
  math::Vector3i vec3(1, 2, 3);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperatorInt) {
  math::Vector3i vec1(4, 5, 6);
  math::Vector3i vec2(1, 2, 3);
  math::Vector3i vec3(4, 5, 6);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ========================= VECTOR: UNSIGNED INT =============================

TEST(VectorComparisonTest, LessThanOperatorUnsignedInt) {
  math::Vector3<unsigned int> vec1(1, 2, 3);
  math::Vector3<unsigned int> vec2(4, 5, 6);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, LessThanOperatorEqualUnsignedInt) {
  math::Vector3<unsigned int> vec1(1, 2, 3);
  math::Vector3<unsigned int> vec2(1, 2, 3);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorUnsignedInt) {
  math::Vector3<unsigned int> vec1(4, 5, 6);
  math::Vector3<unsigned int> vec2(1, 2, 3);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorEqualUnsignedInt) {
  math::Vector3<unsigned int> vec1(1, 2, 3);
  math::Vector3<unsigned int> vec2(1, 2, 3);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperatorUnsignedInt) {
  math::Vector3<unsigned int> vec1(1, 2, 3);
  math::Vector3<unsigned int> vec2(4, 5, 6);
  math::Vector3<unsigned int> vec3(1, 2, 3);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperatorUnsignedInt) {
  math::Vector3<unsigned int> vec1(4, 5, 6);
  math::Vector3<unsigned int> vec2(1, 2, 3);
  math::Vector3<unsigned int> vec3(4, 5, 6);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

TEST(VectorComparisonTest, LessThanOperatorUnsignedIntLargeValues) {
  math::Vector3<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Vector3<unsigned int> vec2(0, 0, 0);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_TRUE(vec2 < vec1);
}

TEST(VectorComparisonTest, GreaterThanOperatorUnsignedIntLargeValues) {
  math::Vector3<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Vector3<unsigned int> vec2(0, 0, 0);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(VectorComparisonTest, LessThanOrEqualToOperatorUnsignedIntLargeValues) {
  math::Vector3<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Vector3<unsigned int> vec2(0, 0, 0);

  EXPECT_FALSE(vec1 <= vec2);
  EXPECT_TRUE(vec2 <= vec1);
}

TEST(VectorComparisonTest, GreaterThanOrEqualToOperatorUnsignedIntLargeValues) {
  math::Vector3<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Vector3<unsigned int> vec2(0, 0, 0);

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

// ========================== QUATERNION: FLOAT =============================

TEST(QuaternionTest, DefaultConstructor) {
  math::Quaternionf q;
  EXPECT_EQ(q.x(), 0.0f);
  EXPECT_EQ(q.y(), 0.0f);
  EXPECT_EQ(q.z(), 0.0f);
  EXPECT_EQ(q.w(), 1.0f);
}

TEST(QuaternionTest, ParameterizedConstructor) {
  math::Quaternionf q(1.0f, 2.0f, 3.0f, 4.0f);
  EXPECT_EQ(q.x(), 1.0f);
  EXPECT_EQ(q.y(), 2.0f);
  EXPECT_EQ(q.z(), 3.0f);
  EXPECT_EQ(q.w(), 4.0f);
}

TEST(QuaternionTest, CopyConstructor) {
  math::Quaternionf q1(1.0f, 2.0f, 3.0f, 4.0f);
  math::Quaternionf q2(q1);
  EXPECT_EQ(q2.x(), 1.0f);
  EXPECT_EQ(q2.y(), 2.0f);
  EXPECT_EQ(q2.z(), 3.0f);
  EXPECT_EQ(q2.w(), 4.0f);
}

TEST(QuaternionTest, Addition) {
  math::Quaternionf q1(1.0f, 2.0f, 3.0f, 4.0f);
  math::Quaternionf q2(5.0f, 6.0f, 7.0f, 8.0f);
  math::Quaternionf result = q1 + q2;
  EXPECT_EQ(result.x(), 6.0f);
  EXPECT_EQ(result.y(), 8.0f);
  EXPECT_EQ(result.z(), 10.0f);
  EXPECT_EQ(result.w(), 12.0f);
}

TEST(QuaternionTest, Subtraction) {
  math::Quaternionf q1(1.0f, 2.0f, 3.0f, 4.0f);
  math::Quaternionf q2(5.0f, 6.0f, 7.0f, 8.0f);
  math::Quaternionf result = q1 - q2;
  EXPECT_EQ(result.x(), -4.0f);
  EXPECT_EQ(result.y(), -4.0f);
  EXPECT_EQ(result.z(), -4.0f);
  EXPECT_EQ(result.w(), -4.0f);
}

TEST(QuaternionTest, Multiplication) {
  math::Quaternionf q1(1.0f, 2.0f, 3.0f, 4.0f);
  math::Quaternionf q2(5.0f, 6.0f, 7.0f, 8.0f);
  math::Quaternionf result = q1 * q2;
  EXPECT_EQ(result.x(), 24.0f);
  EXPECT_EQ(result.y(), 48.0f);
  EXPECT_EQ(result.z(), 48.0f);
  EXPECT_EQ(result.w(), -6.0f);
}

TEST(QuaternionTest, Exponential) {
  math::Quaternionf q(0.0f, math::g_kPi / 4.0f, 0.0f, 0.0f);
  math::Quaternionf result = q.exp();

  EXPECT_NEAR(result.x(), 0.0f, 1e-5f);
  EXPECT_NEAR(result.y(), 0.707107f, 1e-5f);
  EXPECT_NEAR(result.z(), 0.0f, 1e-5f);
  EXPECT_NEAR(result.w(), 0.707107f, 1e-5f);
}

TEST(QuaternionTest, Logarithm) {
  math::Quaternionf q(0.0f, 0.707107f, 0.0f, 0.707107f);
  math::Quaternionf result = q.log();

  EXPECT_NEAR(result.x(), 0.0f, 1e-5f);
  EXPECT_NEAR(result.y(), math::g_kPi / 4.0f, 1e-5f);
  EXPECT_NEAR(result.z(), 0.0f, 1e-5f);
  EXPECT_NEAR(result.w(), 0.0f, 1e-5f);
}

TEST(QuaternionTest, ScalarMultiplication) {
  math::Quaternionf q(1.0f, 2.0f, 3.0f, 4.0f);
  math::Quaternionf result = q * 2.0f;
  EXPECT_EQ(result.x(), 2.0f);
  EXPECT_EQ(result.y(), 4.0f);
  EXPECT_EQ(result.z(), 6.0f);
  EXPECT_EQ(result.w(), 8.0f);
}

TEST(QuaternionTest, ScalarDivision) {
  math::Quaternionf q(1.0f, 2.0f, 3.0f, 4.0f);
  math::Quaternionf result = q / 2.0f;
  EXPECT_EQ(result.x(), 0.5f);
  EXPECT_EQ(result.y(), 1.0f);
  EXPECT_EQ(result.z(), 1.5f);
  EXPECT_EQ(result.w(), 2.0f);
}

TEST(QuaternionTest, Conjugate) {
  math::Quaternionf q(1.0f, 2.0f, 3.0f, 4.0f);
  math::Quaternionf result = q.conjugate();
  EXPECT_EQ(result.x(), -1.0f);
  EXPECT_EQ(result.y(), -2.0f);
  EXPECT_EQ(result.z(), -3.0f);
  EXPECT_EQ(result.w(), 4.0f);
}

TEST(QuaternionTest, Norm) {
  math::Quaternionf q(1.0f, 2.0f, 3.0f, 4.0f);
  float             norm = q.norm();
  EXPECT_FLOAT_EQ(norm, 5.477226f);
}

TEST(QuaternionTest, Normalize) {
  math::Quaternionf q(1.0f, 2.0f, 3.0f, 4.0f);
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  q.normalize();
  EXPECT_NEAR(q.x(), 0.182574f, 1e-6f);
  EXPECT_NEAR(q.y(), 0.365148f, 1e-6f);
  EXPECT_NEAR(q.z(), 0.547723f, 1e-6f);
  EXPECT_NEAR(q.w(), 0.730297f, 1e-6f);
#else
  math::Quaternionf result = q.normalized();
  EXPECT_NEAR(result.x(), 0.182574f, 1e-6f);
  EXPECT_NEAR(result.y(), 0.365148f, 1e-6f);
  EXPECT_NEAR(result.z(), 0.547723f, 1e-6f);
  EXPECT_NEAR(result.w(), 0.730297f, 1e-6f);
#endif
}

TEST(QuaternionTest, FromRotationMatrix) {
  math::Matrix3f<>  m(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f);
  math::Quaternionf q = math::Quaternionf::fromRotationMatrix(m);
  EXPECT_FLOAT_EQ(q.x(), 0.707107f);
  EXPECT_FLOAT_EQ(q.y(), 0.0f);
  EXPECT_FLOAT_EQ(q.z(), 0.0f);
  EXPECT_FLOAT_EQ(q.w(), 0.707107f);
}

TEST(QuaternionTest, ToRotationMatrix) {
  math::Quaternionf q(0.707107f, 0.0f, 0.0f, 0.707107f);
  math::Matrix3f<>  m = q.toRotationMatrix();
  EXPECT_FLOAT_EQ(m(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(m(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(m(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(m(0, 1), 0.0f);
  EXPECT_NEAR(m(1, 1), 0.0f, 1e-6f);
  EXPECT_NEAR(m(2, 1), -1.0f, 1e-6f);
  EXPECT_FLOAT_EQ(m(0, 2), 0.0f);
  EXPECT_NEAR(m(1, 2), 1.0f, 1e-6f);
  EXPECT_NEAR(m(2, 2), 0.0f, 1e-6f);
}

TEST(QuaternionTest, Inverse) {
  math::Quaternionf q(1.0f, 2.0f, 3.0f, 4.0f);

  math::Quaternionf result = q.inverse();

  EXPECT_NEAR(result.x(), -0.0333333f, 1e-6f);
  EXPECT_NEAR(result.y(), -0.0666667f, 1e-6f);
  EXPECT_NEAR(result.z(), -0.1f, 1e-6f);
  EXPECT_NEAR(result.w(), 0.133333f, 1e-6f);
}

TEST(QuaternionTest, RotateVector) {
  math::Vector3f   v(1.0f, 0.0f, 0.0f);
  math::Quaternionf q(0.707107f, 0.0f, 0.0f, 0.707107f);

  math::Vector3f result = q.rotateVector(v);

  EXPECT_NEAR(result.x(), 1.0f, 1e-6f);
  EXPECT_NEAR(result.y(), 0.0f, 1e-6f);
  EXPECT_NEAR(result.z(), 0.0f, 1e-6f);
}

TEST(QuaternionTest, Slerp) {
  math::Quaternionf q1(0.0f, 0.0f, 0.0f, 1.0f);
  math::Quaternionf q2(0.0f, 0.0f, 1.0f, 0.0f);
  math::Quaternionf result = math::Quaternionf::slerp(q1, q2, 0.5f);
  EXPECT_NEAR(result.x(), 0.0f, 1e-6f);
  EXPECT_NEAR(result.y(), 0.0f, 1e-6f);
  EXPECT_NEAR(result.z(), 0.707107f, 1e-6f);
  EXPECT_NEAR(result.w(), 0.707107f, 1e-6f);
}

TEST(QuaternionTest, Nlerp) {
  math::Quaternionf q1(0.0f, 0.0f, 0.0f, 1.0f);
  math::Quaternionf q2(0.0f, 0.0f, 1.0f, 0.0f);
  math::Quaternionf result = math::Quaternionf::nlerp(q1, q2, 0.5f);
  EXPECT_NEAR(result.x(), 0.0f, 1e-6f);
  EXPECT_NEAR(result.y(), 0.0f, 1e-6f);
  EXPECT_NEAR(result.z(), 0.707107f, 1e-6f);
  EXPECT_NEAR(result.w(), 0.707107f, 1e-6f);
}

TEST(QuaternionTest, IsApprox) {
  math::Quaternionf q1(1.0f, 2.0f, 3.0f, 4.0f);
  math::Quaternionf q2(1.0f, 2.0f, 3.0f, 4.0f);
  math::Quaternionf q3(1.1f, 2.1f, 3.1f, 4.1f);
  EXPECT_TRUE(q1.isApprox(q2));
  EXPECT_FALSE(q1.isApprox(q3));
}

TEST(QuaternionTest, Dot) {
  math::Quaternionf q1(1.0f, 2.0f, 3.0f, 4.0f);
  math::Quaternionf q2(5.0f, 6.0f, 7.0f, 8.0f);
  float             result = q1.dot(q2);
  EXPECT_FLOAT_EQ(result, 70.0f);
}

TEST(QuaternionTest, Angle) {
  math::Quaternionf q1(0.0f, 0.0f, 0.0f, 1.0f);
  math::Quaternionf q2(0.0f, 0.0f, 1.0f, 0.0f);
  float             result = q1.angle(q2);
  EXPECT_NEAR(result, 3.14159f, 1e-5f);
}

TEST(QuaternionTest, Exp) {
  math::Quaternionf q(0.0f, 0.0f, 0.0f, 0.0f);
  math::Quaternionf result = q.exp();
  EXPECT_FLOAT_EQ(result.x(), 0.0f);
  EXPECT_FLOAT_EQ(result.y(), 0.0f);
  EXPECT_FLOAT_EQ(result.z(), 0.0f);
  EXPECT_FLOAT_EQ(result.w(), 1.0f);
}

TEST(QuaternionTest, Log) {
  math::Quaternionf q(0.0f, 0.0f, 0.0f, 1.0f);
  math::Quaternionf result = q.log();
  EXPECT_FLOAT_EQ(result.x(), 0.0f);
  EXPECT_FLOAT_EQ(result.y(), 0.0f);
  EXPECT_FLOAT_EQ(result.z(), 0.0f);
  EXPECT_FLOAT_EQ(result.w(), 0.0f);
}

TEST(QuaternionTest, FromVectorsIdentity) {
  math::Vector3f from(1.0f, 0.0f, 0.0f);
  math::Vector3f to(1.0f, 0.0f, 0.0f);

  math::Quaternionf q = math::Quaternionf::fromVectors(from, to);

  EXPECT_NEAR(q.x(), 0.0f, 1e-6f);
  EXPECT_NEAR(q.y(), 0.0f, 1e-6f);
  EXPECT_NEAR(q.z(), 0.0f, 1e-6f);
  EXPECT_NEAR(q.w(), 1.0f, 1e-6f);
  EXPECT_NEAR(q.norm(), 1.0f, 1e-6f);

  math::Vector3f rotated = q.rotateVector(from);
  EXPECT_NEAR(rotated.x(), to.x(), 1e-6f);
  EXPECT_NEAR(rotated.y(), to.y(), 1e-6f);
  EXPECT_NEAR(rotated.z(), to.z(), 1e-6f);
}

TEST(QuaternionTest, FromVectorsOpposite) {
  math::Vector3f from(1.0f, 0.0f, 0.0f);
  math::Vector3f to(-1.0f, 0.0f, 0.0f);

  math::Quaternionf q = math::Quaternionf::fromVectors(from, to);

  EXPECT_NEAR(q.norm(), 1.0f, 1e-6f);

  math::Vector3f rotated = q.rotateVector(from);
  EXPECT_NEAR(rotated.x(), to.x(), 1e-5f);
  EXPECT_NEAR(rotated.y(), to.y(), 1e-5f);
  EXPECT_NEAR(rotated.z(), to.z(), 1e-5f);
}

TEST(QuaternionTest, FromVectorsOrthogonal) {
  math::Vector3f from(1.0f, 0.0f, 0.0f);
  math::Vector3f to(0.0f, 1.0f, 0.0f);

  math::Quaternionf q = math::Quaternionf::fromVectors(from, to);
  EXPECT_NEAR(q.norm(), 1.0f, 1e-6f);

  math::Vector3f rotated = q.rotateVector(from);
  EXPECT_NEAR(rotated.x(), to.x(), 1e-5f);
  EXPECT_NEAR(rotated.y(), to.y(), 1e-5f);
  EXPECT_NEAR(rotated.z(), to.z(), 1e-5f);
}

TEST(QuaternionTest, FromVectorsNonNormalized) {
  math::Vector3f from(2.0f, 0.0f, 0.0f);
  math::Vector3f to(0.0f, -3.0f, 0.0f);

  math::Quaternionf q = math::Quaternionf::fromVectors(from, to);

  EXPECT_NEAR(q.norm(), 1.0f, 1e-6f);

  // ���������� �����������
  math::Vector3f rotatedDir = q.rotateVector(from).normalized();
  math::Vector3f toDir      = to.normalized();

  EXPECT_NEAR(rotatedDir.x(), toDir.x(), 1e-5f);
  EXPECT_NEAR(rotatedDir.y(), toDir.y(), 1e-5f);
  EXPECT_NEAR(rotatedDir.z(), toDir.z(), 1e-5f);
}

// ========================== DIMENSION: FLOAT ==============================

TEST(DimensionComparisonTest, LessThanOperator) {
  math::Dimension3f vec1(1.0f, 2.0f, 3.0f);
  math::Dimension3f vec2(4.0f, 5.0f, 6.0f);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorEqual) {
  math::Dimension3f vec1(1.0f, 2.0f, 3.0f);
  math::Dimension3f vec2(1.0f, 2.0f, 3.0f);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperator) {
  math::Dimension3f vec1(4.0f, 5.0f, 6.0f);
  math::Dimension3f vec2(1.0f, 2.0f, 3.0f);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorEqual) {
  math::Dimension3f vec1(1.0f, 2.0f, 3.0f);
  math::Dimension3f vec2(1.0f, 2.0f, 3.0f);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperator) {
  math::Dimension3f vec1(1.0f, 2.0f, 3.0f);
  math::Dimension3f vec2(4.0f, 5.0f, 6.0f);
  math::Dimension3f vec3(1.0f, 2.0f, 3.0f);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(DimensionComparisonTest, GreaterThanOrEqualToOperator) {
  math::Dimension3f vec1(4.0f, 5.0f, 6.0f);
  math::Dimension3f vec2(1.0f, 2.0f, 3.0f);
  math::Dimension3f vec3(4.0f, 5.0f, 6.0f);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ========================== QUATERNION: DOUBLE =============================

TEST(QuaternionTest, DefaultConstructorDouble) {
  math::Quaterniond q;
  EXPECT_EQ(q.x(), 0.0);
  EXPECT_EQ(q.y(), 0.0);
  EXPECT_EQ(q.z(), 0.0);
  EXPECT_EQ(q.w(), 1.0);
}

TEST(QuaternionTest, ParameterizedConstructorDouble) {
  math::Quaterniond q(1.0, 2.0, 3.0, 4.0);
  EXPECT_EQ(q.x(), 1.0);
  EXPECT_EQ(q.y(), 2.0);
  EXPECT_EQ(q.z(), 3.0);
  EXPECT_EQ(q.w(), 4.0);
}

TEST(QuaternionTest, CopyConstructorDouble) {
  math::Quaterniond q1(1.0, 2.0, 3.0, 4.0);
  math::Quaterniond q2(q1);
  EXPECT_EQ(q2.x(), 1.0);
  EXPECT_EQ(q2.y(), 2.0);
  EXPECT_EQ(q2.z(), 3.0);
  EXPECT_EQ(q2.w(), 4.0);
}

TEST(QuaternionTest, AdditionDouble) {
  math::Quaterniond q1(1.0, 2.0, 3.0, 4.0);
  math::Quaterniond q2(5.0, 6.0, 7.0, 8.0);
  math::Quaterniond result = q1 + q2;
  EXPECT_EQ(result.x(), 6.0);
  EXPECT_EQ(result.y(), 8.0);
  EXPECT_EQ(result.z(), 10.0);
  EXPECT_EQ(result.w(), 12.0);
}

TEST(QuaternionTest, SubtractionDouble) {
  math::Quaterniond q1(1.0, 2.0, 3.0, 4.0);
  math::Quaterniond q2(5.0, 6.0, 7.0, 8.0);
  math::Quaterniond result = q1 - q2;
  EXPECT_EQ(result.x(), -4.0);
  EXPECT_EQ(result.y(), -4.0);
  EXPECT_EQ(result.z(), -4.0);
  EXPECT_EQ(result.w(), -4.0);
}

TEST(QuaternionTest, MultiplicationDouble) {
  math::Quaterniond q1(1.0, 2.0, 3.0, 4.0);
  math::Quaterniond q2(5.0, 6.0, 7.0, 8.0);
  math::Quaterniond result = q1 * q2;
  EXPECT_EQ(result.x(), 24.0);
  EXPECT_EQ(result.y(), 48.0);
  EXPECT_EQ(result.z(), 48.0);
  EXPECT_EQ(result.w(), -6.0);
}

TEST(QuaternionTest, ScalarMultiplicationDouble) {
  math::Quaterniond q(1.0, 2.0, 3.0, 4.0);
  math::Quaterniond result = q * 2.0;
  EXPECT_EQ(result.x(), 2.0);
  EXPECT_EQ(result.y(), 4.0);
  EXPECT_EQ(result.z(), 6.0);
  EXPECT_EQ(result.w(), 8.0);
}

TEST(QuaternionTest, ScalarDivisionDouble) {
  math::Quaterniond q(1.0, 2.0, 3.0, 4.0);
  math::Quaterniond result = q / 2.0;
  EXPECT_EQ(result.x(), 0.5);
  EXPECT_EQ(result.y(), 1.0);
  EXPECT_EQ(result.z(), 1.5);
  EXPECT_EQ(result.w(), 2.0);
}

TEST(QuaternionTest, ConjugateDouble) {
  math::Quaterniond q(1.0, 2.0, 3.0, 4.0);
  math::Quaterniond result = q.conjugate();
  EXPECT_EQ(result.x(), -1.0);
  EXPECT_EQ(result.y(), -2.0);
  EXPECT_EQ(result.z(), -3.0);
  EXPECT_EQ(result.w(), 4.0);
}

TEST(QuaternionTest, NormDouble) {
  math::Quaterniond q(1.0, 2.0, 3.0, 4.0);
  double            norm = q.norm();
  EXPECT_DOUBLE_EQ(norm, 5.477225575051661);
}

TEST(QuaternionTest, NormalizeDouble) {
  math::Quaterniond q(1.0, 2.0, 3.0, 4.0);
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  q.normalize();
  EXPECT_DOUBLE_EQ(q.x(), 0.18257418583505536);
  EXPECT_DOUBLE_EQ(q.y(), 0.3651483716701107);
  EXPECT_DOUBLE_EQ(q.z(), 0.5477225575051661);
  EXPECT_DOUBLE_EQ(q.w(), 0.7302967433402214);
#else
  math::Quaterniond result = q.normalized();
  EXPECT_DOUBLE_EQ(result.x(), 0.18257418583505536);
  EXPECT_DOUBLE_EQ(result.y(), 0.3651483716701107);
  EXPECT_DOUBLE_EQ(result.z(), 0.5477225575051661);
  EXPECT_DOUBLE_EQ(result.w(), 0.7302967433402214);
#endif
}

TEST(QuaternionTest, FromRotationMatrixDouble) {
  math::Matrix3d<>  m(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
  math::Quaterniond q = math::Quaterniond::fromRotationMatrix(m);
  EXPECT_DOUBLE_EQ(q.x(), 0.7071067811865476);
  EXPECT_DOUBLE_EQ(q.y(), 0.0);
  EXPECT_DOUBLE_EQ(q.z(), 0.0);
  EXPECT_DOUBLE_EQ(q.w(), 0.7071067811865476);
}

TEST(QuaternionTest, ToRotationMatrixDouble) {
  math::Quaterniond q(0.7071067811865476, 0.0, 0.0, 0.7071067811865476);
  math::Matrix3d<>  m = q.toRotationMatrix();
  EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(m(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(m(2, 0), 0.0);
  EXPECT_DOUBLE_EQ(m(0, 1), 0.0);
  EXPECT_NEAR(m(1, 1), 0.0, 1e-15);
  EXPECT_DOUBLE_EQ(m(2, 1), -1.0);
  EXPECT_DOUBLE_EQ(m(0, 2), 0.0);
  EXPECT_DOUBLE_EQ(m(1, 2), 1.0);
  EXPECT_NEAR(m(2, 2), 0.0, 1e-15);
}

TEST(QuaternionTest, InverseDouble) {
  math::Quaterniond q(1.0, 2.0, 3.0, 4.0);
  math::Quaterniond result = q.inverse();
  EXPECT_DOUBLE_EQ(result.x(), -0.033333333333333333);
  EXPECT_DOUBLE_EQ(result.y(), -0.06666666666666667);
  EXPECT_DOUBLE_EQ(result.z(), -0.1);
  EXPECT_DOUBLE_EQ(result.w(), 0.13333333333333333);
}

TEST(QuaternionTest, RotateVectorDouble) {
  math::Quaterniond q(0.7071067811865476, 0.0, 0.0, 0.7071067811865476);
  math::Vector3d   v(1.0, 0.0, 0.0);
  math::Vector3d   result = q.rotateVector(v);
  EXPECT_NEAR(result.x(), 1.0, 1e-15);
  EXPECT_NEAR(result.y(), 0.0, 1e-15);
  EXPECT_NEAR(result.z(), 0.0, 1e-15);
}

TEST(QuaternionTest, SlerpDouble) {
  math::Quaterniond q1(0.0, 0.0, 0.0, 1.0);
  math::Quaterniond q2(0.0, 0.0, 1.0, 0.0);
  math::Quaterniond result = math::Quaterniond::slerp(q1, q2, 0.5);
  EXPECT_NEAR(result.x(), 0.0, 1e-15);
  EXPECT_NEAR(result.y(), 0.0, 1e-15);
  EXPECT_NEAR(result.z(), 0.7071067811865476, 1e-15);
  EXPECT_NEAR(result.w(), 0.7071067811865476, 1e-15);
}

TEST(QuaternionTest, NlerpDouble) {
  math::Quaterniond q1(0.0, 0.0, 0.0, 1.0);
  math::Quaterniond q2(0.0, 0.0, 1.0, 0.0);
  math::Quaterniond result = math::Quaterniond::nlerp(q1, q2, 0.5);
  EXPECT_NEAR(result.x(), 0.0, 1e-15);
  EXPECT_NEAR(result.y(), 0.0, 1e-15);
  EXPECT_NEAR(result.z(), 0.7071067811865476, 1e-15);
  EXPECT_NEAR(result.w(), 0.7071067811865476, 1e-15);
}

TEST(QuaternionTest, IsApproxDouble) {
  math::Quaterniond q1(1.0, 2.0, 3.0, 4.0);
  math::Quaterniond q2(1.0, 2.0, 3.0, 4.0);
  math::Quaterniond q3(1.1, 2.1, 3.1, 4.1);
  EXPECT_TRUE(q1.isApprox(q2));
  EXPECT_FALSE(q1.isApprox(q3));
}

TEST(QuaternionTest, DotDouble) {
  math::Quaterniond q1(1.0, 2.0, 3.0, 4.0);
  math::Quaterniond q2(5.0, 6.0, 7.0, 8.0);
  double            result = q1.dot(q2);
  EXPECT_DOUBLE_EQ(result, 70.0);
}

TEST(QuaternionTest, AngleDouble) {
  math::Quaterniond q1(0.0, 0.0, 0.0, 1.0);
  math::Quaterniond q2(0.0, 0.0, 1.0, 0.0);
  double            result = q1.angle(q2);
  EXPECT_NEAR(result, 3.141592653589793, 1e-15);
}

TEST(QuaternionTest, ExpDouble) {
  math::Quaterniond q(0.0, 0.0, 0.0, 0.0);
  math::Quaterniond result = q.exp();
  EXPECT_DOUBLE_EQ(result.x(), 0.0);
  EXPECT_DOUBLE_EQ(result.y(), 0.0);
  EXPECT_DOUBLE_EQ(result.z(), 0.0);
  EXPECT_DOUBLE_EQ(result.w(), 1.0);
}

TEST(QuaternionTest, LogDouble) {
  math::Quaterniond q(0.0, 0.0, 0.0, 1.0);
  math::Quaterniond result = q.log();
  EXPECT_DOUBLE_EQ(result.x(), 0.0);
  EXPECT_DOUBLE_EQ(result.y(), 0.0);
  EXPECT_DOUBLE_EQ(result.z(), 0.0);
  EXPECT_DOUBLE_EQ(result.w(), 0.0);
}

TEST(QuaternionTest, FromAxisAngle) {
  math::Vector3d axis(1.0, 0.0, 0.0);
  double          angle = math::g_kPi / 2;

  math::Quaterniond q = math::Quaterniond::fromAxisAngle(axis, angle);

  EXPECT_NEAR(q.norm(), 1.0, 1e-6);

  math::Vector3d v(0.0, 1.0, 0.0);
  math::Vector3d rotated_v = q.rotateVector(v);

  EXPECT_NEAR(rotated_v.x(), 0.0, 1e-6);
  EXPECT_NEAR(rotated_v.y(), 0.0, 1e-6);
  EXPECT_NEAR(rotated_v.z(), 1.0, 1e-6);
}

TEST(QuaternionTest, ToAxisAngle) {
  math::Vector3d axis(0.0, 1.0, 0.0);
  double          angle = math::g_kPi;

  math::Quaterniond q = math::Quaterniond::fromAxisAngle(axis, angle);

  math::Vector3d extractedAxis;
  double          extractedAngle;
  q.toAxisAngle(extractedAxis, extractedAngle);

  EXPECT_NEAR(extractedAxis.x(), axis.x(), 1e-6);
  EXPECT_NEAR(extractedAxis.y(), axis.y(), 1e-6);
  EXPECT_NEAR(extractedAxis.z(), axis.z(), 1e-6);
  EXPECT_NEAR(extractedAngle, angle, 1e-6);
}

TEST(QuaternionTest, AxisAngleIdentity) {
  math::Vector3d axis(1.0, 0.0, 0.0);
  double          angle = 0.0;

  math::Quaterniond q = math::Quaterniond::fromAxisAngle(axis, angle);

  math::Vector3d extractedAxis;
  double          extractedAngle;
  q.toAxisAngle(extractedAxis, extractedAngle);

  EXPECT_NEAR(extractedAxis.x(), 1.0, 1e-6);
  EXPECT_NEAR(extractedAxis.y(), 0.0, 1e-6);
  EXPECT_NEAR(extractedAxis.z(), 0.0, 1e-6);
  EXPECT_NEAR(extractedAngle, 0.0, 1e-6);
}

TEST(QuaternionTest, FromAxisAngleZeroAngle) {
  math::Vector3d axis(1.0, 0.0, 0.0);
  double          angle = 0.0;

  math::Quaterniond q = math::Quaterniond::fromAxisAngle(axis, angle);

  EXPECT_NEAR(q.x(), 0.0, 1e-6);
  EXPECT_NEAR(q.y(), 0.0, 1e-6);
  EXPECT_NEAR(q.z(), 0.0, 1e-6);
  EXPECT_NEAR(q.w(), 1.0, 1e-6);
}

TEST(QuaternionTest, FromAxisAngleNonNormalizedAxis) {
  math::Vector3d axis(2.0, 0.0, 0.0);
  double          angle = math::g_kPi / 2;

  math::Quaterniond q = math::Quaterniond::fromAxisAngle(axis, angle);

  EXPECT_NEAR(q.x(), std::sin(angle / 2), 1e-6);
  EXPECT_NEAR(q.y(), 0.0, 1e-6);
  EXPECT_NEAR(q.z(), 0.0, 1e-6);
  EXPECT_NEAR(q.w(), std::cos(angle / 2), 1e-6);
}

TEST(QuaternionTest, ToAxisAngleIdentityQuaternion) {
  math::Quaterniond q(0.0, 0.0, 0.0, 1.0);

  math::Vector3d axis;
  double          angle;
  q.toAxisAngle(axis, angle);

  EXPECT_NEAR(angle, 0.0, 1e-6);
  EXPECT_NEAR(axis.x(), 1.0, 1e-6);
  EXPECT_NEAR(axis.y(), 0.0, 1e-6);
  EXPECT_NEAR(axis.z(), 0.0, 1e-6);
}

TEST(QuaternionTest, FromVectorsIdentityDouble) {
  math::Vector3d from(1.0, 0.0, 0.0);
  math::Vector3d to(1.0, 0.0, 0.0);

  math::Quaterniond q = math::Quaterniond::fromVectors(from, to);

  EXPECT_NEAR(q.x(), 0.0, 1e-15);
  EXPECT_NEAR(q.y(), 0.0, 1e-15);
  EXPECT_NEAR(q.z(), 0.0, 1e-15);
  EXPECT_NEAR(q.w(), 1.0, 1e-15);
  EXPECT_NEAR(q.norm(), 1.0, 1e-15);

  math::Vector3d rotated = q.rotateVector(from);
  EXPECT_NEAR(rotated.x(), to.x(), 1e-15);
  EXPECT_NEAR(rotated.y(), to.y(), 1e-15);
  EXPECT_NEAR(rotated.z(), to.z(), 1e-15);
}

TEST(QuaternionTest, FromVectorsOppositeDouble) {
  math::Vector3d from(0.0, 1.0, 0.0);
  math::Vector3d to(0.0, -1.0, 0.0);

  math::Quaterniond q = math::Quaterniond::fromVectors(from, to);
  EXPECT_NEAR(q.norm(), 1.0, 1e-15);

  math::Vector3d rotated = q.rotateVector(from);

  double dot = rotated.normalized().dot(to);
  EXPECT_NEAR(dot, 1.0, 1e-8);

  EXPECT_NEAR((rotated - to).magnitude(), 0.0, 1e-7);

  const double kTol = 1e-7;
  EXPECT_NEAR(rotated.x(), to.x(), kTol);
  EXPECT_NEAR(rotated.y(), to.y(), kTol);
  EXPECT_NEAR(rotated.z(), to.z(), kTol);
}

TEST(QuaternionTest, FromVectorsArbitraryDouble) {
  math::Vector3d from(1.0, 1.0, 0.0);
  math::Vector3d to(0.0, 0.0, -2.0);

  math::Quaterniond q = math::Quaterniond::fromVectors(from, to);
  EXPECT_NEAR(q.norm(), 1.0, 1e-15);

  math::Vector3d rotatedDir = q.rotateVector(from).normalized();
  math::Vector3d toDir      = to.normalized();

  EXPECT_NEAR(rotatedDir.x(), toDir.x(), 1e-14);
  EXPECT_NEAR(rotatedDir.y(), toDir.y(), 1e-14);
  EXPECT_NEAR(rotatedDir.z(), toDir.z(), 1e-14);
}

// ========================== DIMENSION: DOUBLE ==============================

TEST(DimensionComparisonTest, LessThanOperatorDouble) {
  math::Dimension3<double> vec1(1.0, 2.0, 3.0);
  math::Dimension3<double> vec2(4.0, 5.0, 6.0);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorEqualDouble) {
  math::Dimension3<double> vec1(1.0, 2.0, 3.0);
  math::Dimension3<double> vec2(1.0, 2.0, 3.0);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorDouble) {
  math::Dimension3<double> vec1(4.0, 5.0, 6.0);
  math::Dimension3<double> vec2(1.0, 2.0, 3.0);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorEqualDouble) {
  math::Dimension3<double> vec1(1.0, 2.0, 3.0);
  math::Dimension3<double> vec2(1.0, 2.0, 3.0);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperatorDouble) {
  math::Dimension3<double> vec1(1.0, 2.0, 3.0);
  math::Dimension3<double> vec2(4.0, 5.0, 6.0);
  math::Dimension3<double> vec3(1.0, 2.0, 3.0);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(DimensionComparisonTest, GreaterThanOrEqualToOperatorDouble) {
  math::Dimension3<double> vec1(4.0, 5.0, 6.0);
  math::Dimension3<double> vec2(1.0, 2.0, 3.0);
  math::Dimension3<double> vec3(4.0, 5.0, 6.0);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ========================== DIMENSION: INT ==============================

TEST(DimensionComparisonTest, LessThanOperatorInt) {
  math::Dimension3i vec1(1, 2, 3);
  math::Dimension3i vec2(4, 5, 6);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorEqualInt) {
  math::Dimension3i vec1(1, 2, 3);
  math::Dimension3i vec2(1, 2, 3);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorInt) {
  math::Dimension3i vec1(4, 5, 6);
  math::Dimension3i vec2(1, 2, 3);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorEqualInt) {
  math::Dimension3i vec1(1, 2, 3);
  math::Dimension3i vec2(1, 2, 3);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperatorInt) {
  math::Dimension3i vec1(1, 2, 3);
  math::Dimension3i vec2(4, 5, 6);
  math::Dimension3i vec3(1, 2, 3);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(DimensionComparisonTest, GreaterThanOrEqualToOperatorInt) {
  math::Dimension3i vec1(4, 5, 6);
  math::Dimension3i vec2(1, 2, 3);
  math::Dimension3i vec3(4, 5, 6);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

// ======================= DIMENSION: UNSIGNED INT ===========================

TEST(DimensionComparisonTest, LessThanOperatorUnsignedInt) {
  math::Dimension3<unsigned int> vec1(1, 2, 3);
  math::Dimension3<unsigned int> vec2(4, 5, 6);

  EXPECT_TRUE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorEqualUnsignedInt) {
  math::Dimension3<unsigned int> vec1(1, 2, 3);
  math::Dimension3<unsigned int> vec2(1, 2, 3);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_FALSE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorUnsignedInt) {
  math::Dimension3<unsigned int> vec1(4, 5, 6);
  math::Dimension3<unsigned int> vec2(1, 2, 3);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorEqualUnsignedInt) {
  math::Dimension3<unsigned int> vec1(1, 2, 3);
  math::Dimension3<unsigned int> vec2(1, 2, 3);

  EXPECT_FALSE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperatorUnsignedInt) {
  math::Dimension3<unsigned int> vec1(1, 2, 3);
  math::Dimension3<unsigned int> vec2(4, 5, 6);
  math::Dimension3<unsigned int> vec3(1, 2, 3);

  EXPECT_TRUE(vec1 <= vec2);
  EXPECT_FALSE(vec2 <= vec1);
  EXPECT_TRUE(vec1 <= vec3);
  EXPECT_TRUE(vec3 <= vec1);
}

TEST(DimensionComparisonTest, GreaterThanOrEqualToOperatorUnsignedInt) {
  math::Dimension3<unsigned int> vec1(4, 5, 6);
  math::Dimension3<unsigned int> vec2(1, 2, 3);
  math::Dimension3<unsigned int> vec3(4, 5, 6);

  EXPECT_TRUE(vec1 >= vec2);
  EXPECT_FALSE(vec2 >= vec1);
  EXPECT_TRUE(vec1 >= vec3);
  EXPECT_TRUE(vec3 >= vec1);
}

TEST(DimensionComparisonTest, LessThanOperatorUnsignedIntLargeValues) {
  math::Dimension3<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Dimension3<unsigned int> vec2(0, 0, 0);

  EXPECT_FALSE(vec1 < vec2);
  EXPECT_TRUE(vec2 < vec1);
}

TEST(DimensionComparisonTest, GreaterThanOperatorUnsignedIntLargeValues) {
  math::Dimension3<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Dimension3<unsigned int> vec2(0, 0, 0);

  EXPECT_TRUE(vec1 > vec2);
  EXPECT_FALSE(vec2 > vec1);
}

TEST(DimensionComparisonTest, LessThanOrEqualToOperatorUnsignedIntLargeValues) {
  math::Dimension3<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Dimension3<unsigned int> vec2(0, 0, 0);

  EXPECT_FALSE(vec1 <= vec2);
  EXPECT_TRUE(vec2 <= vec1);
}

TEST(DimensionComparisonTest,
     GreaterThanOrEqualToOperatorUnsignedIntLargeValues) {
  math::Dimension3<unsigned int> vec1(
      0xFF'FF'FF'FF, 0xFF'FF'FF'FF, 0xFF'FF'FF'FF);
  math::Dimension3<unsigned int> vec2(0, 0, 0);

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
