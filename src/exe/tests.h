#pragma once

#include <gtest/gtest.h>

#include <All.h>

//TEST(MatrixTest, ConstructorDestructorFloat)
// TEST(MatrixTest, CopyConstructorFloat)
//TEST(MatrixTest, MoveConstructorFloat)
//TEST(MatrixTest, TransposeFloat)
//TEST(MatrixTest, ReshapeFloat)
//TEST(MatrixTest, ElementAccessFloat)
//TEST(MatrixTest, ConstructorDestructorFailureFloat)
//TEST(MatrixTest, ElementAccessFailureFloat)
//TEST(MatrixTest, ElementModificationFloat)
//TEST(MatrixTest, ElementModificationFailureFloat)
//TEST(MatrixTest, OutOfBoundsAccessFloat)
//TEST(MatrixTest, OperatorAccessFloat)
//TEST(MatrixTest, OperatorAccessConstFloat)
//TEST(MatrixTest, CoeffAccessConstFloat)
//TEST(MatrixTest, StackAllocationFloat)
//TEST(MatrixTest, HeapAllocationFloat)
//TEST(MatrixTest, AdditionFloat)
//TEST(MatrixTest, AdditionFailureFloat)
//TEST(MatrixTest, ScalarAdditionFloat)
//TEST(MatrixTest, ScalarAdditionFailureFloat)
//TEST(MatrixTest, SubtractionFloat)
//TEST(MatrixTest, ScalarSubtractionFloat)
//TEST(MatrixTest, MultiplicationRowMajorFloat)
//TEST(MatrixTest, MultiplicationColumnMajorFloat)
//TEST(MatrixTest, MultiplicationRowMajorFloat)
//TEST(MatrixTest, MultiplicationColumnMajorFloat)
//TEST(MatrixTest, ScalarDivisionFloat)
//TEST(MatrixTest, ScalarDivisionFailureFloat)
//TEST(MatrixTest, DeterminantFloat)
//TEST(MatrixTest, DeterminantFailureFloat)
//TEST(MatrixTest, InverseFloat)
//TEST(MatrixTest, InverseFailureFloat)
//TEST(MatrixTest, DeterminantFloat_1)
//TEST(MatrixTest, InverseFloat_1)
//TEST(MatrixTest, RankFloat)
//TEST(MatrixTest, RankFailureFloat)
//TEST(MatrixTest, RankFloat_2)
//TEST(MatrixTest, RankFailureFloat_2)
//TEST(MatrixTest, RankFullFloat)
//TEST(MatrixTest, MagnitudeFloat)
//TEST(MatrixTest, MagnitudeFailureFloat)
//TEST(MatrixTest, NormalizeFloat)
//TEST(MatrixTest, NormalizeFailureFloat)
//TEST(MatrixTest, NormalizeZeroVector)




TEST(MatrixTest, ConstructorDestructorFloat)
{
    math::Matrix<float, 2, 2> matrix;
    // If the constructor and destructor work correctly, this test will pass
}

TEST(MatrixTest, ElementAccessFloat)
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1; matrix.coeffRef(0, 1) = 2;
    matrix.coeffRef(1, 0) = 3; matrix.coeffRef(1, 1) = 4;
    EXPECT_EQ(matrix.coeff(0, 0), 1);
    EXPECT_EQ(matrix.coeff(0, 1), 2);
    EXPECT_EQ(matrix.coeff(1, 0), 3);
    EXPECT_EQ(matrix.coeff(1, 1), 4);
}

TEST(MatrixTest, ConstructorDestructorFailureFloat)
{
    math::Matrix<float, 2, 2> matrix;
    // This test will pass because the matrix is not null after construction
    EXPECT_NE(&matrix, nullptr);
}

TEST(MatrixTest, ElementAccessFailureFloat)
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1; matrix.coeffRef(0, 1) = 2;
    matrix.coeffRef(1, 0) = 3; matrix.coeffRef(1, 1) = 4;
    // These tests will pass because the expected values match the actual values
    EXPECT_NE(matrix.coeff(0, 0), 2);
    EXPECT_NE(matrix.coeff(0, 1), 1);
    EXPECT_NE(matrix.coeff(1, 0), 4);
    EXPECT_NE(matrix.coeff(1, 1), 3);
}

TEST(MatrixTest, ElementModificationFloat)
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    EXPECT_EQ(matrix.coeff(0, 0), 1);
    matrix.coeffRef(0, 0) = 2;
    EXPECT_EQ(matrix.coeff(0, 0), 2);
}

TEST(MatrixTest, ElementModificationFailureFloat)
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    EXPECT_NE(matrix.coeff(0, 0), 2);
    matrix.coeffRef(0, 0) = 2;
    EXPECT_NE(matrix.coeff(0, 0), 1);
}

#ifdef DEBUG

TEST(MatrixTest, OutOfBoundsAccessFloat)
{
    math::Matrix<float, 2, 2> matrix;
    EXPECT_DEATH(matrix.coeff(2, 2), "Index out of bounds");
    EXPECT_DEATH(matrix.coeffRef(2, 2), "Index out of bounds");
}

#endif // DEBUG

TEST(MatrixTest, OperatorAccessFloat)
{
    math::Matrix<float, 2, 2> matrix;
    matrix(0, 0) = 1;
    EXPECT_EQ(matrix(0, 0), 1);
    matrix(0, 0) = 2;
    EXPECT_EQ(matrix(0, 0), 2);
}

TEST(MatrixTest, OperatorAccessConstFloat)
{
    math::Matrix<float, 2, 2> matrix;
    matrix(0, 0) = 1;
    const auto& const_matrix = matrix;
    EXPECT_EQ(const_matrix(0, 0), 1);
    // Check that the type of the returned reference is const
    EXPECT_TRUE((std::is_const_v<std::remove_reference_t<decltype(const_matrix(0, 0))>>));
}

TEST(MatrixTest, CoeffAccessConstFloat)
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    const auto& const_matrix = matrix;
    EXPECT_EQ(const_matrix.coeff(0, 0), 1);
    // Check that the type of the returned reference is const
    EXPECT_TRUE((std::is_const_v<std::remove_reference_t<decltype(const_matrix.coeff(0, 0))>>));
}

TEST(MatrixTest, StackAllocationFloat)
{
    // This matrix should be small enough to be allocated on the stack
    math::Matrix<float, 2, 2> matrix;
    EXPECT_FALSE(matrix.UseHeap);
}

TEST(MatrixTest, HeapAllocationFloat)
{
    // This matrix should be large enough to be allocated on the heap
    math::Matrix<float, 100, 100> matrix;
    EXPECT_TRUE(matrix.UseHeap);
}

TEST(MatrixTest, AdditionFloat)
{
    constexpr int kRows = 2;
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

    math::Matrix<float, kRows, kColumns> matrix3 = matrix1 + matrix2;

    // Check that each element of the result is the sum of the corresponding elements of matrix1 and matrix2
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            EXPECT_EQ(matrix3.coeff(i, j), matrix1.coeff(i, j) + matrix2.coeff(i, j));
        }
    }
}

TEST(MatrixTest, AdditionFailureFloat)
{
    math::Matrix<float, 4, 4, math::Options::COLUMN_MAJOR> matrix1;
    math::Matrix<float, 4, 4, math::Options::COLUMN_MAJOR> matrix2;

    // Populate matrix1 and matrix2 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
        }
    }

    math::Matrix<float, 4, 4, math::Options::COLUMN_MAJOR> matrix3 = matrix1 + matrix2;

    // Check that each element of the result is not the sum of the corresponding elements of matrix1 and matrix2 plus 1
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NE(matrix3.coeff(i, j), matrix1.coeff(i, j) + matrix2.coeff(i, j) + 1);
        }
    }
}

TEST(MatrixTest, ScalarAdditionFloat)
{
    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix1;

    // Populate matrix1 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
        }
    }

    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix2 = matrix1 + 10;

    // Check that each element of the result is the corresponding element of matrix1 plus 10
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) + 10);
        }
    }
}

TEST(MatrixTest, ScalarAdditionFailureFloat)
{
    math::Matrix<float, 4, 4> matrix1;

    // Populate matrix1 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
        }
    }

    math::Matrix<float, 4, 4> matrix2 = matrix1 + 10;

    // Check that each element of the result is not the corresponding element of matrix1 plus 11
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NE(matrix2.coeff(i, j), matrix1.coeff(i, j) + 11);
        }
    }
}

TEST(MatrixTest, SubtractionFloat)
{
    constexpr int kRows = 2;
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

    math::Matrix<int, kRows, kColumns> matrix3 = matrix1 - matrix2;

    // Check that each element of the result is the difference of the corresponding elements of matrix1 and matrix2
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            EXPECT_EQ(matrix3.coeff(i, j), matrix1.coeff(i, j) - matrix2.coeff(i, j));
        }
    }
}

TEST(MatrixTest, ScalarSubtractionFloat)
{
    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix1;

    // Populate matrix1 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
        }
    }

    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix2 = matrix1 - 10;

    // Check that each element of the result is the corresponding element of matrix1 minus 10
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) - 10);
        }
    }
}


TEST(MatrixTest, MultiplicationRowMajorFloat)
{
    constexpr int kRows = 2;
    constexpr int kColumns = 2;

    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix1;
    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix2;

    // Populate matrix1 and matrix2 with some values
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
        }
    }

    // Print initial matrices
    std::cout << "Matrix 1:\n" << matrix1 << "\nMatrix 2:\n" << matrix2 << std::endl;

    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix3 = matrix1 * matrix2;

    // Print result matrix
    std::cout << "Result Matrix:\n" << matrix3 << std::endl;

    // Check that each element of the result is the correct multiplication of the corresponding rows and columns of matrix1 and matrix2
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

TEST(MatrixTest, MultiplicationColumnMajorFloat)
{
    constexpr int kRows = 2;
    constexpr int kColumns = 2;

    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix1;
    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix2;

    // Populate matrix1 and matrix2 with some values
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
        }
    }

    // Print initial matrices
    std::cout << "Matrix 1:\n" << matrix1 << "\nMatrix 2:\n" << matrix2 << std::endl;

    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix3 = matrix1 * matrix2;

    // Print result matrix
    std::cout << "Result Matrix:\n" << matrix3 << std::endl;

    // Check that each element of the result is the correct multiplication of the corresponding rows and columns of matrix1 and matrix2
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

//TEST(MatrixTest, MultiplicationRowMajorFloat)
//{
//    constexpr int kRows = 2;
//    constexpr int kColumns = 2;
//
//    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix1;
//    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix2;
//
//    // Populate matrix1 and matrix2 with some values
//    for (int i = 0; i < kRows; ++i) {
//        for (int j = 0; j < kColumns; ++j) {
//            matrix1.coeffRef(i, j) = i * 4 + j + 1;
//            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
//        }
//    }
//
//    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix3 = matrix1 * matrix2;
//
//    // Check that each element of the result is the correct multiplication of the corresponding rows and columns of matrix1 and matrix2
//    for (int i = 0; i < kRows; ++i) {
//        for (int j = 0; j < kColumns; ++j) {
//            float expected_value = 0;
//            for (int k = 0; k < kColumns; ++k) {
//                expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
//            }
//            EXPECT_EQ(matrix3.coeff(i, j), expected_value);
//        }
//    }
//}
//
//TEST(MatrixTest, MultiplicationColumnMajorFloat)
//{
//    constexpr int kRows = 2;
//    constexpr int kColumns = 2;
//
//    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix1;
//    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix2;
//
//    // Populate matrix1 and matrix2 with some values
//    for (int i = 0; i < kRows; ++i) {
//        for (int j = 0; j < kColumns; ++j) {
//            matrix1.coeffRef(i, j) = i * 4 + j + 1;
//            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
//        }
//    }
//
//    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix3 = matrix1 * matrix2;
//
//    // Check that each element of the result is the correct multiplication of the corresponding rows and columns of matrix1 and matrix2
//    for (int i = 0; i < kRows; ++i) {
//        for (int j = 0; j < kColumns; ++j) {
//            float expected_value = 0;
//            for (int k = 0; k < kColumns; ++k) {
//                expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
//            }
//            EXPECT_EQ(matrix3.coeff(i, j), expected_value);
//        }
//    }
//}


TEST(MatrixTest, ScalarDivisionFloat)
{
    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix1;

    // Populate matrix1 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
        }
    }

    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix2 = matrix1 / 10;

    // Check that each element of the result is the corresponding element of matrix1 divided by 10
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) / 10);
        }
    }
}

TEST(MatrixTest, ScalarDivisionFailureFloat)
{
    math::Matrix<float, 4, 4> matrix1;

    // Populate matrix1 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
        }
    }

    math::Matrix<float, 4, 4> matrix2 = matrix1 / 10;

    // Check that each element of the result is not the corresponding element of matrix1 divided by 11
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NE(matrix2.coeff(i, j), matrix1.coeff(i, j) / 11);
        }
    }
}

//============================================== INT ==============================================

TEST(MatrixTest, ConstructorDestructorInt)
{
    math::Matrix<int, 2, 2> matrix;
    // If the constructor and destructor work correctly, this test will pass
}

TEST(MatrixTest, ElementAccessInt)
{
    math::Matrix<int, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1; matrix.coeffRef(0, 1) = 2;
    matrix.coeffRef(1, 0) = 3; matrix.coeffRef(1, 1) = 4;
    EXPECT_EQ(matrix.coeff(0, 0), 1);
    EXPECT_EQ(matrix.coeff(0, 1), 2);
    EXPECT_EQ(matrix.coeff(1, 0), 3);
    EXPECT_EQ(matrix.coeff(1, 1), 4);
}

TEST(MatrixTest, ConstructorDestructorFailureInt)
{
    math::Matrix<int, 2, 2> matrix;
    // This test will pass because the matrix is not null after construction
    EXPECT_NE(&matrix, nullptr);
}

TEST(MatrixTest, ElementAccessFailureInt)
{
    math::Matrix<int, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1; matrix.coeffRef(0, 1) = 2;
    matrix.coeffRef(1, 0) = 3; matrix.coeffRef(1, 1) = 4;
    // These tests will pass because the expected values match the actual values
    EXPECT_NE(matrix.coeff(0, 0), 2);
    EXPECT_NE(matrix.coeff(0, 1), 1);
    EXPECT_NE(matrix.coeff(1, 0), 4);
    EXPECT_NE(matrix.coeff(1, 1), 3);
}

TEST(MatrixTest, ElementModificationInt)
{
    math::Matrix<int, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    EXPECT_EQ(matrix.coeff(0, 0), 1);
    matrix.coeffRef(0, 0) = 2;
    EXPECT_EQ(matrix.coeff(0, 0), 2);
}

TEST(MatrixTest, ElementModificationFailureInt)
{
    math::Matrix<int, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    EXPECT_NE(matrix.coeff(0, 0), 2);
    matrix.coeffRef(0, 0) = 2;
    EXPECT_NE(matrix.coeff(0, 0), 1);
}

#ifdef DEBUG

TEST(MatrixTest, OutOfBoundsAccessInt)
{
    math::Matrix<int, 2, 2> matrix;
    EXPECT_DEATH(matrix.coeff(2, 2), "Index out of bounds");
    EXPECT_DEATH(matrix.coeffRef(2, 2), "Index out of bounds");
}

#endif // DEBUG


TEST(MatrixTest, OperatorAccessInt)
{
    math::Matrix<int, 2, 2> matrix;
    matrix(0, 0) = 1;
    EXPECT_EQ(matrix(0, 0), 1);
    matrix(0, 0) = 2;
    EXPECT_EQ(matrix(0, 0), 2);
}

TEST(MatrixTest, OperatorAccessConstInt)
{
    math::Matrix<int, 2, 2> matrix;
    matrix(0, 0) = 1;
    const auto& const_matrix = matrix;
    EXPECT_EQ(const_matrix(0, 0), 1);
    // Check that the type of the returned reference is const
    EXPECT_TRUE((std::is_const_v<std::remove_reference_t<decltype(const_matrix(0, 0))>>));
}

TEST(MatrixTest, CoeffAccessConstInt)
{
    math::Matrix<int, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    const auto& const_matrix = matrix;
    EXPECT_EQ(const_matrix.coeff(0, 0), 1);
    // Check that the type of the returned reference is const
    EXPECT_TRUE((std::is_const_v<std::remove_reference_t<decltype(const_matrix.coeff(0, 0))>>));
}

TEST(MatrixTest, StackAllocationInt)
{
    // This matrix should be small enough to be allocated on the stack
    math::Matrix<int, 2, 2> matrix;
    EXPECT_FALSE(matrix.UseHeap);
}

TEST(MatrixTest, HeapAllocationInt)
{
    // This matrix should be large enough to be allocated on the heap
    math::Matrix<int, 100, 100> matrix;
    EXPECT_TRUE(matrix.UseHeap);
}

TEST(MatrixTest, AdditionInt)
{
    constexpr int kRows = 2;
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

    math::Matrix<int, kRows, kColumns> matrix3 = matrix1 + matrix2;

    // Check that each element of the result is the sum of the corresponding elements of matrix1 and matrix2
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            EXPECT_EQ(matrix3.coeff(i, j), matrix1.coeff(i, j) + matrix2.coeff(i, j));
        }
    }
}

TEST(MatrixTest, AdditionFailureInt)
{
    math::Matrix<int, 4, 4, math::Options::COLUMN_MAJOR> matrix1;
    math::Matrix<int, 4, 4, math::Options::COLUMN_MAJOR> matrix2;

    // Populate matrix1 and matrix2 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
        }
    }

    math::Matrix<int, 4, 4, math::Options::COLUMN_MAJOR> matrix3 = matrix1 + matrix2;

    // Check that each element of the result is not the sum of the corresponding elements of matrix1 and matrix2 plus 1
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NE(matrix3.coeff(i, j), matrix1.coeff(i, j) + matrix2.coeff(i, j) + 1);
        }
    }
}

TEST(MatrixTest, SubtractionInt)
{
    constexpr int kRows = 2;
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

    math::Matrix<int, kRows, kColumns> matrix3 = matrix1 - matrix2;

    // Check that each element of the result is the difference of the corresponding elements of matrix1 and matrix2
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            EXPECT_EQ(matrix3.coeff(i, j), matrix1.coeff(i, j) - matrix2.coeff(i, j));
        }
    }
}

TEST(MatrixTest, SubtractionFailureInt)
{
    math::Matrix<int, 4, 4, math::Options::COLUMN_MAJOR> matrix1;
    math::Matrix<int, 4, 4, math::Options::COLUMN_MAJOR> matrix2;

    // Populate matrix1 and matrix2 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
        }
    }

    math::Matrix<int, 4, 4, math::Options::COLUMN_MAJOR> matrix3 = matrix1 - matrix2;

    // Check that each element of the result is not the difference of the corresponding elements of matrix1 and matrix2 plus 1
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NE(matrix3.coeff(i, j), matrix1.coeff(i, j) - matrix2.coeff(i, j) + 1);
        }
    }
}

//TEST(MatrixTest, MultiplicationInt)
//{
//    constexpr int kRows = 2;
//    constexpr int kColumns = 2;
//
//    math::Matrix<int, kRows, kColumns> matrix1;
//    math::Matrix<int, kRows, kColumns> matrix2;
//
//    // Populate matrix1 and matrix2 with some values
//    for (int i = 0; i < kRows; ++i) {
//        for (int j = 0; j < kColumns; ++j) {
//            matrix1.coeffRef(i, j) = i * 4 + j + 1;
//            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
//        }
//    }
//
//    math::Matrix<int, kRows, kColumns> matrix3 = matrix1 * matrix2;
//
//    // Check that each element of the result is the dot product of the corresponding row of matrix1 and column of matrix2
//    for (int i = 0; i < kRows; ++i) {
//        for (int j = 0; j < kColumns; ++j) {
//            int expected_value = 0;
//            for (int k = 0; k < kRows; ++k) {
//                expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
//            }
//            EXPECT_EQ(matrix3.coeff(i, j), expected_value);
//        }
//    }
//}
//
//TEST(MatrixTest, MultiplicationFailureInt)
//{
//    math::Matrix<int, 4, 4, math::Options::COLUMN_MAJOR> matrix1;
//    math::Matrix<int, 4, 4, math::Options::COLUMN_MAJOR> matrix2;
//
//    // Populate matrix1 and matrix2 with some values
//    for (int i = 0; i < 4; ++i) {
//        for (int j = 0; j < 4; ++j) {
//            matrix1.coeffRef(i, j) = i * 4 + j + 1;
//            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
//        }
//    }
//
//    math::Matrix<int, 4, 4, math::Options::COLUMN_MAJOR> matrix3 = matrix1 * matrix2;
//
//    // Check that each element of the result is not the dot product of the corresponding row of matrix1 and column of matrix2 plus 1
//    for (int i = 0; i < 4; ++i) {
//        for (int j = 0; j < 4; ++j) {
//            int expected_value = 0;
//            for (int k = 0; k < 4; ++k) {
//                expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
//            }
//            EXPECT_NE(matrix3.coeff(i, j), expected_value + 1);
//        }
//    }
//}

//============================================== INT ==============================================

