/**
 * @file Options.h
 * @brief This file contains the Options enum used for specifying various configuration options for mathematical objects.
 */

#pragma once

namespace math
{
    /**
     * @enum Options
     * @brief Enum class for specifying various configuration options for mathematical objects.
     *
     * This enum is used to specify different configuration options for mathematical objects such as matrices and vectors.
     * For instance, it can be used to specify the memory layout (row-major or column-major) of a Matrix object.
     * This can affect performance for certain operations.
     */
    enum class Options
    {
        ColumnMajor, /**< Enum value for column-major order. In this layout, the elements in the same column of a matrix are stored contiguously in memory. */
        RowMajor,    /**< Enum value for row-major order. In this layout, the elements in the same row of a matrix are stored contiguously in memory. */
        Count         /**< Enum value for counting the number of options.  */
    };
}
