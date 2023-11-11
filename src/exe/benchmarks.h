#ifndef MATH_LIBRARY_BENCHMARK_H
#define MATH_LIBRARY_BENCHMARK_H

#include <All.h>
#include <benchmark/benchmark.h>

static void bmMatrixCreationStack(benchmark::State& state);
static void bmMatrixCreationHeap(benchmark::State& state);
static void bmMatrixElementAccess(benchmark::State& state);
static void bmMatrixElementAccessRef(benchmark::State& state);
static void bmMatrixOperatorAccess(benchmark::State& state);
static void bmMatrixOperatorAccessRef(benchmark::State& state);
static void bmMatrixAddition(benchmark::State& state);
static void bmMatrixAdditionInPlace(benchmark::State& state);
static void bmMatrixScalarAddition(benchmark::State& state);
static void bmMatrixScalarAdditionInPlace(benchmark::State& state);
static void bmMatrixSubtraction(benchmark::State& state);
static void bmMatrixSubtractionInPlace(benchmark::State& state);
static void bmMatrixScalarSubtraction(benchmark::State& state);
static void bmMatrixScalarSubtractionInPlace(benchmark::State& state);
static void bmMatrixMultiplication(benchmark::State& state);
static void bmMatrixMultiplicationInPlace(benchmark::State& state);
static void bmMatrixScalarMultiplication(benchmark::State& state);
static void bmMatrixScalarMultiplicationInPlace(benchmark::State& state);
static void bmMatrixScalarDivision(benchmark::State& state);
static void bmMatrixScalarDivisionInPlace(benchmark::State& state);
static void bmMatrixTrace(benchmark::State& state);

//=========================================
//              BENCHMARKING
//=========================================

static void bmMatrixCreationStack(benchmark::State& state) {
  for (auto _ : state) {
    math::Matrix<float, 2, 2> matrix;
    benchmark::DoNotOptimize(matrix);
  }
}

BENCHMARK(bmMatrixCreationStack);

static void bmMatrixCreationHeap(benchmark::State& state) {
  for (auto _ : state) {
    math::Matrix<float, 100, 100> matrix;
    benchmark::DoNotOptimize(matrix);
  }
}

BENCHMARK(bmMatrixCreationHeap);

static void bmMatrixElementAccess(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    auto val = matrix.coeff(50, 50);
    benchmark::DoNotOptimize(val);
  }
}

BENCHMARK(bmMatrixElementAccess);

static void bmMatrixElementAccessRef(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    matrix.coeffRef(50, 50) = 1;
  }
}

BENCHMARK(bmMatrixElementAccessRef);

static void bmMatrixOperatorAccess(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    auto val = matrix(50, 50);
    benchmark::DoNotOptimize(val);
  }
}

BENCHMARK(bmMatrixOperatorAccess);

static void bmMatrixOperatorAccessRef(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    matrix(50, 50) = 1;
  }
}

BENCHMARK(bmMatrixOperatorAccessRef);

// BEGIN: addition benchmark
//---------------------------------------------------------------------------

static void bmMatrixAddition(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix1;
  math::Matrix<float, 100, 100> matrix2;
  for (auto _ : state) {
    auto matrix3 = matrix1 + matrix2;
    benchmark::DoNotOptimize(matrix3);
  }
}

BENCHMARK(bmMatrixAddition);

static void bmMatrixAdditionInPlace(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix1;
  math::Matrix<float, 100, 100> matrix2;
  for (auto _ : state) {
    matrix1 += matrix2;
  }
}

BENCHMARK(bmMatrixAdditionInPlace);

static void bmMatrixScalarAddition(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    auto matrix2 = matrix + 1.0f;
    benchmark::DoNotOptimize(matrix2);
  }
}

BENCHMARK(bmMatrixScalarAddition);

static void bmMatrixScalarAdditionInPlace(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    matrix += 1.0f;
  }
}

BENCHMARK(bmMatrixScalarAdditionInPlace);

// END: addition benchmark
//---------------------------------------------------------------------------

// BEGIN: subtraction benchmark
//---------------------------------------------------------------------------

static void bmMatrixSubtraction(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix1;
  math::Matrix<float, 100, 100> matrix2;
  for (auto _ : state) {
    auto matrix3 = matrix1 - matrix2;
    benchmark::DoNotOptimize(matrix3);
  }
}

BENCHMARK(bmMatrixSubtraction);

static void bmMatrixSubtractionInPlace(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix1;
  math::Matrix<float, 100, 100> matrix2;
  for (auto _ : state) {
    matrix1 -= matrix2;
  }
}

BENCHMARK(bmMatrixSubtractionInPlace);

static void bmMatrixScalarSubtraction(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    auto matrix2 = matrix - 1.0f;
    benchmark::DoNotOptimize(matrix2);
  }
}

BENCHMARK(bmMatrixScalarSubtraction);

static void bmMatrixScalarSubtractionInPlace(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    matrix -= 1.0f;
  }
}

BENCHMARK(bmMatrixScalarSubtractionInPlace);

// END: subtraction benchmark
//---------------------------------------------------------------------------

// BEGIN: multiplication benchmark
//---------------------------------------------------------------------------

static void bmMatrixMultiplication(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix1;
  math::Matrix<float, 100, 100> matrix2;
  for (auto _ : state) {
    auto matrix3 = matrix1 * matrix2;
    benchmark::DoNotOptimize(matrix3);
  }
}

BENCHMARK(bmMatrixMultiplication);

static void bmMatrixMultiplicationInPlace(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix1;
  math::Matrix<float, 100, 100> matrix2;
  for (auto _ : state) {
    matrix1 *= matrix2;
  }
}

BENCHMARK(bmMatrixMultiplicationInPlace);

static void bmMatrixScalarMultiplication(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    auto matrix2 = matrix * 2.0f;
    benchmark::DoNotOptimize(matrix2);
  }
}

BENCHMARK(bmMatrixScalarMultiplication);

static void bmMatrixScalarMultiplicationInPlace(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    matrix *= 2.0f;
  }
}

BENCHMARK(bmMatrixScalarMultiplicationInPlace);

// END: multiplication benchmark
//---------------------------------------------------------------------------

// BEGIN: division benchmark
//---------------------------------------------------------------------------

static void bmMatrixScalarDivision(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    auto matrix2 = matrix / 1.0f;
    benchmark::DoNotOptimize(matrix2);
  }
}

BENCHMARK(bmMatrixScalarDivision);

static void bmMatrixScalarDivisionInPlace(benchmark::State& state) {
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    matrix /= 1.0f;
  }
}

BENCHMARK(bmMatrixScalarDivisionInPlace);

// END: division benchmark
//---------------------------------------------------------------------------

static void bmMatrixDeterminant(benchmark::State& state) {
  for (auto _ : state) {
    math::Matrix<float, 5, 5> m;
    m.determinant();
  }
}

BENCHMARK(bmMatrixDeterminant);

static void bmMatrixInverse(benchmark::State& state) {
  for (auto _ : state) {
    math::Matrix<float, 5, 5> m;
    m.inverse();
  }
}

BENCHMARK(bmMatrixInverse);

static void bmMatrixRank(benchmark::State& state) {
  for (auto _ : state) {
    math::Matrix<float, 5, 5> m;
    m.rank();
  }
}

BENCHMARK(bmMatrixRank);

static void bmMatrixMagnitude(benchmark::State& state) {
  for (auto _ : state) {
    math::Matrix<float, 5, 1> m;
    m.magnitude();
  }
}

BENCHMARK(bmMatrixMagnitude);

static void bmMatrixNormalize(benchmark::State& state) {
  for (auto _ : state) {
    math::Matrix<float, 5, 1> m;
    m.normalize();
  }
}

BENCHMARK(bmMatrixNormalize);

// BEGIN: trace benchmark
//---------------------------------------------------------------------------

static void bmMatrixTrace(benchmark::State& state) {
  // Create a 100x100 matrix for the benchmark
  math::Matrix<float, 100, 100> matrix;
  for (auto _ : state) {
    float trace = matrix.trace();
    benchmark::DoNotOptimize(trace);
  }
}

BENCHMARK(bmMatrixTrace);

// END: trace benchmark
//---------------------------------------------------------------------------

//=========================================
//            END::BENCHMARKING
//=========================================

#endif