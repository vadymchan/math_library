
#include "tests.h"

#ifdef MATH_LIBRARY_INCLUDE_G_BENCHMARK
#include "benchmarks.h"
#endif

#include <math_library/all.h>

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  int test_result = RUN_ALL_TESTS();
  if (test_result != 0) {
    return test_result;
  }

#ifdef MATH_LIBRARY_INCLUDE_G_BENCHMARK
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
#endif

  return 0;
}