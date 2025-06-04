
#include "benchmarks.h"
#include "tests.h"

#include <math_library/all.h>

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  int test_result = RUN_ALL_TESTS();
  if (test_result != 0) {
    return test_result;
  }

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  return 0;
}