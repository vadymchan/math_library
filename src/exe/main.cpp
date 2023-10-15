
#include "benchmarks.h"
#include "tests.h"

#include <All.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int test_result = RUN_ALL_TESTS();
  if (test_result != 0) {
    return test_result;
  }

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
