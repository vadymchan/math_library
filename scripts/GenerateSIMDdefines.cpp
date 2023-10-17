#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef _MSC_VER
  #include <intrin.h>
#else
  #include <cpuid.h>
#endif

#define USE_LOCALTIME

#ifndef USE_LOCALTIME
  #define USE_GMT
#endif

std::string getCurrentDateTime() {
  std::ostringstream ss;
  auto               now = std::chrono::system_clock::now();
  auto               itt = std::chrono::system_clock::to_time_t(now);
#ifdef USE_GMT
  ss << std::put_time(gmtime(&itt), "%Y-%m-%d %H:%M:%S");
#else
  ss << std::put_time(localtime(&itt), "%Y-%m-%d %H:%M:%S");
#endif

  return ss.str();
}

int main() {
  std::string   defines_file_path = "src/lib/simd/precompiled/SIMDdefines.h";
  std::ofstream defines(defines_file_path);

  if (!defines) {
    std::cerr << "Failed to open " << defines_file_path << " for writing\n";
    return 1;
  } else {
    std::cout << "Successfully opened " << defines_file_path
              << " for writing\n";
  }

  std::string currentDateTime = getCurrentDateTime();

  // Doxygen comment
  defines
      << "/**\n"
      << " * @file " << defines_file_path << "\n"
      << " * @brief Defines the SIMD capabilities of the current CPU\n"
      << " * @date " << currentDateTime << "\n"
      << " *\n"
      << " * This file is generated by GenerateSIMDdefines.h during build of "
         "the project. It uses the CPUID instruction to\n"
      << " * detect the SIMD capabilities of the current CPU, and then writes\n"
      << " * corresponding preprocessor definitions to this file. The SIMD "
         "capabilities\n"
      << " * that are detected include SSE3, SSSE3, SSE4.1, SSE4.2, AVX, and "
         "AVX2.\n"
      << " */\n\n";

#ifdef _MSC_VER
  int cpuInfo[4];
  __cpuid(cpuInfo, 0);

  int maxLeaf = cpuInfo[0];

  if (maxLeaf >= 1) {
    __cpuid(cpuInfo, 1);

    if (cpuInfo[2] & (1 << 0)) {
      defines << "#define SUPPORTS_SSE3\n";
    }
    if (cpuInfo[2] & (1 << 9)) {
      defines << "#define SUPPORTS_SSSE3\n";
    }
    if (cpuInfo[2] & (1 << 19)) {
      defines << "#define SUPPORTS_SSE4_1\n";
    }
    if (cpuInfo[2] & (1 << 20)) {
      defines << "#define SUPPORTS_SSE4_2\n";
    }
    if (cpuInfo[2] & (1 << 28)) {
      defines << "#define SUPPORTS_AVX\n";
    }
  }

  if (maxLeaf >= 7) {
    __cpuidex(cpuInfo, 7, 0);

    if (cpuInfo[1] & (1 << 5)) {
      defines << "#define SUPPORTS_AVX2\n";
    }
    if (cpuInfo[1] & (1 << 16)) {
      defines << "#define SUPPORTS_AVX512F\n";
    }
  }

#else
  unsigned int eax, ebx, ecx, edx;
  __get_cpuid(0, &eax, &ebx, &ecx, &edx);

  unsigned int maxLeaf = eax;

  if (maxLeaf >= 1) {
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);

    if (ecx & bit_SSE3) {
      defines << "#define SUPPORTS_SSE3\n";
    }
    if (ecx & bit_SSSE3) {
      defines << "#define SUPPORTS_SSSE3\n";
    }
    if (ecx & bit_SSE4_1) {
      defines << "#define SUPPORTS_SSE4_1\n";
    }
    if (ecx & bit_SSE4_2) {
      defines << "#define SUPPORTS_SSE4_2\n";
    }
    if (ecx & bit_AVX) {
      defines << "#define SUPPORTS_AVX\n";
    }
  }

  if (maxLeaf >= 7) {
    __get_cpuid(7, &eax, &ebx, &ecx, &edx);

    if (ebx & (1 << 5)) {
      defines << "#define SUPPORTS_AVX2\n";
    }
    if (ebx & (1 << 16)) {
      defines << "#define SUPPORTS_AVX512F\n";
    }
  }

#endif

  return 0;
}
