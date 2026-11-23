/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <dlfcn.h>

/**
 * Test Description
 * ------------------------
 *  - The runtime exports __hipGetPCH (listed in its version script) whether or not it embeds a
 *    precompiled header, so binaries built against older runtimes still resolve it. Without an
 *    embedded header it returns a null blob of size 0; with one, a non-empty blob.
 * Test source
 * ------------------------
 *  - unit/rtc/hipGetPCH.cc
 * Test requirements
 * ------------------------
 *  - Linux
 *  - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_hipGetPCH_Exported) {
  HIP_CHECK(hipFree(nullptr));  // load and initialize the runtime
  using GetPCH = void (*)(const char**, unsigned int*);
  auto getPCH = reinterpret_cast<GetPCH>(dlsym(RTLD_DEFAULT, "__hipGetPCH"));
  REQUIRE(getPCH != nullptr);

  const char* pch = reinterpret_cast<const char*>(0x1);
  unsigned int size = ~0u;
  getPCH(&pch, &size);
  INFO("pch " << static_cast<const void*>(pch) << ", size " << size);
  REQUIRE((pch == nullptr) == (size == 0));
}
