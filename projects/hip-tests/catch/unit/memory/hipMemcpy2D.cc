/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "memcpy2d_tests_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

HIP_TEST_CASE(Unit_hipMemcpy2D_Positive_Basic) {
  constexpr bool async = false;

  SECTION("Device to Host") { Memcpy2DDeviceToHostShell<async>(hipMemcpy2D); }

  SECTION("Device to Device") {
    SECTION("Peer access enabled") { Memcpy2DDeviceToDeviceShell<async, true>(hipMemcpy2D); }
  }

  SECTION("Host to Device") { Memcpy2DHostToDeviceShell<async>(hipMemcpy2D); }

  SECTION("Host to Host") { Memcpy2DHostToHostShell<async>(hipMemcpy2D); }
}

HIP_TEST_CASE(Unit_hipMemcpy2D_Positive_Synchronization_Behavior) {
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Device") { Memcpy2DHtoDSyncBehavior(hipMemcpy2D, true); }

  SECTION("Device to Host") {
    Memcpy2DDtoHPageableSyncBehavior(hipMemcpy2D, true);
    Memcpy2DDtoHPinnedSyncBehavior(hipMemcpy2D, true);
  }

  SECTION("Device to Device") {
    Memcpy2DDtoDSyncBehavior(hipMemcpy2D, false);
  }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-232
  SECTION("Host to Host") { Memcpy2DHtoHSyncBehavior(hipMemcpy2D, true); }
#endif
}

HIP_TEST_CASE(Unit_hipMemcpy2D_Positive_Parameters) {
  constexpr bool async = false;
  Memcpy2DZeroWidthHeight<async>(hipMemcpy2D);
}

HIP_TEST_CASE(Unit_hipMemcpy2D_Negative_Parameters) {
  constexpr size_t cols = 128;
  constexpr size_t rows = 128;

  constexpr auto NegativeTests = [](void* dst, size_t dpitch, const void* src, size_t spitch,
                                    size_t width, size_t height, hipMemcpyKind kind) {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2D(nullptr, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2D(dst, dpitch, nullptr, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }

    SECTION("dpitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2D(dst, width - 1, src, spitch, width, height, kind),
                      hipErrorInvalidPitchValue);
    }

    SECTION("spitch < width") {
      HIP_CHECK_ERROR(hipMemcpy2D(dst, dpitch, src, width - 1, width, height, kind),
                      hipErrorInvalidPitchValue);
    }

    SECTION("dpitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(
          hipMemcpy2D(dst, static_cast<size_t>(attr) + 1, src, spitch, width, height, kind),
          hipErrorInvalidValue);
    }

    SECTION("spitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(
          hipMemcpy2D(dst, dpitch, src, static_cast<size_t>(attr) + 1, width, height, kind),
          hipErrorInvalidValue);
    }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-234
    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(
          hipMemcpy2D(dst, dpitch, src, spitch, width, height, static_cast<hipMemcpyKind>(-1)),
          hipErrorInvalidMemcpyDirection);
    }
#endif
  };

  SECTION("Host to Device") {
    LinearAllocGuard2D<int> device_alloc(cols, rows);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);
    NegativeTests(device_alloc.ptr(), device_alloc.pitch(), host_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyHostToDevice);
  }

  SECTION("Device to Host") {
    LinearAllocGuard2D<int> device_alloc(cols, rows);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);
    NegativeTests(host_alloc.ptr(), device_alloc.pitch(), device_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyDeviceToHost);
  }

  SECTION("Host to Host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    NegativeTests(dst_alloc.ptr(), cols * sizeof(int), src_alloc.ptr(), cols * sizeof(int),
                  cols * sizeof(int), rows, hipMemcpyHostToHost);
  }

  SECTION("Device to Device") {
    LinearAllocGuard2D<int> src_alloc(cols, rows);
    LinearAllocGuard2D<int> dst_alloc(cols, rows);
    NegativeTests(dst_alloc.ptr(), dst_alloc.pitch(), src_alloc.ptr(), src_alloc.pitch(),
                  dst_alloc.width(), dst_alloc.height(), hipMemcpyDeviceToDevice);
  }
}

HIP_TEST_CASE(Unit_hipMemcpy2D_Capture) {

  constexpr size_t width = 16;
  constexpr size_t height = 16;

  LinearAllocGuard2D<int> device_alloc(width, height);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * width);

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, false);
  HIP_CHECK_ERROR(
      hipMemcpy2D(device_alloc.ptr(), device_alloc.pitch(), host_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyHostToDevice),
      memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);
}

/**
 * Test Description
 * ------------------------
 *  - A tall, one-byte-wide host<->device rect has pitches that aren't dword aligned. The SDMA
 *    rect copy can't take those, and copying them one SDMA submission per row took seconds for a
 *    million rows; the copy must be done in one pass. Checks the data both ways and bounds the
 *    time generously (well under the per-row cost).
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipMemcpy2D_Positive_NarrowTallHostRect) {
  constexpr size_t kHeight = 1 << 20;
  constexpr size_t kHostPitch = 1;
  constexpr size_t kDevPitch = 3;
  const hipMemcpyKind kind = GENERATE(hipMemcpyHostToDevice, hipMemcpyDeviceToHost);

  LinearAllocGuard<uint8_t> host_src(LinearAllocs::hipHostMalloc, kHeight * kHostPitch);
  LinearAllocGuard<uint8_t> host_dst(LinearAllocs::hipHostMalloc, kHeight * kHostPitch);
  LinearAllocGuard<uint8_t> dev(LinearAllocs::hipMalloc, kHeight * kDevPitch);
  for (size_t i = 0; i < kHeight; i++) {
    host_src.host_ptr()[i] = static_cast<uint8_t>(i * 13 + 1);
    host_dst.host_ptr()[i] = 0;
  }

  if (kind == hipMemcpyDeviceToHost) {
    HIP_CHECK(hipMemcpy2D(dev.ptr(), kDevPitch, host_src.host_ptr(), kHostPitch, 1, kHeight,
                          hipMemcpyHostToDevice));
  }
  const auto start = std::chrono::steady_clock::now();
  if (kind == hipMemcpyHostToDevice) {
    HIP_CHECK(hipMemcpy2D(dev.ptr(), kDevPitch, host_src.host_ptr(), kHostPitch, 1, kHeight,
                          hipMemcpyHostToDevice));
  } else {
    HIP_CHECK(hipMemcpy2D(host_dst.host_ptr(), kHostPitch, dev.ptr(), kDevPitch, 1, kHeight,
                          hipMemcpyDeviceToHost));
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  if (kind == hipMemcpyHostToDevice) {
    HIP_CHECK(hipMemcpy2D(host_dst.host_ptr(), kHostPitch, dev.ptr(), kDevPitch, 1, kHeight,
                          hipMemcpyDeviceToHost));
  }

  for (size_t i = 0; i < kHeight; i++) {
    if (host_dst.host_ptr()[i] != host_src.host_ptr()[i]) {
      INFO("row " << i);
      REQUIRE(host_dst.host_ptr()[i] == host_src.host_ptr()[i]);
    }
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
  INFO("copy took " << elapsed_ms << " ms");
  REQUIRE(elapsed < std::chrono::seconds(1));
}
