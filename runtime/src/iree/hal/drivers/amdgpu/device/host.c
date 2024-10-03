// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/host.h"

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_device_host_post(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_host_t* IREE_OCL_RESTRICT host,
    uint16_t type, uint64_t return_address, uint64_t arg0, uint64_t arg1,
    uint64_t arg2, uint64_t arg3, iree_hsa_signal_t completion_signal) {
  // Reserve a packet write index and wait for it to become available in cases
  // where the queue is exhausted.
  uint64_t packet_id = iree_hsa_queue_add_write_index(
      host, 1u, iree_hal_amdgpu_device_memory_order_relaxed);
  while (packet_id - iree_hsa_queue_load_read_index(
                         host, iree_hal_amdgpu_device_memory_order_acquire) >=
         host->size) {
    iree_hal_amdgpu_device_yield();  // spinning
  }
  const uint64_t queue_mask = host->size - 1;  // power of two
  IREE_OCL_GLOBAL iree_hsa_agent_dispatch_packet_t* IREE_OCL_RESTRICT
      agent_packet = host->base_address + (packet_id & queue_mask) * 64;

  // Populate all of the packet besides the header.
  agent_packet->reserved0 = 0;
  agent_packet->return_address = (void*)return_address;
  agent_packet->arg[0] = arg0;
  agent_packet->arg[1] = arg1;
  agent_packet->arg[2] = arg2;
  agent_packet->arg[3] = arg3;
  agent_packet->reserved2 = 0;
  agent_packet->completion_signal = completion_signal;

  // Populate the header and release the packet to the queue.
  uint16_t header = IREE_HSA_PACKET_TYPE_AGENT_DISPATCH
                    << IREE_HSA_PACKET_HEADER_TYPE;

  // NOTE: no barrier bit as posts should technically be executed back-to-back.
  // If a particular post type supports concurrent or out-of-order execution
  // then it _may_ do so unless the bit is set.
  header |= 1 << IREE_HSA_PACKET_HEADER_BARRIER;

  // Posts are unidirectional and take device agent resources and make them
  // available to the host. We may be able to get away with an scacquire of
  // IREE_HSA_FENCE_SCOPE_AGENT here but conservatively use
  // IREE_HSA_FENCE_SCOPE_SYSTEM so that if any resources happen to have been
  // touched on other agents (such as when executing multi-device work as part
  // of a command buffer collective operation) the host can see all of that.
  // It certainly is not optimal to do, though.
  header |= IREE_HSA_FENCE_SCOPE_SYSTEM
            << IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  header |= IREE_HSA_FENCE_SCOPE_SYSTEM
            << IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;

  const uint32_t header_type = header | (type << 16);
  iree_hal_amdgpu_device_atomic_store_explicit(
      (IREE_OCL_GLOBAL iree_hal_amdgpu_device_atomic_uint32_t*)agent_packet,
      header_type, iree_hal_amdgpu_device_memory_order_release,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);

  // Signal the queue doorbell.
  // This will store the packet_id to the doorbell signal (though in MULTI mode
  // it's ignored) and in the case of the host agent trigger a hardware
  // interrupt via the event mailbox pointer on the signal. If the host is doing
  // a kernel wait via the HSA APIs it should be woken pretty quickly.
  // https://sourcegraph.com/github.com/ROCm/rocMLIR/-/blob/external/llvm-project/amd/device-libs/ockl/src/hsaqs.cl?L69
  iree_hsa_signal_store(host->doorbell_signal, packet_id,
                        iree_hal_amdgpu_device_memory_order_relaxed);
}

void iree_hal_amdgpu_device_host_post_release(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_host_t* IREE_OCL_RESTRICT host,
    uint64_t resource0, uint64_t resource1, uint64_t resource2,
    uint64_t resource3, iree_hsa_signal_t completion_signal) {
  iree_hal_amdgpu_device_host_post(
      host, IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_RELEASE,
      /*return_address=*/0, resource0, resource1, resource2, resource3,
      completion_signal);
}

void iree_hal_amdgpu_device_host_post_error(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_host_t* IREE_OCL_RESTRICT host,
    uint64_t code, uint64_t arg0, uint64_t arg1) {
  iree_hal_amdgpu_device_host_post(
      host, IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_ERROR, /*return_address=*/0,
      /*reserved=*/0, code, arg0, arg1, iree_hsa_signal_null());
}

void iree_hal_amdgpu_device_host_post_signal(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_host_t* IREE_OCL_RESTRICT host,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_semaphore_t* IREE_OCL_RESTRICT
        semaphore,
    uint64_t payload) {
  iree_hal_amdgpu_device_host_post(
      host, IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_SIGNAL,
      /*return_address=*/0, (uint64_t)semaphore, payload,
      /*unused=*/0,
      /*unused=*/0, iree_hsa_signal_null());
}

void iree_hal_amdgpu_device_host_post_trace_flush(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_host_t* IREE_OCL_RESTRICT host,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hsa_signal_t completion_signal) {
  iree_hal_amdgpu_device_host_post(
      host, IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_TRACE_FLUSH,
      /*return_address=*/0, (uint64_t)trace_buffer, /*unused=*/0,
      /*unused=*/0,
      /*unused=*/0, completion_signal);
}
