// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_HOST_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_HOST_H_

#include "iree/hal/drivers/amdgpu/device/semaphore.h"
#include "iree/hal/drivers/amdgpu/device/support/opencl.h"
#include "iree/hal/drivers/amdgpu/device/support/queue.h"
#include "iree/hal/drivers/amdgpu/device/support/signal.h"
#include "iree/hal/drivers/amdgpu/device/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_host_t
//===----------------------------------------------------------------------===//

enum iree_hal_amdgpu_device_host_call_e {
  // Host will route to iree_hal_hsa_pool_grow.
  //
  // Signature:
  //   arg0: iree_hal_hsa_pool_t* pool
  //   arg1: block?
  //   arg2: uint64_t allocation_size
  //   arg3: uint32_t allocation_offset (offset into block)
  //         uint32_t min_alignment
  //   return_address: iree_hal_amdgpu_device_allocation_handle_t* handle
  //   completion_signal: signaled when the pool has grown
  IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POOL_GROW = 0u,

  // Host will route to iree_hal_hsa_pool_trim.
  //
  // Signature:
  //   arg0: iree_hal_hsa_pool_t* pool
  //   arg1: block?
  //   arg2:
  //   arg3:
  //   return_address:
  //   completion_signal: signaled when the pool has been trimmed
  IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POOL_TRIM,

  // Host will call iree_hal_resource_release on each non-NULL resource pointer.
  // This is effectively a transfer operation indicating that the device will no
  // longer be using the resources.
  //
  // It's strongly recommended that iree_hal_resource_set_t is used where
  // appropriate so that the number of packets required to release a set of
  // resources can be kept small. The 4 available here is just enough for the
  // common case of submissions like execute that are a wait semaphore, the
  // command buffer, the binding table resource set, and the signal semaphore.
  //
  // TODO(benvanik): evaluate a version that takes a ringbuffer of uint64_t
  // pointers and make this a drain request instead. Then we can enqueue as many
  // as we want and kick the host to drain as it is able.
  //
  // Signature:
  //   arg0: iree_hal_resource_t* resource0
  //   arg1: iree_hal_resource_t* resource1
  //   arg2: iree_hal_resource_t* resource2
  //   arg3: iree_hal_resource_t* resource3
  //   return_address: unused
  //   completion_signal: optional, signaled when the release has completed
  IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_RELEASE,

  // Host will mark the device as lost and start returning failures.
  // The provided code and arguments will be included in the failure messages.
  //
  // Signature:
  //   arg0: uint64_t reserved 0
  //   arg1: uint64_t code
  //   arg2: uint64_t error-specific arg0
  //   arg3: uint64_t error-specific arg1
  //   return_address: unused
  //   completion_signal: unused
  IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_ERROR,

  // Host will notify any registered listeners of the semaphore signal.
  //
  // Signature:
  //   arg0: iree_hal_amdgpu_device_semaphore_t* semaphore
  //   arg1: uint64_t payload
  //   arg2: unused
  //   arg3: unused
  //   return_address: unused
  //   completion_signal: unused
  IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_SIGNAL,

  // Host will flush all committed trace events in the given trace buffer.
  //
  // Signature:
  //   arg0: iree_hal_amdgpu_device_trace_buffer_t* trace_buffer
  //   arg1: unused
  //   arg2: unused
  //   arg3: unused
  //   return_address: unused
  //   completion_signal: optional, signaled when the flush has completed
  IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POST_TRACE_FLUSH,
};

// Represents the host runtime thread that is managing host interrupts.
// One or more schedulers may share a single host queue. An host calls that need
// to identify the scheduler or scheduler-related resources must pass those as
// arguments.
//
// NOTE: for now this is just the HSA soft queue used by the host thread. It may
// have multiple producers if there are multiple schedulers sharing the same
// host queue but only one consumer.
typedef iree_hsa_queue_t iree_hal_amdgpu_device_host_t;

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

#if defined(IREE_OCL_TARGET_DEVICE)

// TODO(benvanik): IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POOL_GROW
// TODO(benvanik): IREE_HAL_AMDGPU_DEVICE_HOST_CALL_POOL_TRIM

// Enqueues a unidirection host agent packet.
// Since this is device->host only operation this acquires only from the agent
// and releases to the entire system so the host agent can observe changes. The
// completion signal is optional and may be `iree_hsa_signal_null()`.
//
// NOTE: the barrier bit is set but the host processing is (today) synchronous
// with respect to other packets and generally only executes in FIFO order with
// respect to what each packet may affect anyway. We could tweak this in the
// future e.g. posts to flush a ringbuffer don't need to block and can be
// eagerly processed. Maybe. For non-post operations we'd rely on queue barrier
// packets.
void iree_hal_amdgpu_device_host_post(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_host_t* IREE_OCL_RESTRICT host,
    uint16_t type, uint64_t return_address, uint64_t arg0, uint64_t arg1,
    uint64_t arg2, uint64_t arg3, iree_hsa_signal_t completion_signal);

// Posts a multi-resource release request to the host.
// The host will call iree_hal_resource_release on each non-NULL resource
// pointer provided. The optional |completion_signal| will be signaled when the
// release has completed.
void iree_hal_amdgpu_device_host_post_release(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_host_t* IREE_OCL_RESTRICT host,
    uint64_t resource0, uint64_t resource1, uint64_t resource2,
    uint64_t resource3, iree_hsa_signal_t completion_signal);

// Posts an error code to the host.
// The provided arguments are appended to the error message emitted.
// After posting an error it may not be possible to continue execution and the
// device is considered "lost".
void iree_hal_amdgpu_device_host_post_error(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_host_t* IREE_OCL_RESTRICT host,
    uint64_t code, uint64_t arg0, uint64_t arg1);

// Posts a semaphore signal notification to the host.
// The order is not guaranteed and by the time the host processes the message
// the semaphore may have already advanced past the specified payload value.
void iree_hal_amdgpu_device_host_post_signal(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_host_t* IREE_OCL_RESTRICT host,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_semaphore_t* IREE_OCL_RESTRICT
        semaphore,
    uint64_t payload);

// Posts a trace flush request to the host for the given trace buffer.
// The host should quickly consume all committed trace events and may do so up
// to the committed write index even if that has advanced since the flush is
// requested. The optional |completion_signal| will be signaled when the
// flush has completed and the read commit offset has advanced.
void iree_hal_amdgpu_device_host_post_trace_flush(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_host_t* IREE_OCL_RESTRICT host,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hsa_signal_t completion_signal);

#endif  // IREE_OCL_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_HOST_H_
