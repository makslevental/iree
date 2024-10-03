// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_ALLOCATOR_H_

#include "iree/hal/drivers/amdgpu/device/buffer.h"
#include "iree/hal/drivers/amdgpu/device/support/opencl.h"
#include "iree/hal/drivers/amdgpu/device/support/queue.h"
#include "iree/hal/drivers/amdgpu/device/support/signal.h"

//===----------------------------------------------------------------------===//
// DO NOT SUBMIT
//===----------------------------------------------------------------------===//

// each device/agent has a set of pools
// pools have types (ringbuffer vs suballocator vs dedicated)
// pool logic happens on device
//
// block ids are dense, preallocated indirection
// each pool has its own block id table
// sizes are pool dependent
//
// iree_hal_amdgpu_device_pool_t
//   agent
//   vtable
//     grow
//     trim
// iree_hal_amdgpu_device_pool_grow(pool, block_id, [allocation], signal)
//   iree_hal_amdgpu_device_pool_allocation_t
//     allocation_size
//     allocation_offset
//     min_alignment
//     handle*
//   grow block
//   perform allocation, populate handle
//   signal completion
// iree_hal_amdgpu_device_pool_trim(pool, block_id, [handle], signal)
//   perform deallocation, clear handle
//   trim block
//   signal completion
//
// iree_hal_amdgpu_device_ringbuffer_pool_t
// iree_hal_amdgpu_device_ringbuffer_pool_block_t
//
// iree_hal_amdgpu_device_suballocator_pool_t
// iree_hal_amdgpu_device_suballocator_pool_block_t
//
// iree_hal_amdgpu_device_dedicated_pool_t
//   blocks[max]
// iree_hal_amdgpu_device_dedicated_pool_block_t
//   status [PENDING_COMMIT, COMMITTED, PENDING_DECOMMIT, DECOMMITTED]
//   device ptr
//   size
// -- alloca --
// << POOL TYPE >> device picks block
//   try to find committed with free space
// if failed to find with free space:
//   scan to find free, CAS DECOMMITTED -> PENDING_COMMIT
// if no free blocks:
//   fail with device->host exceeded post?
//   could make host retry after every trim? need to keep it pumping
//   for now fail!
// << POOL TYPE >> device does pool logic (bitmaps/mumble), calculates offset
// if committing:
//   send device->host grow with block_id
//   host populates block (atomic)
//   host sets PENDING->COMMITTED
//   host: <set handle w/ block_id embedded>
//   host signals completion
// else:
//   device: <set handle w/ block_id embedded>
//   device signals completion
// -- dealloca --
// device has block_id in handle
// << POOL TYPE >> do pool logic (bitmaps/mumble)
// clear handle
// << POOL TYPE >> if policy allows decommit:
//   CAS COMMITTED -> PENDING_DECOMMIT
//   send device->host decommit with block_id
//   host pool deletes memory
//   host sets PENDING->DECOMMITTED
//   host signals completion
// else:
//   device signals completion

// pool growth + fused alloc (if handle != NULL)
// arg0: uint32_t pool
//       uint32_t block; ?
// arg1: reserved
// arg2: uint64_t allocation_size;
// arg3: uint32_t allocation_offset;  // offset into slab for allocation
//       uint32_t min_alignment;
// return: iree_hal_amdgpu_device_allocation_handle_t* handle;

// -- device passes handle just as with dedicated alloc
//    host allocates/grows then populates handle at offset as the initial alloc
// completion signal is for both pool growth *and* the async alloca
// only one device->host->device round trip needed

// pool trim
// arg0: uint32_t pool;
//       uint32_t block; ?
// arg1: reserved
// arg2: reserved
// arg3: reserved
// return: reserved

// -- device passes handle if an update is needed
// completion signal is for both pool trim and async dealloca

//
typedef struct iree_hal_amdgpu_device_allocator_t {
  // host agent
  // device->host queue
  // local pools
  int reserved;
} iree_hal_amdgpu_device_allocator_t;

//===----------------------------------------------------------------------===//
// DO NOT SUBMIT
//===----------------------------------------------------------------------===//

#if defined(IREE_OCL_TARGET_DEVICE)

//

#endif  // IREE_OCL_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_ALLOCATOR_H_
