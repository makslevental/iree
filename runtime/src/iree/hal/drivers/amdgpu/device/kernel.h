// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNEL_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNEL_H_

#include "iree/hal/drivers/amdgpu/device/support/opencl.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_kernel_args_t
//===----------------------------------------------------------------------===//

// Kernel arguments used for fixed-size kernels.
// This must match what the kernel was compiled to support.
typedef struct iree_hal_amdgpu_device_kernel_args_s {
  // Opaque handle to the kernel object to execute.
  uint64_t kernel_object;
  // hsa_kernel_dispatch_packet_setup_t (grid dimension count).
  uint16_t setup;
  // XYZ dimensions of work-group, in work-items. Must be greater than 0.
  // If the grid has fewer than 3 dimensions the unused must be 1.
  uint16_t workgroup_size[3];
  // Size in bytes of private memory allocation request (per work-item).
  uint32_t private_segment_size;
  // Size in bytes of group memory allocation request (per work-group). Must
  // not be less than the sum of the group memory used by the kernel (and the
  // functions it calls directly or indirectly) and the dynamically allocated
  // group segment variables.
  uint32_t group_segment_size;
  // Allocated source location in host memory. Inaccessible and only here to
  // feed back to the host for trace processing.
  uint64_t trace_src_loc;
} iree_hal_amdgpu_device_kernel_args_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_kernel_implicit_args_t
//===----------------------------------------------------------------------===//

// Implicit kernel arguments passed to OpenCL/HIP kernels that use them.
// Not all kernels require this and the metadata needs to be checked to detect
// its use (or if the total kernargs size is > what we think it should be).
// Layout-wise explicit args always start at offset 0 and implicit args follow
// those with 8-byte alignment.
//
// The metadata will contain exact fields and offsets and most driver code will
// carefully walk to detect, align, pad, and write each field:
// OpenCL/HIP: (`amd::KernelParameterDescriptor`...)
// https://github.com/ROCm/clr/blob/5da72f9d524420c43fe3eee44b11ac875d884e0f/rocclr/device/rocm/rocvirtual.cpp#L3197
//
// This complex construction was required once upon a time. The LLVM code
// producing the kernargs layout and metadata handles these cases much more
// simply by only ever truncating the implicit args at the last used field:
// https://github.com/llvm/llvm-project/blob/7f1b465c6ae476e59dc90652d58fc648932d23b1/llvm/lib/Target/AMDGPU/AMDGPUHSAMetadataStreamer.cpp#L389
//
// Then at some point in time someone was like "meh, who cares about optimizing"
// and decided to include all of them always 🤦:
// https://github.com/llvm/llvm-project/blob/7f1b465c6ae476e59dc90652d58fc648932d23b1/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp#L299
//
// What this means in practice is that if any implicit arg is used then all will
// be included and declared in the metadata even if only one is actually read by
// the kernel -- there's no way for us to know. In the ideal case none of them
// are read and the kernel function gets the `amdgpu-no-implicitarg-ptr` attr
// so that all of them can be skipped. Otherwise we reserve the 256 bytes and
// just splat them all in. This at least keeps our code simple relative to all
// the implementations that enumerate the metadata and write args one at a time.
// We really should try to force `amdgpu-no-implicitarg-ptr` when we generate
// code, though.
//
// For our OpenCL runtime device code we have less freedom and may always need
// to support implicit args. We try to avoid it but quite a few innocuous things
// can result in compiler builtins that cause it to be emitted.
typedef struct IREE_OCL_ALIGNAS(8)
    iree_hal_amdgpu_device_kernel_implicit_args_s {
  // Grid dispatch workgroup count.
  // Some languages, such as OpenCL, support a last workgroup in each
  // dimension being partial. This count only includes the non-partial
  // workgroup count. This is not the same as the value in the AQL dispatch
  // packet, which has the grid size in workitems.
  //
  // Represented in metadata as:
  //   hidden_block_count_x
  //   hidden_block_count_y
  //   hidden_block_count_z
  uint32_t block_count[3];  // + 0/4/8

  // Grid dispatch workgroup size.
  // This size only applies to the non-partial workgroups. This is the same
  // value as the AQL dispatch packet workgroup size.
  //
  // Represented in metadata as:
  //   hidden_group_size_x
  //   hidden_group_size_y
  //   hidden_group_size_z
  uint16_t group_size[3];  // + 12/14/16

  // Grid dispatch work group size of the partial work group, if it exists.
  // Any dimension that does not exist must be 0.
  //
  // Represented in metadata as:
  //   hidden_remainder_x
  //   hidden_remainder_y
  //   hidden_remainder_z
  uint16_t remainder[3];  // + 18/20/22

  uint64_t reserved0;  // + 24 hidden_tool_correlation_id
  uint64_t reserved1;  // + 32

  // OpenCL grid dispatch global offset.
  //
  // Represented in metadata as:
  //   hidden_global_offset_x
  //   hidden_global_offset_y
  //   hidden_global_offset_z
  uint64_t global_offset[3];  // + 40/48/56

  // Grid dispatch dimensionality. This is the same value as the AQL
  // dispatch packet dimensionality. Must be a value between 1 and 3.
  //
  // Represented in metadata as:
  //   hidden_grid_dims
  uint16_t grid_dims;  // + 64
} iree_hal_amdgpu_device_kernel_implicit_args_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_kernels_t
//===----------------------------------------------------------------------===//

// Opaque handles used to launch builtin kernels.
// Stored on the command buffer as they are constant for the lifetime of the
// program and we may have command buffers opt into different DMA modes.
typedef struct iree_hal_amdgpu_device_kernels_s {
  // `iree_hal_amdgpu_device_queue_scheduler_tick` kernel.
  iree_hal_amdgpu_device_kernel_args_t scheduler_tick;
  // `iree_hal_amdgpu_device_command_buffer_issue_block` kernel.
  iree_hal_amdgpu_device_kernel_args_t issue_block;
  // `iree_hal_amdgpu_device_command_buffer_workgroup_count_update` kernel.
  iree_hal_amdgpu_device_kernel_args_t workgroup_count_update;
  // Kernels used to implement DMA-like operations.
  struct {
    iree_hal_amdgpu_device_kernel_args_t
        fill_x1;  // iree_hal_amdgpu_device_buffer_fill_x1
    iree_hal_amdgpu_device_kernel_args_t
        fill_x2;  // iree_hal_amdgpu_device_buffer_fill_x2
    iree_hal_amdgpu_device_kernel_args_t
        fill_x4;  // iree_hal_amdgpu_device_buffer_fill_x4
    iree_hal_amdgpu_device_kernel_args_t
        fill_x8;  // iree_hal_amdgpu_device_buffer_fill_x8
    iree_hal_amdgpu_device_kernel_args_t
        copy_x1;  // iree_hal_amdgpu_device_buffer_copy_x1
    iree_hal_amdgpu_device_kernel_args_t
        copy_x2;  // iree_hal_amdgpu_device_buffer_copy_x2
    iree_hal_amdgpu_device_kernel_args_t
        copy_x4;  // iree_hal_amdgpu_device_buffer_copy_x4
    iree_hal_amdgpu_device_kernel_args_t
        copy_x8;  // iree_hal_amdgpu_device_buffer_copy_x8
    iree_hal_amdgpu_device_kernel_args_t
        copy_x64;  // iree_hal_amdgpu_device_buffer_copy_x64
  } blit;
} iree_hal_amdgpu_device_kernels_t;

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_KERNEL_H_
