// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/command_buffer.h"

// iree_hal_amdgpu_device_execution_state_enqueue_begin
//   given command buffer and parent info (queues/etc)
//   allocate execution state from mutable ringbuffer
//   allocate kernarg storage from kernarg ringbuffer
//   set next_block to first
//   copy binding table
//   enqueue aql continue dispatch
// iree_hal_amdgpu_device_execution_state_enqueue_continue
//   process the next block, update next_block with control instruction
//   if next_block: enqueue continue
//   else: enqueue end
// iree_hal_amdgpu_device_execution_state_enqueue_end
//   deallocate kernarg storage
//   deallocate state storage
//   run queue scheduler inline? or enqueue scheduler?
//   inline to reduce latency, but makes kernel more complex/bigger
//   also hurts timing/debugging?
//
// queue execute:
//   begin (allocate/populate)
//   while running: continue
//   end (deallocate, run next)
//
// is queue locked while executing?
// may want to allow multiple executes to run concurrently?
// or other queue operations to proceed?
// if locked then no need to allocate state
// still may need kernarg buffer, but could potentially reuse a large slab of
// max size
//
// could store execution state in queue entry
// then queue entry remains live until execution completed
// ringbuffer needs to block read head
//
// REUSE queue storage
// queue state set to EXECUTING
// queue entry has binding table embedded
// begin: set pointers, set next_block, enqueue continue
// continue: operate in-place on execution state
// end: enqueue scheduler, it checks state and sees done, cleans up
//
// queue sniffing current EXECUTING state allows it to be scheduled out-of-band
// and no-op if executing - knows it will always be enqueued again after
// execution
//
// scheduler enqueue needs an arg for "ENDED"? needs to know when it's the
// command buffer that is enqueuing it for completion
// otherwise need a dispatch just to set the state to execution-completed
//
// fill kernel could be used to set state
// enqueue packet as part of RETURN/etc to reset state
// need a BARRIER bit on the queue schedule to ensure write makes it

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

// Enqueues a dispatch for issuing all commands with the specified block.
// Kernel arguments will be written to state->control_kernargs and the AQL
// packet will be written to the absolute (unwrapped) queue_index. Callers must
// reserve the packet by bumping the queue write_index and signal the queue
// doorbell with the new index.
static void iree_hal_amdgpu_device_command_buffer_emplace_issue_block(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    const uint64_t queue_index) {
#if 0
  // aql dispatch packet enqueue of
  // `iree_hal_amdgpu_device_command_buffer_issue_block`
  // state->issue_block_kernel_object
  // workgroup count based on block->command_count
  // state->issue_block_kernargs populated with [state, block] ptrs + extras

  // kernargs can be written in parallel
  // who reserves queue space? this?
  // called at tail where queue may not be able to be allocated yet
  // need two kernels? or atomic?
  // could have atomic state var for the block queue offset
  // zeroed after last packet enqueued? or epoch?
  //
  // enqueue:
  //   ++state->epoch;
  //   state->queue_offset = -1;
  //
  // want to ensure only first workgroup executing reserves all space
  // want to ensure last workgroup executing does doorbell
  // use a barrier instead of signal?

  // increment write index by 1 + total aql packet count
  // enqueue: issue_block - barrier bit set (ensure kernargs/etc free)
  // (remainder are INVALID and will block processing)
  // knock doorbell with final write index

  // processor gets issue_block
  // issue_block fills all packets
  // as packets are completed (type changed from INVALID) they can process
  // order guarantee mostly ensures in-order processing
  // could optimize latency by filling the first packet in the enqueue? meh

  // DO NOT SUBMIT
  // select count based on flags and target block map
  // (state->flags & DISPATCH) ?
  //     block->query_map.query_ids[command_index].dispatch_id :
  //     block->query_map.query_ids[command_index].control_id
  state->trace_block_query_base_id =
      iree_hal_amdgpu_device_query_ringbuffer_acquire(
          &state->trace_buffer->query_ringbuffer, 0);

  // Reserve space for all of the packets.
  const uint64_t queue_index = 0;
  // block->max_packet_count;

  // Pass the result of the command buffer execution back to the scheduler.
  IREE_OCL_GLOBAL void* control_kernargs[3] = {
      state, block,
      // const uint64_t queue_index) {
  };
  // DO NOT SUBMIT implicit opencl args
  IREE_OCL_GLOBAL void* control_kernarg_ptr = state->control_kernarg_storage;
  memcpy(control_kernarg_ptr, control_kernargs, sizeof(control_kernargs));

  // Construct the control packet.
  // Note that the header is not written until the end so that the
  // hardware command processor stalls until we're done writing.
  const iree_hal_amdgpu_device_kernel_args_t control_args =
      state->kernels->issue_block;
  // DO NOT SUBMIT queue_index
  IREE_OCL_GLOBAL hsa_kernel_dispatch_packet_t* control_packet = NULL;
  control_packet->setup = control_args.setup;
  control_packet->workgroup_size_x = control_args.workgroup_size_x;
  control_packet->workgroup_size_y = control_args.workgroup_size_y;
  control_packet->workgroup_size_z = control_args.workgroup_size_z;
  control_packet->reserved0 = 0;
  control_packet->grid_size_x = 0;  // DO NOT SUBMIT command count?
  control_packet->grid_size_y = 1;
  control_packet->grid_size_z = 1;
  control_packet->private_segment_size = control_args.private_segment_size;
  control_packet->group_segment_size =
      control_args.kernel_args.group_segment_size;
  control_packet->kernel_object = control_args.kernel_object;
  control_packet->kernarg_address = control_kernarg_ptr;
  control_packet->reserved2 = 0;

  // DO NOT SUBMIT
  // control_packet->completion_signal = ;

  // Mark the update packet as ready to execute. The hardware command processor
  // may begin executing it immediately after performing the atomic swap.
  //
  // DO NOT SUBMIT
  // control_packet->header
  // barrier bit
  // fence bits type INVALID -> KERNEL_DISPATCH
#endif
}

void iree_hal_amdgpu_device_command_buffer_enqueue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state) {
  // Execution always begins at the entry block.
  IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t* const block =
      &state->command_buffer->blocks[0];

  // Reserve the next packet in the queue.
  // DO NOT SUBMIT
  // reserve write_index
  const uint64_t queue_index = 0;

  // Emplace the issue_block dispatch packet into the queue.
  // Note that the dispatch may begin executing immediately.
  iree_hal_amdgpu_device_command_buffer_emplace_issue_block(state, block,
                                                            queue_index);

  // Signal the queue doorbell indicating the packet has been updated.
  // DO NOT SUBMIT
  // knock doorbell queue_index
}

//===----------------------------------------------------------------------===//
// Utility Packets
//===----------------------------------------------------------------------===//

// Emits a lightweight barrier packet (no cache management, no-op wait) to and
// associates the optional completion_signal. The packet processor will populate
// the timestamps on the signal after the packet has retired.
static void iree_hal_amdgpu_device_cmd_marker(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    const uint64_t queue_index, iree_hsa_signal_t completion_signal) {
  // DO NOT SUBMIT
  // nop barrier packet (?) with barrier bit set
  // set completion_signal
  // NONE acquire/release
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_BEGIN
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_debug_group_begin_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_debug_group_begin_t*
        IREE_OCL_RESTRICT cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // If tracing is enabled then get the signal used to query timestamps.
  iree_hsa_signal_t completion_signal = iree_hsa_signal_null();
#if IREE_HAL_AMDGPU_TRACING_FEATURES & \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL
  if (execution_query_id != IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID) {
    completion_signal = iree_hal_amdgpu_device_trace_execution_zone_begin(
        state->trace_buffer, execution_query_id, cmd->src_loc);
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

  // Emit a lightweight barrier packet (no cache management, no-op wait) to
  // force the command buffer to execute as if we were capturing timing even if
  // we aren't. This can be useful for native debugging tools and also lets us
  // more easily detect the overhead of tracing.
  iree_hal_amdgpu_device_cmd_marker(state, queue_index, completion_signal);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_END
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_debug_group_end_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_debug_group_end_t*
        IREE_OCL_RESTRICT cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // If tracing is enabled then get the signal used to query timestamps.
  iree_hsa_signal_t completion_signal = iree_hsa_signal_null();
#if IREE_HAL_AMDGPU_TRACING_FEATURES & \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL
  if (execution_query_id != IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID) {
    completion_signal = iree_hal_amdgpu_device_trace_execution_zone_end(
        state->trace_buffer, execution_query_id);
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

  // Emit a lightweight barrier packet (no cache management, no-op wait) to
  // force the command buffer to execute as if we were capturing timing even if
  // we aren't. This can be useful for native debugging tools and also lets us
  // more easily detect the overhead of tracing.
  iree_hal_amdgpu_device_cmd_marker(state, queue_index, completion_signal);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_BARRIER
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_barrier_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_barrier_t*
        IREE_OCL_RESTRICT cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // DO NOT SUBMIT
  // nop barrier packet (?) with barrier bit set
  // still acquire/release based on header
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_SIGNAL_EVENT
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_signal_event_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_signal_event_t*
        IREE_OCL_RESTRICT cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // TODO(benvanik): HSA signal handling.
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_RESET_EVENT
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_reset_event_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_reset_event_t*
        IREE_OCL_RESTRICT cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // TODO(benvanik): HSA signal handling.
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENTS
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_wait_events_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_wait_events_t*
        IREE_OCL_RESTRICT cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // TODO(benvanik): HSA signal handling.
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_FILL_BUFFER
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_fill_buffer_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_fill_buffer_t*
        IREE_OCL_RESTRICT cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  IREE_OCL_GLOBAL void* target_ptr = iree_hal_amdgpu_device_buffer_ref_resolve(
      cmd->target_ref, state->bindings);
  const uint64_t length = cmd->target_ref.length;
  IREE_OCL_GLOBAL uint64_t* kernargs_ptr =
      (IREE_OCL_GLOBAL uint64_t*)(state->execution_kernarg_storage +
                                  cmd->kernarg_offset);
  iree_hal_amdgpu_device_buffer_fill_emplace(
      target_ptr, length, cmd->pattern, cmd->pattern_length, state->kernels,
      kernargs_ptr, state->execution_queue, queue_index);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_COPY_BUFFER
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_copy_buffer_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_copy_buffer_t*
        IREE_OCL_RESTRICT cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  IREE_OCL_GLOBAL const void* source_ptr =
      iree_hal_amdgpu_device_buffer_ref_resolve(cmd->source_ref,
                                                state->bindings);
  IREE_OCL_GLOBAL void* target_ptr = iree_hal_amdgpu_device_buffer_ref_resolve(
      cmd->target_ref, state->bindings);
  const uint64_t length = cmd->target_ref.length;
  IREE_OCL_GLOBAL uint64_t* kernargs_ptr =
      (IREE_OCL_GLOBAL uint64_t*)(state->execution_kernarg_storage +
                                  cmd->kernarg_offset);
  iree_hal_amdgpu_device_buffer_copy_emplace(
      source_ptr, target_ptr, length, state->kernels,
      kernargs_ptr state->execution_queue, queue_index);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH
//===----------------------------------------------------------------------===//

static IREE_OCL_GLOBAL hsa_kernel_dispatch_packet_t*
iree_hal_amdgpu_device_cmd_dispatch_prepare(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_dispatch_t*
        IREE_OCL_RESTRICT cmd,
    IREE_OCL_GLOBAL uint64_t* IREE_OCL_RESTRICT kernarg_ptr,
    const uint64_t queue_index) {
  // DO NOT SUBMIT
  // verify this is correct
  for (uint16_t i = 0; i < cmd->binding_count; ++i) {
    ((IREE_OCL_GLOBAL uint64_t*)kernarg_ptr)[i] =
        iree_hal_amdgpu_device_buffer_ref_resolve(cmd->bindings[i],
                                                  state->bindings);
  }
  memcpy(kernarg_ptr + cmd->binding_count, cmd->constants,
         cmd->constant_count * sizeof(uint32_t));

  // Construct the dispatch packet based on the template embedded in the command
  // buffer. Note that the header is not written until the end so that the
  // hardware command processor stalls until we're done writing.
  // DO NOT SUBMIT queue_index
  IREE_OCL_GLOBAL hsa_kernel_dispatch_packet_t* dispatch_packet = NULL;
  const iree_hal_amdgpu_device_kernel_args_t dispatch_args =
      cmd->packet.kernel_args;
  dispatch_packet->setup = dispatch_args.setup;
  dispatch_packet->workgroup_size_x = dispatch_args.workgroup_size_x;
  dispatch_packet->workgroup_size_y = dispatch_args.workgroup_size_y;
  dispatch_packet->workgroup_size_z = dispatch_args.workgroup_size_z;
  dispatch_packet->reserved0 = 0;
  dispatch_packet->private_segment_size = dispatch_args.private_segment_size;
  dispatch_packet->group_segment_size = dispatch_args.group_segment_size;
  dispatch_packet->kernel_object = dispatch_args.kernel_object;
  dispatch_packet->kernarg_address = kernarg_ptr;
  dispatch_packet->reserved2 = 0;

  // DO NOT SUBMIT
  // dispatch_packet->completion_signal = ;

  // Resolve the workgroup count (if possible).
  if (cmd->flags & IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_STATIC) {
    // Workgroup count is indirect but statically available and can be resolved
    // during issue. This is the common case where the workgroup count is stored
    // in a uniform buffer by the launcher and it allows us to avoid any
    // additional dispatch overhead.
    IREE_OCL_GLOBAL const uint32_t* workgroups_ptr =
        iree_hal_amdgpu_device_buffer_ref_resolve(cmd->workgroups_ref,
                                                  state->bindings);
    dispatch_packet->grid_size_x = workgroups_ptr[0];
    dispatch_packet->grid_size_y = workgroups_ptr[1];
    dispatch_packet->grid_size_z = workgroups_ptr[2];
  } else {
    // Workgroup count is constant.
    dispatch_packet->grid_size_x = cmd->packet.grid_size_x;
    dispatch_packet->grid_size_y = cmd->packet.grid_size_y;
    dispatch_packet->grid_size_z = cmd->packet.grid_size_z;
  }

  // NOTE: we return the packet without having updated the header. The caller
  // is responsible for calling iree_hal_amdgpu_device_cmd_dispatch_mark_ready
  // when it is ready for the hardware command processor to pick up the packet.
  return dispatch_packet;
}

static IREE_OCL_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_cmd_dispatch_commit(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_dispatch_t*
        IREE_OCL_RESTRICT cmd,
    IREE_OCL_GLOBAL hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT
        dispatch_packet) {
  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  if (cmd->header.flags & IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER) {
    header |= 1 << HSA_PACKET_HEADER_BARRIER;
  }
  header |= HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  uint32_t header_type = header | (dispatch_packet->setup << 16);

  //
  // barrier bit set? derive from cmd->header.flags &
  // IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER?
  //
  // fence scopes?

  // type -> KERNEL_DISPATCH
}

static void iree_hal_amdgpu_device_cmd_dispatch_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_dispatch_t*
        IREE_OCL_RESTRICT cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // Enqueue the dispatch packet but do not mark it as ready yet.
  IREE_OCL_GLOBAL uint64_t* kernarg_ptr =
      (IREE_OCL_GLOBAL uint64_t*)(state->execution_kernarg_storage +
                                  cmd->kernarg_offset);
  IREE_OCL_GLOBAL hsa_kernel_dispatch_packet_t* dispatch_packet =
      iree_hal_amdgpu_device_cmd_dispatch_prepare(state, block, cmd,
                                                  kernarg_ptr, queue_index);

  // Mark the dispatch as complete and allow the hardware command processor to
  // process it.
  iree_hal_amdgpu_device_cmd_dispatch_commit(cmd, dispatch_packet);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_INDIRECT_DYNAMIC
//===----------------------------------------------------------------------===//

__kernel IREE_OCL_ATTRIBUTE_SINGLE_WORK_ITEM void
iree_hal_amdgpu_device_command_buffer_workgroup_count_update(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_dispatch_t*
        IREE_OCL_RESTRICT cmd,
    IREE_OCL_GLOBAL const uint32_t* IREE_OCL_RESTRICT workgroups_ptr,
    IREE_OCL_GLOBAL hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT
        dispatch_packet) {
  // Read the uint32_t[3] workgroup count buffer and update the packet in-place.
  dispatch_packet->grid_size_x = workgroups_ptr[0];
  dispatch_packet->grid_size_y = workgroups_ptr[1];
  dispatch_packet->grid_size_z = workgroups_ptr[2];

  // Now that the packet has been updated we can mark it as ready so that the
  // hardware command processor can take it.
  iree_hal_amdgpu_device_cmd_dispatch_commit(cmd, dispatch_packet);
}

static void iree_hal_amdgpu_device_cmd_dispatch_indirect_dynamic_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_dispatch_t*
        IREE_OCL_RESTRICT cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  const uint32_t update_index = queue_index;
  const uint32_t dispatch_index = update_index + 1;

  // Enqueue the dispatch packet but do not mark it as ready yet.
  // We do this first so that if the workgroup count update dispatch begins
  // executing while we're still running we want it to have valid data to
  // manipulate.
  IREE_OCL_GLOBAL uint64_t* dispatch_kernarg_ptr =
      (IREE_OCL_GLOBAL uint64_t*)(state->execution_kernarg_storage +
                                  cmd->kernarg_offset +
                                  IREE_HAL_AMDGPU_DEVICE_WORKGROUP_COUNT_UPDATE_KERNARG_SIZE);
  IREE_OCL_GLOBAL hsa_kernel_dispatch_packet_t* dispatch_packet =
      iree_hal_amdgpu_device_cmd_dispatch_prepare(
          state, block, cmd, dispatch_kernarg_ptr, dispatch_index);

  // Workgroup count is dynamic and must be resolved just prior to executing
  // the dispatch. There's no native AQL dispatch behavior to enable this so
  // we have to emulate it by enqueuing a builtin that performs the
  // indirection and overwrites the packet memory directly.
  IREE_OCL_GLOBAL void* update_kernargs[3] = {
      cmd,
      iree_hal_amdgpu_device_buffer_ref_resolve(cmd->workgroups_ref,
                                                state->bindings),
      dispatch_packet,
  };
  // DO NOT SUBMIT implicit opencl args
  IREE_OCL_GLOBAL void* update_kernarg_ptr =
      state->execution_kernarg_storage + cmd->kernarg_offset;
  memcpy(update_kernarg_ptr, update_kernargs, sizeof(update_kernargs));

  // Construct the update packet.
  // Note that the header is not written until the end so that the
  // hardware command processor stalls until we're done writing.
  const iree_hal_amdgpu_device_kernel_args_t update_args =
      state->kernels->workgroup_count_update;
  // DO NOT SUBMIT queue_index
  IREE_OCL_GLOBAL hsa_kernel_dispatch_packet_t* update_packet = NULL;
  update_packet->setup = update_args.setup;
  update_packet->workgroup_size_x = update_args.workgroup_size_x;
  update_packet->workgroup_size_y = update_args.workgroup_size_y;
  update_packet->workgroup_size_z = update_args.workgroup_size_z;
  update_packet->reserved0 = 0;
  update_packet->grid_size_x = 1;
  update_packet->grid_size_y = 1;
  update_packet->grid_size_z = 1;
  update_packet->private_segment_size = update_args.private_segment_size;
  update_packet->group_segment_size =
      update_args.kernel_args.group_segment_size;
  update_packet->kernel_object = update_args.kernel_object;
  update_packet->kernarg_address = update_kernarg_ptr;
  update_packet->reserved2 = 0;

  // DO NOT SUBMIT
  // update_packet->completion_signal = ;

  // Mark the update packet as ready to execute. The hardware command processor
  // may begin executing it immediately after performing the atomic swap.
  //
  // DO NOT SUBMIT
  // update_packet->header
  // barrier bit
  // fence bits type INVALID -> KERNEL_DISPATCH

  // NOTE: the following dispatch packet is still marked INVALID and is only
  // changed after the update dispatch completes. The hardware command processor
  // should process the update (as we change it from INVALID here) and then
  // block before reading the contents of the dispatch packet.
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_branch_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_branch_t* IREE_OCL_RESTRICT
        cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // Direct branches are like tail calls and can simply begin issuing the
  // following block. The kernargs are stored in state->control_kernargs so that
  // the issue_block can completely overwrite the kernargs_storage. Command
  // buffer issue has already bumped the write_index and all we need to do is
  // populate the packet.
  //
  // NOTE: we implicitly assume
  // IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER but need not do so
  // (technically) when continuing within the same command buffer. Performing a
  // barrier is a more conservative operation and may mask compiler/command
  // buffer construction issues with the more strict execution model but in
  // practice is unlikely to have an appreciable effect on latency.

  // DO NOT SUBMIT
  // need to place this on the scheduler queue
  // iree_hal_amdgpu_device_command_buffer_emplace_issue_block(
  //     state, &state->command_buffer->blocks[cmd->target_block], queue_index);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_RETURN
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_return_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_return_t* IREE_OCL_RESTRICT
        cmd,
    const uint64_t queue_index,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // TODO(benvanik): handle call stacks when nesting command buffers. For now a
  // return is always going back to the queue scheduler and can be enqueued as
  // such.

  // DO NOT SUBMIT
  // barrier packet that signals the top-level state->completion_signal
  // iree_hal_amdgpu_device_barrier_and_emplace(deps, state->completion_signal)
  // or any other way to have a full barrier packet? AND with 0 deps?

  // Enqueue the parent queue scheduler tick.
  // It will clean up the command buffer execution state and resume
  // processing queue entries.
  iree_hal_amdgpu_device_queue_scheduler_enqueue(
      state->scheduler,
      IREE_HAL_AMDGPU_DEVICE_QUEUE_SCHEDULING_REASON_COMMAND_BUFFER_RETURN,
      (uint64_t)state);
}

//===----------------------------------------------------------------------===//
// Command issue
//===----------------------------------------------------------------------===//

// Issues a block of commands in parallel.
// Each work item processes a single command. Each command in the block contains
// a relative offset into the queue where AQL packets should be placed and must
// fill all packets that were declared when the command buffer was recorded
// (even if they are no-oped).
//
// This relies on the AQL queue mechanics defined in section 2.8.3 of the HSA
// System Architecture Specification. The parent enqueuing this kernel reserves
// sufficient queue space for all AQL packets and bumps the write_index to the
// end of the block. Each command processed combines the base queue index
// provided with the per-command relative offset and performs the required queue
// masking to get the final packet pointer. Packets are written by populating
// all kernel arguments (if any), populating the packet fields, and finally
// atomically changing the packet type from INVALID to (likely) KERNEL_DISPATCH.
// Even though the write_index of the queue was bumped to the end the queue
// processor is required to block on the first packet it finds with an INVALID
// type and as such we don't require ordering guarantees on the packet
// population. It's of course better if the first packet completes first so that
// the queue processor can launch it and that will often be the case given that
// HSA mandates that workgroups with lower indices are scheduled to resources
// before those with higher ones.
__kernel void iree_hal_amdgpu_device_command_buffer_issue_block(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_execution_state_t* IREE_OCL_RESTRICT
        state,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_command_block_t*
        IREE_OCL_RESTRICT block,
    const uint64_t base_queue_index) {
  // Each invocation handles a single command in the block.
  const uint32_t command_ordinal = iree_hal_amdgpu_device_global_id_x();
  if (command_ordinal >= command_count) return;

  // When device control or dispatch tracing is enabled we need to pass a query
  // signal with any work we do. Prior to the block starting execution we
  // acquire a range for all commands on the scheduler queue and store it in
  // state->trace_block_query_base_id. Here we then take that base ID and add a
  // relative offset that was precomputed when the command buffer was recorded.
  // This allows us to support sparse/partial queries and still issue in
  // parallel while respecting the required query ordering.
  //
  // There's probably a much simpler way of doing this - not needing all this
  // branching per command or the precomputed query map would be nice.
  iree_hal_amdgpu_trace_execution_query_id_t execution_query_id =
      IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID;
#if IREE_HAL_AMDGPU_TRACING_FEATURES & \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL
  const iree_hal_amdgpu_device_command_query_id_t command_query_id =
      block->query_map.query_ids[command_ordinal];
  if ((state->flags & IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_DISPATCH) &&
      command_query_id.dispatch_id !=
          IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID) {
    execution_query_id = iree_hal_amdgpu_device_query_ringbuffer_query_id(
        &state->trace_buffer.query_ringbuffer,
        state->trace_block_query_base_id + command_query_id.dispatch_id);
  } else if ((state->flags &
              IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_CONTROL) &&
             command_query_id.control_id !=
                 IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID) {
    execution_query_id = iree_hal_amdgpu_device_query_ringbuffer_query_id(
        &state->trace_buffer.query_ringbuffer,
        state->trace_block_query_base_id + command_query_id.control_id);
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

  // Tail-call into the command handler.
  IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_t* IREE_OCL_RESTRICT cmd =
      &block->commands[command_ordinal];
  const uint64_t queue_index = base_queue_index + cmd->header.packet_offset;
  switch (cmd->header.type) {
    default:
      return;  // no-op
    case IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_BEGIN:
      return iree_hal_amdgpu_device_cmd_debug_group_begin_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_debug_group_begin_t*)
              cmd,
          queue_index, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_END:
      return iree_hal_amdgpu_device_cmd_debug_group_end_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_debug_group_end_t*)
              cmd,
          queue_index, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_BARRIER:
      return iree_hal_amdgpu_device_cmd_barrier_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_barrier_t*)cmd,
          queue_index, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_SIGNAL_EVENT:
      return iree_hal_amdgpu_device_cmd_signal_event_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_signal_event_t*)cmd,
          queue_index, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_RESET_EVENT:
      return iree_hal_amdgpu_device_cmd_reset_event_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_reset_event_t*)cmd,
          queue_index, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENTS:
      return iree_hal_amdgpu_device_cmd_wait_events_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_wait_events_t*)cmd,
          queue_index, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_FILL_BUFFER:
      return iree_hal_amdgpu_device_cmd_fill_buffer_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_fill_buffer_t*)cmd,
          queue_index, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_COPY_BUFFER:
      return iree_hal_amdgpu_device_cmd_copy_buffer_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_copy_buffer_t*)cmd,
          queue_index, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH:
      return iree_hal_amdgpu_device_cmd_dispatch_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_dispatch_t*)cmd,
          queue_index, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_INDIRECT_DYNAMIC:
      return iree_hal_amdgpu_device_cmd_dispatch_indirect_dynamic_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_dispatch_t*)cmd,
          queue_index, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH:
      return iree_hal_amdgpu_device_cmd_branch_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_branch_t*)cmd,
          queue_index, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_RETURN:
      return iree_hal_amdgpu_device_cmd_return_issue(
          state, block,
          (IREE_OCL_GLOBAL const iree_hal_amdgpu_device_cmd_return_t*)cmd,
          queue_index, execution_query_id);
  }
  // NOTE: we need the above switch to end in tail calls in all cases.
}
