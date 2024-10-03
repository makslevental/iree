// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/scheduler.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_queue_list_t
//===----------------------------------------------------------------------===//

// An singly-linked intrusive list of queue entries.
// This uses the `list_next` field of each entry and requires that an entry only
// be in one list at a time. Because we use these lists to manage wait and run
// lists and entries can only be in one at a time we don't run into collisions.
//
// List order is determined by how entries are inserted. Producers must ensure
// they are consistent about either inserting in FIFO list order or FIFO
// submission order (using queue entry epochs).
//
// Thread-compatible; expected to only be accessed locally.
// Zero initialization compatible.
typedef struct iree_hal_amdgpu_device_queue_list_t {
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* head;
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* tail;
} iree_hal_amdgpu_device_queue_list_t;

// Appends the given |entry| to the end of the |list|. Exclusively using this
// will make the list be treated like a queue with respect to the list
// manipulations but will not order entries with respect to when they were
// originally submitted.
static inline void iree_hal_amdgpu_device_queue_list_append(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_list_t* IREE_OCL_RESTRICT list,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        IREE_OCL_RESTRICT entry) {
  entry->list_next = NULL;
  if (list->head == NULL) {
    list->head = entry;
    list->tail = entry;
  } else {
    list->tail->list_next = entry;
    list->tail = entry;
  }
}

// Inserts the given |entry| in the |list| immediately before the first entry
// with a larger epoch. Exclusively using this will make the list be treated
// like a FIFO.
static inline void iree_hal_amdgpu_device_queue_list_insert(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_list_t* IREE_OCL_RESTRICT list,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        IREE_OCL_RESTRICT entry) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_cursor =
      list->head;
  if (list->head == NULL) {
    // First entry in the list.
    list->head = entry;
    list->tail = entry;
  } else {
    // Find the insertion point and splice in. Insert immediately prior to the
    // next epoch greater than the requested (or the tail).
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_prev =
        NULL;
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_cursor =
        list->head;
    while (list_cursor != NULL) {
      if (list_cursor == list->tail) {
        list_cursor->list_next = entry;
        list->tail = entry;
        break;
      } else if (list_cursor->epoch > entry->epoch) {
        if (list_prev != NULL) {
          if (list_prev->list_next != NULL) {
            entry->list_next = list_prev->list_next;
          }
          list_prev->list_next = entry;
        }
        break;
      }
      list_prev = list_cursor;
      list_cursor = list_cursor->list_next;
    }
  }
}

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

static IREE_OCL_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_queue_issue_initialize(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_device_queue_initialize_args_t*
        IREE_OCL_RESTRICT args) {
  // Initialize the signal pool with the provided HSA signals.
  iree_hal_amdgpu_device_signal_pool_initialize(
      scheduler->signal_pool, args->signal_count, args->signals);
}

static IREE_OCL_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_queue_issue_deinitialize(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_device_queue_deinitialize_args_t*
        IREE_OCL_RESTRICT args) {
  //
}

static IREE_OCL_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_queue_issue_alloca(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_device_queue_alloca_args_t*
        IREE_OCL_RESTRICT args) {
  // check satisfied
  // lookup pool
  // switch on pool type
  // call pool handler method
  //   if host needed then pass queue
  //   set suspend state?
}

static IREE_OCL_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_queue_issue_dealloca(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_device_queue_dealloca_args_t*
        IREE_OCL_RESTRICT args) {
  //
}

static IREE_OCL_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_queue_issue_fill(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_device_queue_fill_args_t*
        IREE_OCL_RESTRICT args) {
  // check satisfied
  // enqueue blit kernel (today)
  // barrier + enqueue signal (if needed)
}

static IREE_OCL_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_queue_issue_copy(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_device_queue_copy_args_t*
        IREE_OCL_RESTRICT args) {
  // check satisfied
  // enqueue blit kernel (today)
  // barrier + enqueue signal (if needed)
}

static IREE_OCL_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_queue_issue_execute(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_device_queue_execute_args_t*
        IREE_OCL_RESTRICT args) {
  //
  // enqueue command buffer launch kernel
  //   could do first chunk inline?
  // barrier + enqueue signal (if needed)
}

static IREE_OCL_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_queue_issue_barrier(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_device_queue_barrier_args_t*
        IREE_OCL_RESTRICT args) {
  //
  // barrier + enqueue signal (if needed)
}

//===----------------------------------------------------------------------===//
// DO NOT SUBMIT
//===----------------------------------------------------------------------===//

// A fixed-size list of pending waits.
//
// Thread-compatible; only intended to be accessed by the owning scheduler.

// when wait satisfied:
//   clear semaphore required bit
//   move to incoming queue if more bits set
//   move to run list of no bits set (all waits satisfied)

//
// iree_hal_amdgpu_device_queue_entry_header_t
//   contains storage for wait list?
//   wait list is linked, contains all waiting entries
//   only one registered wake at a time per entry
//   wait list is run down, move to run list

// Accepts all incoming queue operations from the HSA softqueue.
// Operations are immediately moved into the scheduler run list if they have
// no dependencies and otherwise are put in the scheduler wait list to be
// evaluated during the tick.
// Returns true if any operations were added to the wait list.
static bool iree_hal_amdgpu_device_queue_scheduler_accept_incoming(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler) {
  // drain softqueue
  // move to waitlist or runlist
  // if no wait semas then
  // iree_hal_amdgpu_device_queue_list_insert(run_list, entry);
  // set epoch++
  return false;
}

// Checks each waiting queue entry for whether it is able to be run.
// Maintains the per-semaphore wake lists and does other bookkeeping as-needed.
// Upon return the scheduler run list may have new entries in it.
static void iree_hal_amdgpu_device_queue_scheduler_check_wait_list(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_prev = NULL;
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_cursor =
      scheduler->wait_list.head;
  while (list_cursor != NULL) {
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_next =
        list_cursor->list_next;

    IREE_OCL_GLOBAL iree_hal_amdgpu_device_semaphore_list_t* semaphore_list =
        list_cursor->wait_semaphore_list;
    do {
      // DO NOT SUBMIT use wake list lock?
      // try-insert?
      // *must* put in wake list if not already there
      // *must* do so atomically (if not in list insert and not reached, else
      // do not insert and remove wait)
      //-
      // lookup/reserve
      // the local wait set is thread-compat, can just search and find
      // if fail to find then raise error here in scheduler
      // if reserved then set

      // Get the semaphore wake list for the head wait. Note that waits are
      // unordered so this is just "the next wait to check" and not "the first
      // wait that must be satisfied".
      IREE_OCL_GLOBAL iree_hal_amdgpu_device_semaphore_t* semaphore =
          semaphore_list->entries[0].semaphore;

      // Reserve (or find) the wake list entry in the scheduler pool.
      // We may already be registered to wait on the semaphore in which case
      // we'll no-op this check or modify the minimum required value if this new
      // wait happens to be less than the old one. If not already waiting the
      // entry we get back will be initialized for use.
      IREE_OCL_GLOBAL iree_hal_amdgpu_wake_list_entry_t* wake_list_entry =
          iree_hal_amdgpu_wake_pool_reserve(&scheduler->wake_pool, semaphore);

      // Break on the first wait that isn't satisfied - we only need one to
      // track as the barrier is a wait-all and so long as any single wait is
      // not satisfied we can't progress.
      //
      // This operation takes the lock on the target semaphore wake list and if
      // it returns true it means that this scheduler will be woken when the
      // requested value is reached. If it returns false we know the value is
      // already satisfied and can treat the wait as resolved.
      const bool is_waiting = iree_hal_amdgpu_device_semaphore_update_wait(
          semaphore, wake_list_entry, semaphore_list->entries[0].payload);
      if (is_waiting) {
        // Already waiting or now waiting - either way, we're blocked until the
        // wake resolves. Stop processing the waits for this entry and move on
        // to the next.
        list_prev = list_cursor;
        break;
      } else {
        // Not waiting - release the reserved wake list entry.
        iree_hal_amdgpu_wake_pool_release(&scheduler->wake_pool,
                                          wake_list_entry);
      }

      // Remove the semaphore from the wait list by swapping in the last
      // element. If this was the last wait in the list then the operation is
      // ready to run and can be moved to the run list.
      if (semaphore_list->count == 1) {
        // Last wait - move to run list.
        // Note we leave the list_prev pointer at the prior entry so that the
        // next wait list entry loop will be able to remove itself if needed.
        if (list_prev != NULL) {
          list_prev->list_next = list_next;
        } else {
          scheduler->wait_list.head = list_next;
        }
        if (list_next == NULL) {
          scheduler->wait_list.tail = NULL;
        }
        list_cursor->list_next = NULL;
        iree_hal_amdgpu_device_queue_list_insert(&scheduler->run_list,
                                                 list_cursor);
        break;
      } else {
        // Remaining waits - swap in one (order doesn't matter) and retry.
        semaphore_list->entries[0] =
            semaphore_list->entries[semaphore_list->count - 1];
        --semaphore_list->count;
        continue;
      }
    } while (semaphore_list->count > 0);

    // NOTE: list_prev set above based on whether the entry is moved to the run
    // list or not.
    list_cursor = list_next;
  }
}

static void iree_hal_amdgpu_device_queue_scheduler_tick(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler) {
  // DO NOT SUBMIT
  //   clear scheduler pending flag first

  // Accept all incoming queue operations from the HSA softqueue.
  // This may immediately place operations in the run list if they have no
  // dependencies or are known to have been satisfied. If any entries are added
  // to the wait list then we'll do a full verification below.
  bool check_wait_list =
      iree_hal_amdgpu_device_queue_scheduler_accept_incoming(scheduler);

  // Quickly scan the wait set to see if any semaphore we're waiting on has
  // changed in a way that may wake one of our wait list entries.
  // DO NOT SUBMIT
  // for now this just forces a full check each tick
  // if (wake_pool.slots[i].wake_entry.last_value >
  //     wake_pool.slots[i].wake_entry.minimum_value) {
  //   check_wait_list = true;
  //   break;
  // }
  check_wait_list = true;

  // Refresh the wait list by checking the leading wait of each entry.
  // If the leading wait has been satisfied then we can move on to the next wait
  // and if all waits are satisfied the entry is moved to the run list.
  if (check_wait_list) {
    iree_hal_amdgpu_device_queue_scheduler_check_wait_list(scheduler);
  }

  // Drain the run list and issue all pending queue operations.
  // Note that we accumulate targets that need to be woken and flush them after
  // retiring commands.
  bool self_wake = false;
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* run_entry =
      scheduler->run_list.head;
  while (run_entry != NULL) {
    // Issue the ready-to-run queue entry. Provide the wake set but note that
    // the operation may be asynchronous and not wake anything yet.
    iree_hal_amdgpu_device_queue_issue(scheduler, run_entry,
                                       &scheduler->wake_set);

    // Notifies all targets that may now be able to progress due to work
    // completed by the prior issue. If self_wake is true it means that we
    // ourselves have new work and should restart processing after the run list
    // is empty.
    self_wake =
        iree_hal_amdgpu_wake_set_flush(&scheduler->wake_set) || self_wake;

    run_entry = run_entry->list_next;
  }
  scheduler->run_list.head = scheduler->run_list.tail = NULL;

  // Flush the trace buffer, if needed.
  // This will contain any trace events emitted during this tick as well as any
  // imported from command buffers. The host may be notified with an interrupt.
  if (iree_hal_amdgpu_device_trace_commit_range(&scheduler->trace_buffer)) {
    // DO NOT SUBMIT host post with trace buffer handle
  }

  // To give the hardware queue some time to breathe we re-enqueue ourselves.
  // This may increase latency but makes debugging easier and ensures we don't
  // end up in an infinite loop within the tick.
  if (self_wake) {
    iree_hal_amdgpu_device_queue_scheduler_enqueue(
        scheduler,
        IREE_HAL_AMDGPU_DEVICE_QUEUE_SCHEDULING_REASON_WORK_AVAILABLE, 0);
  }
}

// scheduler init
// // INIT
// iree_hal_amdgpu_wake_set_t wake_set;
// iree_hal_amdgpu_wake_target_t self = {
//     .scheduler = scheduler,
// };
// iree_hal_amdgpu_wake_set_initialize(self, &scheduler->wake_set);
// iree_hal_amdgpu_wake_pool_initialize

// Run list is only used within this tick so we keep it local.
// Entries are moved to the run list as we accept incoming entries that are
// ready immediately (no waits) or poll waiting entries and find they are
// ready. Ownership is transferred to the list and we must drain it prior to
// exiting the tick.
// DO NOT SUBMIT initializer
// iree_hal_amdgpu_device_queue_list_t run_list = {
//     .head = NULL,
//     .tail = NULL,
// };
// iree_hal_amdgpu_device_queue_list_t wait_list;
//

// atomic flag for "pending schedule" (barrier on softqueue + scheduler
// enqueued)
// barrier on incoming softqueue -> scheduler
// host/devices can enqueue incoming
// could self-enqueue for continuation
// need to clear flag before processing
// may get spurious wakes if new work comes in while processing but it's handled
//
// hsa_queue_t* mailbox;
//

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

__kernel IREE_OCL_ATTRIBUTE_SINGLE_WORK_ITEM void
iree_hal_amdgpu_device_queue_scheduler_tick(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    iree_hal_amdgpu_device_queue_scheduling_reason_t reason,
    uint64_t reason_arg) {
  // DO NOT SUBMIT
  //
  // if reason is COMMAND_BUFFER_RETURN:
  //   check reason_arg against execution state and cleanup
  //
  // dequeue work from queue and try to run it?
  // enqueue as much as possible?
  // need to chain signals

  // who owns kernargs?
  // ringbuffer? or always as part of queue operation?
  // can't mix kernarg region with non-kernarg region
  // command buffer return could provide as part of its storage
  // could be per-execution-queue (simultaneous command buffer count)
  // if only one scheduler tick can be pending at a time could be reused
  // need fancy atomics?
  //   if (atomic inc scheduler_request_pending == 0) {
  //     none was pending
  //     update kernargs
  //     enqueue
  //   }
  //   on tick: atomic dec scheduler_request_pending
  //
  // then reason needs to be an atomic bitmask? request pending could be
  // atomic OR the reason for scheduling
  // is reason needed?
  //
  // scheduler could poke execution state of all running
  // then could use static kernargs: each execution uses the same scheduler ptr
  // command buffer return could just be a bit indicating that it should be
  // checked in the next schedule run ("an execution completed")
  //
  // chaining signals? completion signal assigned by scheduler
  // command buffer return signals
  // do we need a barrier command at RETURN?
  // barrier command could use completion signal
  //
  // NEED AQL PACKET THEN!
}

void iree_hal_amdgpu_device_queue_scheduler_enqueue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    iree_hal_amdgpu_device_queue_scheduling_reason_t reason,
    uint64_t reason_arg) {
  // Pass the reason.
  IREE_OCL_GLOBAL void* control_kernargs[3] = {
      scheduler,
      reason,
      reason_arg,
  };
  // DO NOT SUBMIT implicit opencl args
  IREE_OCL_GLOBAL void* control_kernarg_ptr = state->control_kernarg_storage;
  memcpy(kernarg_ptr, control_kernargs, sizeof(control_kernargs));

  // Construct the control packet.
  // Note that the header is not written until the end so that the
  // hardware command processor stalls until we're done writing.
  const iree_hal_amdgpu_device_kernel_args_t tick_args =
      scheduler->kernels.scheduler_tick;
  // DO NOT SUBMIT queue_index
  IREE_OCL_GLOBAL hsa_kernel_dispatch_packet_t* tick_packet = NULL;
  tick_packet->setup = control_args.setup;
  tick_packet->workgroup_size_x = control_args.workgroup_size_x;
  tick_packet->workgroup_size_y = control_args.workgroup_size_y;
  tick_packet->workgroup_size_z = control_args.workgroup_size_z;
  tick_packet->reserved0 = 0;
  tick_packet->grid_size_x = 1;
  tick_packet->grid_size_y = 1;
  tick_packet->grid_size_z = 1;
  tick_packet->private_segment_size = control_args.private_segment_size;
  tick_packet->group_segment_size = control_args.kernel_args.group_segment_size;
  tick_packet->kernel_object = control_args.kernel_object;
  tick_packet->kernarg_address = kernarg_ptr;
  tick_packet->reserved2 = 0;

  // DO NOT SUBMIT
  // tick_packet->completion_signal = ;

  // NOTE: we implicitly assume
  // IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER and may always want to
  // do that - though _technically_ we should be processing all commands
  // submitted in multiple command buffers as if they were submitted in a single
  // one the granularity is such that the 0.001% of potential concurrency is not
  // worth the risk. If we did want to allow command buffers to execute
  // concurrently we'd need to probably re-evaluate cross-command-buffer event
  // handles such that a signal in one could be used in another and that's
  // currently out of scope.

  // Mark the update packet as ready to execute. The hardware command processor
  // may begin executing it immediately after performing the atomic swap.
  //
  // DO NOT SUBMIT
  // tick_packet->header
  // barrier bit
  // fence bits type INVALID -> KERNEL_DISPATCH
}
