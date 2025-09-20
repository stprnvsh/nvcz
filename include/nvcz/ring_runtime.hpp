#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <map>
#include <cstdio>
#include <cstdlib>
#include "nvcz/pinned_ring.hpp"

namespace nvcz {

/** One device slot per stream. */
struct DevSlot {
  uint8_t* d_raw  = nullptr;   // input (raw for compress, comp for decompress)
  uint8_t* d_out  = nullptr;   // output (comp for compress, raw for decompress)
  size_t   d_out_cap = 0;
  cudaEvent_t done{};          // signals D2H finished for this slot
};

/** A CUDA stream with N slots + host rings attached. */
struct StreamCtx {
  cudaStream_t stream{};
  std::vector<DevSlot> slots;  // size = RING_SLOTS
  size_t next_slot = 0;        // round-robin slot selection
  PinnedRing h_in;             // pinned host input ring (H->D source)
  PinnedRing h_out;            // pinned host output ring (D->H dest)

  void init(int ring_slots, size_t raw_bytes, size_t out_bytes_cap) {
    cuda_ck(cudaStreamCreate(&stream), "mk stream");
    slots.resize(ring_slots);
    for (int s=0; s<ring_slots; ++s) {
      auto& sl = slots[s];
      cuda_ck(cudaMalloc(&sl.d_raw, raw_bytes), "cudaMalloc d_raw");
      sl.d_out_cap = out_bytes_cap;
      cuda_ck(cudaMalloc(&sl.d_out, sl.d_out_cap), "cudaMalloc d_out");
      cuda_ck(cudaEventCreateWithFlags(&sl.done, cudaEventDisableTiming), "event");
    }
    h_in.init(ring_slots,  raw_bytes);
    h_out.init(ring_slots, out_bytes_cap);
  }
  void destroy() {
    for (auto& sl : slots) {
      if (sl.done) cudaEventDestroy(sl.done);
      if (sl.d_raw) cudaFree(sl.d_raw);
      if (sl.d_out) cudaFree(sl.d_out);
    }
    slots.clear();
    h_in.destroy();
    h_out.destroy();
    if (stream) cudaStreamDestroy(stream);
    stream = {};
    next_slot = 0;
  }
  int acquire_slot() {
    int s = static_cast<int>(next_slot);
    next_slot = (next_slot + 1) % slots.size();
    return s;
  }
};

/**
 * A set of streams that lets you:
 *  - submit work per stream/slot (no sync),
 *  - track completion with events,
 *  - and write out results strictly in-order by global index
 *    without stalling other streams (event-based polling).
 *
 * You feed (index, stream, slot, raw_n, out_n, out_host_ptr), and
 * RingSet will only write when `index == next_to_emit` **and** its
 * event is already complete.
 */
struct RingSet {
  std::vector<StreamCtx> streams;
  // in-order writer state:
  uint64_t next_to_emit = 0;

  struct Pending {
    uint64_t index;
    int stream;
    int slot;
    size_t raw_n;
    size_t out_n;
  };
  std::map<uint64_t, Pending> pending; // keyed by index

  void init(int n_streams, int ring_slots, size_t raw_bytes, size_t out_bytes_cap) {
    streams.resize(n_streams);
    for (auto& st : streams) st.init(ring_slots, raw_bytes, out_bytes_cap);
    next_to_emit = 0;
    pending.clear();
  }
  void destroy() {
    for (auto& st : streams) st.destroy();
    streams.clear();
    pending.clear();
    next_to_emit = 0;
  }

  /**
   * After you enqueue the async pipeline on `streams[si]` using `slot`,
   * call `remember(index, si, slot, raw_n, out_n)` so the writer can
   * emit frames in order later.
   */
  void remember(uint64_t index, int si, int slot, size_t raw_n, size_t out_n) {
    pending.emplace(index, Pending{index, si, slot, raw_n, out_n});
  }

  /**
   * Try to emit any ready-in-order completions:
   *  - checks next_to_emit in `pending`
   *  - queries the slot's `done` event
   *  - if ready: writes (raw_n, out_n, data) using supplied writer lambdas
   * Returns number of chunks emitted.
   *
   * You provide:
   *   write_hdr(raw_n, out_n)  -> e.g. fwrite_exact(&raw_n,8); fwrite_exact(&out_n,8);
   *   write_body(ptr, n)       -> e.g. fwrite_exact(ptr, n);
   */
  template <class WriteHdr, class WriteBody>
  size_t try_emit_ready(WriteHdr write_hdr, WriteBody write_body) {
    size_t emitted = 0;
    while (true) {
      auto it = pending.find(next_to_emit);
      if (it == pending.end()) break;
      const auto& p = it->second;

      auto& st = streams[p.stream];
      auto& sl = st.slots[p.slot];
      // only emit when the stream finished D->H for this slot
      auto q = cudaEventQuery(sl.done);
      if (q == cudaErrorNotReady) break;
      cuda_ck(q, "event query");

      // Header + body (no sync needed; data is in host pinned buffer)
      write_hdr(p.raw_n, p.out_n);
      auto& hb = st.h_out.at(p.slot);
      write_body(hb.ptr, p.out_n);

      pending.erase(it);
      ++next_to_emit;
      ++emitted;
    }
    return emitted;
  }

  /** Force-flush any remaining chunks in order (blocking on events). */
  template <class WriteHdr, class WriteBody>
  void flush_all(WriteHdr write_hdr, WriteBody write_body) {
    while (!pending.empty()) {
      auto it = pending.find(next_to_emit);
      if (it == pending.end()) {
        // if gaps exist, wait for the smallest index we have
        it = pending.begin();
      }
      const auto& p = it->second;
      auto& st = streams[p.stream];
      auto& sl = st.slots[p.slot];
      cuda_ck(cudaEventSynchronize(sl.done), "event sync");

      write_hdr(p.raw_n, p.out_n);
      auto& hb = st.h_out.at(p.slot);
      write_body(hb.ptr, p.out_n);

      pending.erase(it);
      ++next_to_emit;
    }
  }
};

} // namespace nvcz
