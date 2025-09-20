#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace nvcz {

inline void cuda_ck(cudaError_t st, const char* what) {
  if (st != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s: %s\n", what, cudaGetErrorString(st));
    std::quick_exit(3);
  }
}

/** One pinned host buffer. */
struct PinnedBlock {
  uint8_t* ptr = nullptr;
  size_t   cap = 0;
  size_t   len = 0;  // valid bytes
};

/** A small ring of pinned buffers you reuse forever. */
struct PinnedRing {
  std::vector<PinnedBlock> blocks;
  size_t next = 0;

  void init(size_t n_blocks, size_t bytes_per_block) {
    blocks.resize(n_blocks);
    for (auto& b : blocks) {
      b.cap = bytes_per_block;
      b.len = 0;
      cuda_ck(cudaHostAlloc((void**)&b.ptr, b.cap, cudaHostAllocDefault), "cudaHostAlloc");
    }
  }
  /** Cycles through blocks in a ring. */
  PinnedBlock& acquire() {
    PinnedBlock& b = blocks[next];
    next = (next + 1) % blocks.size();
    return b;
  }
  /** Access a specific slot (often you’ll tie slot := (chunk_idx % RING_SLOTS)). */
  PinnedBlock& at(size_t slot) { return blocks[slot % blocks.size()]; }

  void destroy() {
    for (auto& b : blocks) if (b.ptr) cuda_ck(cudaFreeHost(b.ptr), "cudaFreeHost");
    blocks.clear();
    next = 0;
  }
};

} // namespace nvcz
