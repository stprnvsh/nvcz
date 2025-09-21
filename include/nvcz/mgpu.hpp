#pragma once
#include <cstdint>
#include <vector>
#include <string>

#include <cuda_runtime.h>

#include "nvcz/codec.hpp"
#include "nvcz/autotune.hpp"
#include "nvcz/framing.hpp"
#include "nvcz/util.hpp"

namespace nvcz {

// Which GPUs and how much parallelism to use.
struct MgpuTune {
  std::vector<int> gpu_ids;      // e.g., {0,1}
  int streams_per_gpu = 2;       // per-GPU CUDA streams (>=1)
  uint32_t chunk_mb   = 32;      // must match framing's chunk size
  bool size_aware     = false;   // allow bumping streams for very large regular files
};

// Discover GPUs (ids only) from CUDA_VISIBLE_DEVICES or system
std::vector<int> discover_gpus_ids();

// Build MgpuTune from AutoTune + available GPUs
MgpuTune pick_mgpu_tuning(const AutoTune& base, bool size_aware, int override_streams_per_gpu,
                          const std::vector<int>& gpu_ids_override);

// Compress stdin -> stdout in-order using multiple GPUs (same v1 framing)
void compress_mgpu(Algo algo, const MgpuTune& t, FILE* input_fp, FILE* output_fp, bool show_progress);

// Compress using GPUDirect Storage for multi-GPU (reads directly into GPU)
void compress_mgpu_gds(Algo algo, const MgpuTune& t, FILE* input_fp, FILE* output_fp, bool show_progress);

// Decompress stdin -> stdout in-order using multiple GPUs
void decompress_mgpu(const MgpuTune& t, FILE* input_fp, FILE* output_fp, bool show_progress);

// (Optional) Decompress using GPUDirect Storage (to be implemented later)
// void decompress_mgpu_gds(const MgpuTune& t, FILE* input_fp, FILE* output_fp, bool show_progress);

} // namespace nvcz
