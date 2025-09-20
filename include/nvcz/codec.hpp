#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <cuda_runtime.h>
#include "nvcz/util.hpp"   // brings cuda_ck/nv_ck helpers

namespace nvcz {

enum class Algo : uint8_t { LZ4 = 0, GDEFLATE = 1, SNAPPY = 2, ZSTD = 3 };

struct Codec {
  virtual ~Codec() = default;

  // Optional: human-readable name
  virtual const char* name() const { return "nvcomp-codec"; }

  // Upper bound for compressed size for input of length n
  virtual size_t max_compressed_bound(size_t n) const = 0;

  // New device-size pattern:
  //  - Writes exact compressed size to device pointer d_comp_size
  //  - Copies *max-bound* output into dst via cudaMemcpyAsync inside impl
  //  - Does NOT synchronize the stream
  virtual void compress_with_stream(const uint8_t* src, size_t n,
                                    uint8_t* dst, cudaStream_t stream,
                                    size_t* d_comp_size) = 0;

  // Decompress (no sync; all async on 'stream')
  virtual void decompress_with_stream(const uint8_t* comp, size_t comp_n,
                                      uint8_t* dst, size_t raw_n,
                                      cudaStream_t stream) = 0;
};

// Global configuration for nvCOMP chunk size (default 64KB)
extern size_t g_nvcomp_chunk_size_kb;

// Central factory (implemented in src/codec_factory.cpp)
std::unique_ptr<Codec> make_codec(Algo, size_t nvcomp_chunk_size_kb = g_nvcomp_chunk_size_kb);

} // namespace nvcz
