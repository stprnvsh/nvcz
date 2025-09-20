// src/codec_zstd.cpp
#include "nvcz/codec.hpp"
#include "nvcz/util.hpp"

#include <memory>

#include <nvcomp/zstd.hpp>                  // nvcomp::ZstdManager
#include <nvcomp/nvcompManagerFactory.hpp>  // nvcomp::create_manager
#include <nvcomp.hpp>                       // CompressionConfig / DecompressionConfig, BitstreamKind, ChecksumPolicy

namespace nvcz {

struct NvcompZstd : Codec {
  const size_t chunk_size;

  NvcompZstd(size_t chunk_size_kb) : chunk_size(chunk_size_kb * 1024) {}
  const char* name() const override { return "nvcomp-zstd"; }

  // Ask the manager for a safe max compressed size for an input of length n.
  size_t max_compressed_bound(size_t n) const override {
    nvcompBatchedZstdOpts_t opts{};     // defaults

    nvcomp::ZstdManager mgr{
        chunk_size,
        opts,
        /*stream*/ 0,
        nvcomp::ChecksumPolicy::NoComputeNoVerify,
        nvcomp::BitstreamKind::NVCOMP_NATIVE};

    nvcomp::CompressionConfig cfg = mgr.configure_compression(n);
    return cfg.max_compressed_buffer_size;
  }

  // New contract:
  // - Writes exact compressed size to device pointer d_comp_size.
  // - Copies the *max-bound* payload back to host (dst). The caller will later
  //   write only the first 'comp_len' bytes after it reads a pinned size_t
  //   (post-event) that was D2H-copied separately.
  // - No stream syncs here.
  void compress_with_stream(const uint8_t* src, size_t n,
                            uint8_t* dst, cudaStream_t s,
                            size_t* d_comp_size) override
  {
    nvcompBatchedZstdOpts_t opts{};

    nvcomp::ZstdManager mgr{
        chunk_size,
        opts,
        s,
        nvcomp::ChecksumPolicy::NoComputeNoVerify,
        nvcomp::BitstreamKind::NVCOMP_NATIVE};

    nvcomp::CompressionConfig cfg = mgr.configure_compression(n);

    // Device staging buffers
    void* d_in  = nullptr;
    void* d_out = nullptr;
    cuda_ck(cudaMallocAsync(&d_in,  n, s), "zstd d_in");
    cuda_ck(cudaMallocAsync(&d_out, cfg.max_compressed_buffer_size, s), "zstd d_out");

    // H2D input
    cuda_ck(cudaMemcpyAsync(d_in, src, n, cudaMemcpyHostToDevice, s), "zstd H2D");

    // Compress (nvCOMP writes exact size to d_comp_size on device)
    mgr.compress(
        static_cast<const uint8_t*>(d_in),
        static_cast<uint8_t*>(d_out),
        cfg,
        d_comp_size);

    // Copy the max-bound output payload back to host. Writer will emit only the
    // first 'comp_len' bytes after it reads the pinned size (post-event).
    cuda_ck(cudaMemcpyAsync(
        dst,
        d_out,
        cfg.max_compressed_buffer_size,
        cudaMemcpyDeviceToHost,
        s), "zstd D2H (max-bound)");

    // Free staging buffers when the stream reaches these ops
    cuda_ck(cudaFreeAsync(d_in,  s), "zstd free d_in");
    cuda_ck(cudaFreeAsync(d_out, s), "zstd free d_out");
  }

  // Decompress on the provided CUDA stream (no syncs here).
  void decompress_with_stream(const uint8_t* comp, size_t comp_n,
                              uint8_t* dst, size_t raw_n,
                              cudaStream_t s) override
  {
    // Copy compressed data to device
    void* d_in = nullptr;
    cuda_ck(cudaMallocAsync(&d_in, comp_n, s), "zstd d_in (comp)");
    cuda_ck(cudaMemcpyAsync(d_in, comp, comp_n, cudaMemcpyHostToDevice, s), "zstd H2D comp");

    // Create a manager from the device buffer header
    std::shared_ptr<nvcomp::nvcompManagerBase> mgr =
        nvcomp::create_manager(static_cast<const uint8_t*>(d_in), s);

    // Configure decompression (comp_size optional for NVCOMP_NATIVE)
    nvcomp::DecompressionConfig dcfg =
        mgr->configure_decompression(static_cast<const uint8_t*>(d_in), /*comp_size*/ nullptr);

    // Device output
    void* d_out = nullptr;
    cuda_ck(cudaMallocAsync(&d_out, raw_n, s), "zstd d_out (raw)");

    // Decompress
    mgr->decompress(
        static_cast<uint8_t*>(d_out),
        static_cast<const uint8_t*>(d_in),
        dcfg,
        /*comp_size*/ nullptr);

    // D2H payload
    cuda_ck(cudaMemcpyAsync(dst, d_out, raw_n, cudaMemcpyDeviceToHost, s), "zstd D2H raw");

    // Free when the stream reaches these ops
    cuda_ck(cudaFreeAsync(d_in,  s), "zstd free d_in");
    cuda_ck(cudaFreeAsync(d_out, s), "zstd free d_out");
  }
};

std::unique_ptr<Codec> make_codec_zstd(size_t chunk_size_kb) { return std::make_unique<NvcompZstd>(chunk_size_kb); }

} // namespace nvcz
