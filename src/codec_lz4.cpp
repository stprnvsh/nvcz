// src/codec_lz4.cpp
#include "nvcz/codec.hpp"
#include "nvcz/util.hpp"

#include <memory>

#include <nvcomp/lz4.hpp>                    // nvcomp::LZ4Manager
#include <nvcomp/nvcompManagerFactory.hpp>   // nvcomp::create_manager
#include <nvcomp.hpp>                        // CompressionConfig / DecompressionConfig, BitstreamKind, ChecksumPolicy

namespace nvcz {

struct NvcompLZ4 : Codec {
  const size_t chunk_size;
  const bool enable_checksum;

  NvcompLZ4(size_t chunk_size_kb, bool enable_checksum)
    : chunk_size(chunk_size_kb * 1024), enable_checksum(enable_checksum) {}
  const char* name() const override { return enable_checksum ? "nvcomp-lz4-checksum" : "nvcomp-lz4"; }

  // Ask the manager for a tight bound for an input of length n.
  size_t max_compressed_bound(size_t n) const override {
    nvcompBatchedLZ4Opts_t opts{};      // defaults

    nvcomp::ChecksumPolicy checksum_policy = enable_checksum ?
        nvcomp::ChecksumPolicy::ComputeAndVerify :
        nvcomp::ChecksumPolicy::NoComputeNoVerify;

    nvcomp::LZ4Manager mgr{
        chunk_size,
        opts,
        /*stream*/ 0,
        checksum_policy,
        nvcomp::BitstreamKind::NVCOMP_NATIVE};

    nvcomp::CompressionConfig cfg = mgr.configure_compression(n);
    return cfg.max_compressed_buffer_size;
  }

  // New pattern:
  //  - d_comp_size: device pointer (size_t*) provided by the caller (worker)
  //  - We DO NOT sync the stream
  //  - We copy the max-bound compressed output to host 'dst' (D2H) so it's ready
  //    when the stream event fires; writer will later read the actual size
  //    from the pinned host size_t it copied asynchronously from d_comp_size.
  void compress_with_stream(const uint8_t* src, size_t n,
                            uint8_t* dst, cudaStream_t s,
                            size_t* d_comp_size) override {
    nvcompBatchedLZ4Opts_t opts{};

    nvcomp::ChecksumPolicy checksum_policy = enable_checksum ?
        nvcomp::ChecksumPolicy::ComputeAndVerify :
        nvcomp::ChecksumPolicy::NoComputeNoVerify;

    nvcomp::LZ4Manager mgr{
        chunk_size,
        opts,
        s,
        checksum_policy,
        nvcomp::BitstreamKind::NVCOMP_NATIVE};

    // Sizing for this input
    nvcomp::CompressionConfig cfg = mgr.configure_compression(n);

    // Device staging buffers
    void* d_in  = nullptr;
    void* d_out = nullptr;
    cuda_ck(cudaMallocAsync(&d_in,  n, s), "lz4 d_in");
    cuda_ck(cudaMallocAsync(&d_out, cfg.max_compressed_buffer_size, s), "lz4 d_out");

    // H2D input
    cuda_ck(cudaMemcpyAsync(d_in, src, n, cudaMemcpyHostToDevice, s), "lz4 H2D");

    // Compress (writes exact size into device pointer d_comp_size)
    mgr.compress(static_cast<const uint8_t*>(d_in),
                 static_cast<uint8_t*>(d_out),
                 cfg,
                 d_comp_size);

    // Stage the whole bound-sized output on host; writer will trim to true size.
    cuda_ck(cudaMemcpyAsync(dst, d_out, cfg.max_compressed_buffer_size,
                            cudaMemcpyDeviceToHost, s),
            "lz4 D2H bound");

    // Deferred frees on the same stream (no sync)
    cuda_ck(cudaFreeAsync(d_in,  s), "lz4 free d_in");
    cuda_ck(cudaFreeAsync(d_out, s), "lz4 free d_out");
  }

  // Decompression (no sync; all async on stream)
  void decompress_with_stream(const uint8_t* comp, size_t comp_n,
                              uint8_t* dst, size_t raw_n,
                              cudaStream_t s) override {
    // Copy compressed bytes to device
    void* d_in = nullptr;
    cuda_ck(cudaMallocAsync(&d_in, comp_n, s), "lz4 d_in (comp)");
    cuda_ck(cudaMemcpyAsync(d_in, comp, comp_n, cudaMemcpyHostToDevice, s), "lz4 H2D comp");

    // Manager from header in device memory (auto-selects correct manager)
    std::shared_ptr<nvcomp::nvcompManagerBase> mgr =
        nvcomp::create_manager(static_cast<const uint8_t*>(d_in), s);

    // Configure from the bitstream
    nvcomp::DecompressionConfig dcfg =
        mgr->configure_decompression(static_cast<const uint8_t*>(d_in), /*comp_size*/ nullptr);

    // Device output
    void* d_out = nullptr;
    cuda_ck(cudaMallocAsync(&d_out, raw_n, s), "lz4 d_out (raw)");

    // Decompress
    mgr->decompress(static_cast<uint8_t*>(d_out),
                    static_cast<const uint8_t*>(d_in),
                    dcfg,
                    /*comp_size*/ nullptr);

    // D2H raw
    cuda_ck(cudaMemcpyAsync(dst, d_out, raw_n, cudaMemcpyDeviceToHost, s), "lz4 D2H raw");

    // Deferred frees
    cuda_ck(cudaFreeAsync(d_in,  s), "lz4 free d_in");
    cuda_ck(cudaFreeAsync(d_out, s), "lz4 free d_out");
  }
};

std::unique_ptr<Codec> make_codec_lz4(size_t chunk_size_kb, bool enable_checksum) { return std::make_unique<NvcompLZ4>(chunk_size_kb, enable_checksum); }

} // namespace nvcz
